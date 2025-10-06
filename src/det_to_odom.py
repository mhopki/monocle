#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tf
import math

class PixelOdometry:
    def __init__(self):
        rospy.init_node('pixel_odometry_node')
        
        self.scale = rospy.get_param("~scale", 0.01)  # meters per pixel
        self.pos = [0.0, 0.0]
        self.yaw = 0.0
        self.last_px = None
        self.last_yaw = None
        self.last_time = rospy.Time.now()
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_v_yaw = 0.0
        self.filter_alpha = rospy.get_param("~filter_alpha", 0.7)  # smoothing factor
        self.speed_threshold = rospy.get_param("~speed_threshold", 0.05)  # meters per second

        self.detection_buffer = []  # To store first 10 detections
        self.origin_px = None       # Will hold the average of first 10 detections
        self.map_pose = [0.0, 0.0]  # Objective position in meters relative to image origin

        self.bridge = CvBridge()
        self.scale_initialized = False

        self.image_width = 1280.0
        self.image_height = 800.0
        self.hfov_deg = 87.0  # Horizontal FOV
        self.vfov_deg = 58.0  # Vertical FOV

        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)


        self.odom_pub = rospy.Publisher('/object/odom', Odometry, queue_size=1)
        self.scale_pub = rospy.Publisher("/object/scale", Float32, queue_size=1)  # <-- NEW
        self.origin_pub = rospy.Publisher("/object/origin", Point, queue_size=1)  # <-- NEW
        self.map_pose_pub = rospy.Publisher("/object/map_pose", Pose2D, queue_size=1)  # <-- NEW

        rospy.Subscriber('/yolov7/yolov7', Detection2DArray, self.pixel_callback)

        rospy.loginfo("PixelOdometry node started.")

    def depth_callback(self, msg):
        if self.scale_initialized:
            self.scale_pub.publish(Float32(self.scale))
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logwarn(f"Depth image conversion failed: {e}")
            return

        # Convert to numpy float32 (if needed)
        depth_array = np.array(depth_image, dtype=np.float32)

        rospy.logwarn(np.mean(depth_array))

        # Mask out invalid depths (zero or NaN)
        valid_depths = depth_array[np.isfinite(depth_array) & (depth_array > 0.1) & (depth_array < 5000)]

        if valid_depths.size == 0:
            rospy.logwarn("No valid depth values found.")
            return

        floor_depth = np.max(valid_depths) / 1000 # Furthest valid depth = floor
        rospy.loginfo(f"Estimated floor depth: {floor_depth:.2f} meters")

        # Compute horizontal field of view in radians
        hfov_rad = math.radians(self.hfov_deg)
        visible_width = 2.0 * floor_depth * math.tan(hfov_rad / 2.0)

        # Now compute meters-per-pixel
        self.scale = visible_width / self.image_width
        self.scale_initialized = True

        rospy.loginfo(f"Scale initialized: {self.scale:.6f} meters/pixel")
        self.scale_pub.publish(Float32(self.scale))

        rospy.loginfo(f"Scale initialized: {self.scale:.6f} meters/pixel")


    def pixel_callback(self, data):
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        if dt == 0:
            return

        if not data.detections:
            return  # No detections to process

        msg = data.detections[0].bbox.center
        x = msg.x
        y = msg.y

        # Publish map pose as Pose2D
        pose_msg = Pose2D()
        pose_msg.x = self.map_pose[0]
        pose_msg.y = self.map_pose[1]
        pose_msg.theta = self.yaw
        self.map_pose_pub.publish(pose_msg) 

        if self.origin_px is None:
            self.detection_buffer.append((x, y))
            if len(self.detection_buffer) == 10:
                avg_x = sum(p[0] for p in self.detection_buffer) / 10.0
                avg_y = sum(p[1] for p in self.detection_buffer) / 10.0
                self.origin_px = (avg_x, avg_y)
                rospy.loginfo(f"Initialized origin_px at: {self.origin_px}")
            return  # Wait until we initialize origin_px

        if not self.scale_initialized:
            rospy.loginfo("Waiting for scale initialization from depth image...")
            return

        origin_msg = Point()
        origin_msg.x = self.origin_px[0]
        origin_msg.y = self.origin_px[1]
        origin_msg.z = 0.0
        self.origin_pub.publish(origin_msg)

        # Step 2: Compute objective map position
        self.map_pose[0] = (x) * self.scale
        self.map_pose[1] = (y) * self.scale

        #rospy.loginfo(f"Map Pose (Pixel Position): {(x,y)}")
        #rospy.loginfo(f"Map Pose (Objective Position): {self.map_pose}")
        #rospy.loginfo(f"Robot Pose (Relative Position): {self.pos}")

        if self.last_px is not None:
            dx_px = x - self.last_px[0]
            dy_px = y - self.last_px[1]

            dx = dx_px * self.scale
            dy = dy_px * self.scale

            # Update world position
            self.pos[0] += dx
            self.pos[1] += dy

            # Linear velocity (raw)
            vx_raw = dx / dt
            vy_raw = dy / dt

            # Filtered linear velocity
            self.filtered_vx = self.filter_alpha * vx_raw + (1 - self.filter_alpha) * self.filtered_vx
            self.filtered_vy = self.filter_alpha * vy_raw + (1 - self.filter_alpha) * self.filtered_vy

            # Angular velocity
            v_yaw_raw = 0.0
            if self.last_yaw is not None:
                # Orientation (yaw)
                # Speed check (threshold)
                speed = math.sqrt(dx**2 + dy**2)

                # Only update yaw if speed is above threshold
                if speed > self.speed_threshold:
                    new_yaw = math.atan2(dy_px, dx_px)

                    # Normalize the yaw between -pi and pi
                    dyaw = new_yaw - self.yaw
                    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

                    # Apply the new yaw
                    #self.yaw += dyaw
                    self.yaw = new_yaw

                    # Wrap the yaw to make sure it's always between -pi and pi
                    self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi


            self.filtered_v_yaw = self.filter_alpha * v_yaw_raw + (1 - self.filter_alpha) * self.filtered_v_yaw

            self.last_yaw = self.yaw

            # Quaternion from yaw
            q = tf.transformations.quaternion_from_euler(0, 0, self.yaw)

            # Odometry message
            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"

            odom.pose.pose.position.x = self.pos[0]
            odom.pose.pose.position.y = self.pos[1]
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation = Quaternion(*q)

            odom.twist.twist = Twist(Vector3(self.filtered_vx, self.filtered_vy, 0), Vector3(0, 0, self.filtered_v_yaw))
            self.odom_pub.publish(odom)

            # Print yaw in degrees
            yaw_deg = math.degrees(self.yaw)
            rospy.loginfo(f"Yaw: {math.degrees(self.yaw):.2f}°, Angular Velocity: {math.degrees(self.filtered_v_yaw):.2f}°/s")


        self.last_px = (x, y)
        self.last_time = now

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PixelOdometry()
        node.spin()
    except rospy.ROSInterruptException:
        pass
