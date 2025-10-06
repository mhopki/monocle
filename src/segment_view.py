#!/usr/bin/env python3

# ROS 1 Python Node for Visualizing Odometry Orientation and Velocity

import rospy
import cv2
import numpy as np
import math
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

class OdomVisualizer:
    """
    Subscribes to the aligned depth image, the /tracked_odom, and CameraInfo.
    Projects the object's 3D pose back into 2D to draw a scaled red arrow 
    showing the object's instantaneous velocity vector.
    """

    def __init__(self):
        rospy.init_node('odom_visualizer_node', anonymous=True)
        self.bridge = CvBridge()
        
        # --- Configuration ---
        self.max_arrow_length = 50      # Max pixel length of the arrow head
        self.scale_factor = 75          # Multiplier for speed (m/s) to pixel length
        
        # Adjustable Parameter for correcting visual offset
        # A value of -math.pi / 2 (-90 degrees) or math.pi / 2 (90 degrees) often fixes the visual rotation.
        self.YAW_OFFSET = rospy.get_param('~yaw_offset', -math.pi / 2.0) 
        
        # Camera Intrinsics 
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.intrinsics_ready = False
        
        # Publishers
        self.vis_pub = rospy.Publisher("/odom_visualization", Image, queue_size=1)

        # --- Subscribers ---
        # 1. Camera Info (for calibration parameters)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self._camera_info_callback, queue_size=1)

        # 2. Aligned Depth Image (for the visual background)
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        # 3. Odometry Message (for tracking state)
        odom_sub = message_filters.Subscriber("/tracked_odom", Odometry)
        
        # Synchronize these two topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, odom_sub], 
            queue_size=10, 
            slop=0.05 
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        rospy.loginfo("Odometry Visualizer initialized. Waiting for CameraInfo...")
        rospy.loginfo(f"Applying initial Yaw Offset: {math.degrees(self.YAW_OFFSET):.1f} degrees.")


    def _camera_info_callback(self, info_msg):
        """Extracts camera intrinsic matrix (K) from CameraInfo message."""
        if not self.intrinsics_ready:
            try:
                # K matrix elements (0, 4, 2, 5 are fx, fy, cx, cy)
                self.fx = info_msg.K[0]
                self.fy = info_msg.K[4]
                self.cx = info_msg.K[2]
                self.cy = info_msg.K[5]
                self.intrinsics_ready = True
                rospy.loginfo(f"Intrinsics loaded: FX={self.fx:.2f}, FY={self.fy:.2f}, CX={self.cx:.2f}, CY={self.cy:.2f}")
            except IndexError:
                rospy.logerr("CameraInfo message is incomplete. Cannot extract intrinsics.")
        
    def convert_to_32fc1_meters(self, img_msg, encoding):
        """Converts depth image to 32FC1 (meters) and returns 8-bit BGR for drawing."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, encoding)
            depth_array = np.array(cv_image, dtype=np.float32)

            if encoding == "16UC1":
                depth_array /= 1000.0
            
            # Normalize and convert to 8-bit BGR for drawing 
            min_depth_vis = np.nanmin(depth_array[depth_array > 0]) if np.sum(depth_array > 0) > 0 else 0
            max_depth_vis = np.nanmax(depth_array) if np.sum(depth_array > 0) > 0 else 1

            if max_depth_vis > min_depth_vis:
                normalized_depth = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            else:
                normalized_depth = np.zeros_like(depth_array, dtype=np.uint8)
                
            return cv2.cvtColor(normalized_depth, cv2.COLOR_GRAY2BGR)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return None

    def quaternion_to_yaw(self, q):
        """Converts geometry_msgs/Quaternion to yaw (rotation about Z axis)."""
        # atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))

    def _project_3d_to_2d(self, X, Y, Z):
        """Projects 3D camera coordinates (X,Y,Z) to 2D pixel coordinates (u, v)."""
        if Z <= 0.0 or not self.intrinsics_ready: # Check Z > 0
            return None, None
        
        # Pinhole camera model: u = (X/Z) * Fx + Cx; v = (Y/Z) * Fy + Cy
        u = (X / Z) * self.fx + self.cx
        v = (Y / Z) * self.fy + self.cy
        
        return int(u), int(v)


    def synchronized_callback(self, depth_data, odom_msg):
        """
        Processes synchronized depth and odometry data, draws the odometry vector.
        """
        if not self.intrinsics_ready:
            rospy.logwarn_throttle(2.0, "Waiting for camera intrinsics before drawing odometry.")
            return

        # Convert depth to drawable BGR image
        drawable_cv = self.convert_to_32fc1_meters(depth_data, depth_data.encoding)
        if drawable_cv is None:
            return
        
        # --- 1. Reproject Odometry Position to Pixel Center ---
        X = odom_msg.pose.pose.position.x
        Y = odom_msg.pose.pose.position.y
        Z = odom_msg.pose.pose.position.z

        center_x, center_y = self._project_3d_to_2d(X, Y, Z)

        if center_x is None:
             rospy.logwarn_throttle(2.0, "Object position has zero depth (Z=0.0). Cannot project to pixel location.")
             return
        
        # --- 2. Extract Odometry Velocity ---
        vx = odom_msg.twist.twist.linear.x
        vy = odom_msg.twist.twist.linear.y
        
        # Linear velocity magnitude (speed)
        speed = math.sqrt(vx**2 + vy**2)
        
        # Calculate the YAW angle from the Odometry's orientation quaternion
        odom_yaw = self.quaternion_to_yaw(odom_msg.pose.pose.orientation)

        # 3. Calculate Arrow Endpoint based on Yaw and Scaled Speed
        
        # Apply the visual offset correction to the Yaw angle
        visual_yaw = odom_yaw + self.YAW_OFFSET
        
        # Clamp velocity magnitude to prevent excessive arrow length
        clamped_speed = min(speed, self.max_arrow_length / self.scale_factor)
        
        # Arrow length is based on the speed magnitude
        arrow_length = clamped_speed * self.scale_factor
        
        # Calculate the end point based on the corrected YAW
        # X component: cos(yaw). Y component: sin(yaw).
        # We use the corrected visual_yaw here.
        end_x = int(center_x + arrow_length * math.cos(visual_yaw))
        # NOTE: Screen Y axis points DOWN, so we SUBTRACT sin(visual_yaw) to draw it correctly.
        end_y = int(center_y - arrow_length * math.sin(visual_yaw))
        
        # --- 4. Draw Visualization ---
        start_point = (center_x, center_y)
        end_point = (end_x, end_y)
        color = (0, 0, 255) # Red (BGR)
        thickness = 3

        # Draw the arrow showing the object's forward direction/velocity
        cv2.arrowedLine(drawable_cv, start_point, end_point, color, thickness, tipLength=0.2)
        cv2.circle(drawable_cv, start_point, 5, (0, 255, 0), -1) # Green dot at object center
        
        # Add text for speed
        H, W, _ = drawable_cv.shape
        cv2.putText(
            drawable_cv,
            f"Speed: {speed:.2f} m/s | Yaw: {math.degrees(odom_yaw):.1f} deg",
            (20, H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255), # White text
            2
        )

        # Publish the visualized image
        vis_msg = self.bridge.cv2_to_imgmsg(drawable_cv, "bgr8")
        vis_msg.header = depth_data.header
        self.vis_pub.publish(vis_msg)

if __name__ == '__main__':
    try:
        node = OdomVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
