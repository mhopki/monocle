#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import math
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

class Visualizer:
    def __init__(self):
        rospy.init_node("motion_visualizer")

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.odom_sub = rospy.Subscriber("/object/odom", Odometry, self.odom_callback)
        self.cmd_sub = rospy.Subscriber("/cmd_vel", Twist, self.cmd_callback)
        self.origin_sub = rospy.Subscriber("/object/origin", Point, self.origin_callback)
        self.scale_sub = rospy.Subscriber("/object/scale", Float32, self.scale_callback)

        self.image_pub = rospy.Publisher("/visualization/image", Image, queue_size=1)

        self.latest_image = None
        self.latest_odom = None
        self.latest_cmd = None

        # Set goal for visualization (in map frame)
        self.goal_x = rospy.get_param("goal_x")
        self.goal_y = rospy.get_param("goal_y")

        # Camera visualization assumptions
        self.origin = (0,0,0)
        self.scale = rospy.get_param("~scale", 0.01)  # meters per pixel
        self.image_width = 1280
        self.image_height = 800

        rospy.loginfo("Motion visualizer started.")

    def scale_callback(self, msg):
        self.scale = msg.data

    def origin_callback(self, msg):
        self.origin = (msg.x,msg.y,msg.z)
        #print(msg)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.visualize()
        except Exception as e:
            rospy.logwarn(f"Image conversion failed: {e}")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # Extract yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.latest_odom = (pos.x, pos.y, yaw)

    def cmd_callback(self, msg):
        self.latest_cmd = (msg.linear.x, msg.angular.z)

    def visualize(self):
        if self.latest_image is None or self.latest_odom is None or self.latest_cmd is None:
            return

        print(self.goal_x, self.goal_y)
        self.goal_x = rospy.get_param("goal_x")
        self.goal_y = rospy.get_param("goal_y")

        img = self.latest_image.copy()
        x, y, yaw = self.latest_odom
        lin_vel, ang_vel = self.latest_cmd

        # Compute image-space center from odom position
        cx = int((x / self.scale) + self.origin[0])
        cy = int((y / self.scale) + self.origin[1])

        # Draw goal point
        gx = int((self.goal_x / self.scale) + self.origin[0])
        gy = int((self.goal_y / self.scale) + self.origin[1])
        cv2.circle(img, (gx, gy), 10, (0, 255, 255), -1)  # yellow

        # Draw command velocity arrow (blue)
        arrow_length = 50
        dx = int(arrow_length * math.cos(yaw + ang_vel * 0.5))
        dy = int(arrow_length * math.sin(yaw + ang_vel * 0.5))
        end_cmd = (cx + dx, cy + dy)
        cv2.arrowedLine(img, (cx, cy), end_cmd, (255, 0, 0), 4)  # blue

        # Draw current yaw arrow (red)
        dx_yaw = int(arrow_length * math.cos(yaw))
        dy_yaw = int(arrow_length * math.sin(yaw))
        end_yaw = (cx + dx_yaw, cy + dy_yaw)
        cv2.arrowedLine(img, (cx, cy), end_yaw, (0, 0, 255), 4)  # red

        # Annotate
        cv2.putText(img, "GOAL", (gx + 10, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, "CMD DIR", (end_cmd[0] + 10, end_cmd[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, "YAW DIR", (end_yaw[0] + 10, end_yaw[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Publish
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self.image_pub.publish(img_msg)
        except Exception as e:
            rospy.logwarn(f"Image publish failed: {e}")


    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = Visualizer()
        node.spin()
    except rospy.ROSInterruptException:
        pass
