#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import atan2, sqrt, pi
import random
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

class ConstantVelocityController:
    def __init__(self):
        rospy.init_node("constant_velocity_controller")

        self.goal_x = rospy.set_param("goal_x", 0.0)
        self.goal_y = rospy.set_param("goal_y", 0.0)

        self.linear_speed = rospy.get_param("~linear_speed", 0.2)
        self.angular_kp = rospy.get_param("~angular_kp", 2.0)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 1.0)

        self.distance_tolerance = rospy.get_param("~distance_tolerance", 0.3)
        self.angle_tolerance = rospy.get_param("~angle_tolerance", pi / 12)  # 15 degrees

        self.pose = None
        self.scale = None
        self.origin = None
        self.goal_initialized = False

        rospy.Subscriber("/object/odom", Odometry, self.odom_callback)
        self.origin_sub = rospy.Subscriber("/object/origin", Point, self.origin_callback)
        self.scale_sub = rospy.Subscriber("/object/scale", Float32, self.scale_callback)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.loginfo("Constant velocity controller started.")

    def odom_callback(self, msg):
        # Get position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Get orientation as yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = atan2(siny_cosp, cosy_cosp)

        self.pose = (x, y, theta)
        self.control()

    def scale_callback(self, msg):
        self.scale = msg.data

    def origin_callback(self, msg):
        self.origin = (msg.x,msg.y,msg.z)
        #print(msg)

    def control(self):
        if self.pose is None:
            return

        if self.scale is None or self.origin is None:
            return

        if self.goal_initialized is False:
            self.goal_initialized = True
            self.goal_x = 1.4#(random.uniform(1.5, 3.0))# / self.scale)
            self.goal_y = 1.3#(random.uniform(0.5, 1.8))# / self.scale)
            rospy.set_param("goal_x", self.goal_x)
            rospy.set_param("goal_y", self.goal_y)
            print("GOAL: ", self.goal_x, self.goal_y)

        x, y, theta = self.pose
        print("ME: ", x, y)

        dx = self.goal_x - x
        dy = self.goal_y - y
        distance = sqrt(dx**2 + dy**2)
        target_theta = atan2(dy, dx)

        angle_error = (target_theta - theta + pi) % (2 * pi) - pi  # wrap to [-pi, pi]

        cmd = Twist()

        if distance < self.distance_tolerance:
            rospy.loginfo("Goal reached.")
            self.cmd_pub.publish(cmd)  # publish zero
            return

        # Always drive forward
        cmd.linear.x = self.linear_speed

        # Add steering via angular.z, proportional to heading error
        cmd.angular.z = max(-self.max_angular_speed,
                            min(self.angular_kp * angle_error, self.max_angular_speed))

        # Motion intent (for logging)
        if abs(cmd.angular.z) < 0.1:
            action = "Going forward"
        elif cmd.angular.z > 0.1:
            action = "Going forward and turning right"
        elif cmd.angular.z < -0.1:
            action = "Going forward and turning left"
        else:
            action = "Going forward (unusual angular)"

        # Additional log output
        #rospy.loginfo(f"Current position: x={x:.2f}, y={y:.2f}")
        #rospy.loginfo(f"Goal position:    x={self.goal_x:.2f}, y={self.goal_y:.2f}")
        #rospy.loginfo(f"Current yaw:      {theta:.2f} rad ({theta * 180 / pi:.1f} deg)")
        #rospy.loginfo(f"Target yaw:       {target_theta:.2f} rad ({target_theta * 180 / pi:.1f} deg)")

        rospy.loginfo(
            f"[Action: {action}] Dist: {distance:.2f} | Angle err: {angle_error:.2f} | "
            f"Linear: {cmd.linear.x:.2f} | Angular: {cmd.angular.z:.2f}"
        )

        self.cmd_pub.publish(cmd)



    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = ConstantVelocityController()
        node.spin()
    except rospy.ROSInterruptException:
        pass
