#!/usr/bin/env python3

# ROS 1 Python Node for 3D Odometry Calculation from 2D Pixel Tracking

import rospy
import numpy as np
import math
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from cv_bridge import CvBridge, CvBridgeError

class PixelOdometryNode:
    """
    Subscribes to tracking data and calculates smoothed 3D Odometry using a Kalman Filter.
    The KF state is [X, Y, Z, Vx, Vy, Vz].
    The measurement is [X, Y, Z].
    
    This version includes Adaptive Process Noise and Predictive Time Step for reduced lag.
    """

    MAX_DEPTH_VALUE = 1000.0
    
    def __init__(self):
        rospy.init_node('pixel_odometry_node', anonymous=True)
        self.bridge = CvBridge()

        # --- Intrinsics State ---
        self.fx, self.fy, self.cx, self.cy = 0.0, 0.0, 0.0, 0.0
        self.intrinsics_ready = False
        self.CAMERA_FRAME_ID = "base_link" 
        
        # --- Tracking State ---
        self.current_pixel = None 
        self.last_pose = None
        self.last_time = None
        
        # --- Kalman Filter State ---
        self.kf_initialized = False
        
        # FIX 1: Update nominal dt to 1/30 (0.0333...)
        self.dt = rospy.get_param('~nominal_dt', 1.0/30.0) 
        
        # FIX 2: Predictive time step is now 1/30 second
        self.PREDICTIVE_TIME_STEP = rospy.get_param('~predictive_dt', self.dt * 1.0) 
        
        self._initialize_kalman_filter()

        # --- Publishers ---
        self.odom_pub = rospy.Publisher("/tracked_odom", Odometry, queue_size=1) 

        # --- Subscribers ---
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber("/tracked_pixel_location", Point, self.pixel_location_callback, queue_size=1)
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback, queue_size=1)
        
        rospy.loginfo(f"Pixel Odometry Node initialized. Nominal FPS: {1.0/self.dt:.1f}Hz")

    def _initialize_kalman_filter(self):
        """Initializes the 6-state (Position + Velocity) Kalman Filter."""
        
        # State vector: [x, y, z, vx, vy, vz] (6x1)
        self.x = np.zeros((6, 1))

        # Transition matrix (F): Constant velocity model (6x6)
        self.F = np.eye(6)

        # Measurement function (H): We only observe position [x, y, z] (3x6)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # FIX 3: Base Process Noise Covariance (Q_base) calculation is now based on the SMALLER dt
        Q_std = 0.01  
        self.Q_base = np.eye(6) * Q_std**2 # We use this simple Q matrix
        
        # Measurement Noise Covariance (R): Near-perfect trust in the sensor (3x3)
        R_std = 0.01
        self.R = np.eye(3) * R_std**2

        # Initial State Covariance (P): High initial uncertainty
        self.P = np.eye(6) * 1.0 

        self.current_Q_scale = 1.0 
        self.kf_initialized = False

    def _predict_kalman_filter(self, dt):
        """KF Prediction step (based on constant velocity model)."""
        
        # Update F matrix with current dt
        # Note: dt is now small (0.033s), ensuring the prediction is only for a short time slice.
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        self.x = self.F @ self.x 

        Q_adaptive = self.Q_base * self.current_Q_scale 

        self.P = self.F @ self.P @ self.F.T + Q_adaptive 

    def _update_kalman_filter(self, z):
        """KF Update step (incorporate measurement z=[X, Y, Z]) and calculates adaptive scaling."""
        
        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        gamma = y.T @ np.linalg.inv(S) @ y 
        
        CHI2_THRESHOLD = 7.815 
        
        if gamma[0, 0] > CHI2_THRESHOLD:
            self.current_Q_scale = 10.0
        else:
            self.current_Q_scale = 1.0 
            
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def _predict_future_state(self, future_dt):
        """Predicts the state forward by a fixed factor for lag compensation."""
        F_future = np.eye(6)
        F_future[0, 3] = future_dt
        F_future[1, 4] = future_dt
        F_future[2, 5] = future_dt
        
        x_future = F_future @ self.x
        return x_future
        
    def camera_info_callback(self, camera_info_msg):
        """Processes the CameraInfo message to extract and store intrinsics."""
        if self.intrinsics_ready: return

        try:
            P = camera_info_msg.P
            self.fx = P[0]
            self.fy = P[5]
            self.cx = P[2]
            self.cy = P[6]
            self.CAMERA_FRAME_ID = camera_info_msg.header.frame_id
            
            if self.fx == 0 or self.fy == 0:
                 rospy.logerr("Camera intrinsics Fx/Fy are zero.")
                 return
            
            self.intrinsics_ready = True
            rospy.loginfo(f"Intrinsics loaded. Frame: {self.CAMERA_FRAME_ID}.")
            
        except Exception as e:
            rospy.logerr(f"Error processing CameraInfo: {e}")

    def pixel_location_callback(self, point_msg):
        """Stores the latest pixel location from the tracker."""
        self.current_pixel = (int(point_msg.x), int(point_msg.y))
        
    def convert_to_32fc1_meters(self, img_msg, encoding):
        """Converts an image message (16UC1 or 32FC1) to a 32FC1 NumPy array (depth in meters)."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, encoding)
            depth_array = np.array(cv_image, dtype=np.float32)

            if encoding == "16UC1":
                depth_array /= 1000.0
            
            depth_array[depth_array == 0] = np.nan
            return depth_array
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return None


    def _calculate_3d_position(self, center_x, center_y, depth_array):
        """Converts pixel coordinates and depth to 3D coordinates (X, Y, Z) in the camera frame."""
        
        if center_y < 0 or center_y >= depth_array.shape[0] or center_x < 0 or center_x >= depth_array.shape[1]:
            return None

        Z = depth_array[center_y, center_x]

        if Z == 0.0 or np.isnan(Z) or Z > self.MAX_DEPTH_VALUE:
            return None 

        # Pinhole model conversion (X right, Y down, Z forward)
        X = (center_x - self.cx) * Z / self.fx
        Y = (center_y - self.cy) * Z / self.fy
        
        return Point(x=X, y=Y, z=Z)

    def _get_quaternion_from_yaw(self, yaw):
        """Simple helper to create a Quaternion message from yaw (Z rotation)"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return Quaternion(x=0.0, y=0.0, z=sy, w=cy)


    def _update_odometry(self, current_measurement, current_time, header):
        """
        Calculates and publishes the Odometry message using the Kalman Filter.
        """
        
        # 0. Calculate Delta Time
        if self.last_time is not None:
            dt = (current_time - self.last_time).to_sec()
            
            # Clamp dt for sanity checks, but rely on actual dt for prediction
            if dt < 0 or dt > (self.dt * 5): 
                 dt = self.dt 
            self.dt = dt # The internal dt variable tracks the actual time difference
        else:
            self.dt = 1.0/30.0 # Use actual nominal dt for first prediction

        self._predict_kalman_filter(self.dt) # Predict to the current time step (t + dt)

        # 1. Initialization / Measurement Update
        if current_measurement is not None:
            z = np.array([[current_measurement.x], [current_measurement.y], [current_measurement.z]])
            
            if not self.kf_initialized:
                self.x[:3] = z
                self.kf_initialized = True
            else:
                self._update_kalman_filter(z)
        
        # 2. Publish Smoothed State (WITH LAG COMPENSATION)
        if self.kf_initialized:
            
            # Predict the state forward by the compensation factor to counteract latency.
            x_future = self._predict_future_state(self.PREDICTIVE_TIME_STEP)

            current_pose = Point(x=x_future[0, 0], y=x_future[1, 0], z=x_future[2, 0])
            
            vx, vy, vz = self.x[3, 0], self.x[4, 0], self.x[5, 0]

            # Calculate Orientation (Yaw) from estimated velocity
            current_yaw = math.atan2(vx, vy) 
            current_orientation = self._get_quaternion_from_yaw(current_yaw)

            # Build and publish the Odometry message
            odom_msg = Odometry()
            odom_msg.header = header
            odom_msg.header.frame_id = self.CAMERA_FRAME_ID
            odom_msg.child_frame_id = "tracked_object" 
            
            # Smoothed Pose (extrapolated)
            odom_msg.pose.pose.position = current_pose
            odom_msg.pose.pose.orientation = current_orientation
            
            # Smoothed Twist 
            odom_msg.twist.twist.linear = Vector3(x=vx, y=vy, z=vz)
            odom_msg.twist.twist.angular = Vector3(0, 0, 0)
            
            self.odom_pub.publish(odom_msg)

            # Update state for next cycle
            self.last_time = current_time
        
        elif current_measurement is not None:
            self.last_time = current_time


    def depth_callback(self, depth_msg):
        """
        Main callback: uses the latest pixel location and the current depth frame
        to calculate 3D pose (measurement) and trigger the Kalman Filter update.
        """
        if not self.intrinsics_ready:
            rospy.logwarn_throttle(5.0, "Waiting for camera intrinsics.")
            return

        if self.current_pixel is None:
            rospy.logwarn_throttle(5.0, "Waiting for initial pixel location from tracker.")
            return

        # Convert depth message to meters
        depth_array = self.convert_to_32fc1_meters(depth_msg, depth_msg.encoding)
        if depth_array is None:
             return

        center_x, center_y = self.current_pixel
        
        # 1. Calculate the current 3D position (The Measurement Z)
        current_pose_3d_measurement = self._calculate_3d_position(center_x, center_y, depth_array)
        current_time = depth_msg.header.stamp

        # 2. Trigger the Odometry update and filtering
        self._update_odometry(current_pose_3d_measurement, current_time, depth_msg.header)


if __name__ == '__main__':
    try:
        node = PixelOdometryNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass