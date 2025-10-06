#!/usr/bin/env python

# ROS 1 Python Node for Depth Image Segmentation

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class DepthSegmenter:
    """
    Subscribes to a raw depth image topic, segments a region of interest 
    based on pixel coordinates and depth thresholds, and publishes the 
    segmented depth image.
    """

    # --- Segmentation Parameters (Set your desired values here) ---
    # Pixel coordinates for the bounding box
    BOX_X_MIN = 225  # Column start
    BOX_Y_MIN = 200  # Row start
    BOX_WIDTH = 125  # Width in pixels
    BOX_HEIGHT = 125 # Height in pixels

    # Depth range in meters (m)
    MIN_DEPTH = 2.0  # Minimum accepted depth
    MAX_DEPTH = 2.5  # Maximum accepted depth
    # -----------------------------------------------------------

    def __init__(self):
        """Initializes the node, bridge, publisher, and subscriber."""
        rospy.init_node('depth_segmenter_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Publisher for the segmented depth image
        self.image_pub = rospy.Publisher("/segmented_depth_image", Image, queue_size=1)
        
        # Subscriber to the raw depth image topic
        # Assume the input topic is the raw depth image, e.g., from a Kinect or RealSense
        self.image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback, queue_size=1)
        
        rospy.loginfo("Depth Segmenter Node initialized.")
        rospy.loginfo(f"Segmenting region: X=[{self.BOX_X_MIN}, {self.BOX_X_MIN + self.BOX_WIDTH}], Y=[{self.BOX_Y_MIN}, {self.BOX_Y_MIN + self.BOX_HEIGHT}]")
        rospy.loginfo(f"Depth Threshold: [{self.MIN_DEPTH}m, {self.MAX_DEPTH}m]")

    def depth_callback(self, data):
        """
        Callback function executed upon receiving a new depth image message.
        This version automatically handles the conversion from millimeters (16UC1) to meters (32FC1).
        """
        try:
            # Convert ROS Image message to OpenCV image. 
            # We request the raw encoding of the data so we can handle 16UC1 or 32FC1 ourselves.
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Ensure the image is a float array (32FC1) for consistency in meter-based comparisons
        depth_array = np.array(cv_image, dtype=np.float32)

        # If the incoming encoding is 16UC1 (common for RealSense/Kinect), the values are in millimeters.
        if data.encoding == "16UC1":
            # Convert from millimeters to meters by dividing by 1000
            depth_array /= 1000.0
            rospy.logdebug("Converted 16UC1 (mm) depth image to 32FC1 (m).")
        # If it's already 32FC1, no division is needed as it's assumed to be in meters.

        # Get the dimensions of the image
        rows, cols = depth_array.shape
        
        # Calculate the actual bounding box boundaries, ensuring they don't exceed image bounds
        x_start = max(0, self.BOX_X_MIN)
        y_start = max(0, self.BOX_Y_MIN)
        x_end = min(cols, self.BOX_X_MIN + self.BOX_WIDTH)
        y_end = min(rows, self.BOX_Y_MIN + self.BOX_HEIGHT)

        # Create a segmented image initialized to zero (background)
        # Keep the output type as 32FC1 (meters)
        segmented_array = np.zeros_like(depth_array, dtype=np.float32)

        # 1. Extract the Region of Interest (ROI)
        roi = depth_array[y_start:y_end, x_start:x_end]

        # 2. Create a combined boolean mask for the segmentation criteria
        # All comparisons are now done in meters.
        valid_mask = (roi > 0.0) & (roi >= self.MIN_DEPTH) & (roi <= self.MAX_DEPTH)

        # 3. Apply the mask to the ROI: set non-masked values to zero
        segmented_roi = roi * valid_mask.astype(np.float32)

        # 4. Place the segmented ROI back into the full segmented image
        segmented_array[y_start:y_end, x_start:x_end] = segmented_roi

        # Convert the segmented OpenCV/NumPy image back to a ROS Image message
        try:
            # Publish as 32FC1 (float, meters)
            segmented_msg = self.bridge.cv2_to_imgmsg(segmented_array, "32FC1")
            segmented_msg.header = data.header 
            self.image_pub.publish(segmented_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error on publish: {e}")

if __name__ == '__main__':
    try:
        segmenter = DepthSegmenter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
