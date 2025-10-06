#!/usr/bin/env python3

import rospy
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class MidasDepthNode:
    def __init__(self):
        rospy.init_node('midas_depth_node')

        # Load MiDaS model
        self.model_type = "MiDaS_small"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)

        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform if "DPT" in self.model_type \
            else torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        self.bridge = CvBridge()

        self.frame_count = 0
        self.latest_midas_depth = None
        self.latest_sensor_depth = None

        # Subscribe to ROS image topic
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.depth_pub = rospy.Publisher("/camera/depth/MiDaS", Image, queue_size=1)
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.sensor_depth_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo("MiDaS Depth Node initialized and subscribed to /camera/image_raw")
        rospy.spin()

    def image_callback(self, msg):
        try:
            self.frame_count += 1
            if self.frame_count % 10 != 0:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

                # Apply MiDaS transform
                input_batch = self.transform(img_rgb).to(self.device)

                with torch.no_grad():
                    prediction = self.midas(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=cv_image.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                # Convert to NumPy
                depth_map = prediction.cpu().numpy()

                self.latest_midas_depth = depth_map

                # Normalize and colorize
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)

                # Publish the depth image
                try:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_normalized.astype(np.uint8), encoding="mono8")
                    depth_msg.header = msg.header  # Preserve timestamp/frame_id
                    self.depth_pub.publish(depth_msg)
                except Exception as e:
                    rospy.logerr(f"Error in image_callback: {e}")

                if self.latest_sensor_depth is not None:
                    self.compare_and_show(cv_image, depth_map, self.latest_sensor_depth)

                # Show image and depth
                #cv2.imshow("Original", cv_image)
                #cv2.imshow("MiDaS Depth Map", depth_colormap)
                #cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def sensor_depth_callback(self, msg):
        try:
            # Convert sensor depth image (often 32FC1 or 16UC1)
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_sensor_depth = depth_img
        except Exception as e:
            rospy.logerr(f"Error in sensor_depth_callback: {e}")

    def compare_and_show(self, color_img, midas_depth, sensor_depth):
        # Normalize midas and sensor depth for visualization
        midas_norm = cv2.normalize(midas_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Handle sensor depth NaNs/infs & clip to 5m for visualization
        sensor_depth_clean = np.nan_to_num(sensor_depth, nan=0.0, posinf=0.0, neginf=0.0)
        sensor_depth_clip = np.clip(sensor_depth_clean, 0, 5000)
        sensor_norm = cv2.normalize(sensor_depth_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize sensor depth to midas size if needed
        if sensor_norm.shape != midas_norm.shape:
            sensor_norm = cv2.resize(sensor_norm, (midas_norm.shape[1], midas_norm.shape[0]))

        sensor_norm = 255 - sensor_norm

        # Compute global min and max over both depth maps combined
        #min_depth = min(np.min(midas_depth), np.min(sensor_depth_clip))
        #max_depth = max(np.max(midas_depth), np.max(sensor_depth_clip))

        # Normalize both using this common scale
        #midas_norm = cv2.normalize(midas_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, alpha=0, beta=255,
        #                          normType=cv2.NORM_MINMAX)
        #sensor_norm = cv2.normalize(sensor_depth_clip, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, alpha=0, beta=255,
        #                           normType=cv2.NORM_MINMAX)

        # Instead of above, do this manually:

        #midas_norm = ((midas_depth - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        #sensor_norm = ((sensor_depth_clip - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

        # Colorize
        midas_colormap = cv2.applyColorMap(midas_norm, cv2.COLORMAP_MAGMA)
        sensor_colormap = cv2.applyColorMap(sensor_norm, cv2.COLORMAP_MAGMA)

        # Compute absolute difference
        diff = cv2.absdiff(midas_norm, sensor_norm)
        diff_colormap = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
        #overlay = cv2.addWeighted(color_img, 0.6, diff_colormap, 0.4, 0)

        # Compute absolute difference
        #diff = np.abs(midas_norm - sensor_norm)

        # Optional: mask out tiny differences (e.g., noise)
        #threshold = 0.1  # meters or your unit
        #mask = diff > threshold

        # Normalize difference image for visualization
        #diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        #diff_colormap = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)

        # Make a 3-channel mask to apply only where there's difference
        #mask_3c = np.stack([mask]*3, axis=-1)  # shape: H x W x 3

        # Resize diff_colormap and color_img if needed to match
        #diff_colormap = cv2.resize(diff_colormap, (color_img.shape[1], color_img.shape[0]))

        # Fuse: keep original color where no diff, overlay where there's difference
        #fused = np.where(mask_3c, diff_colormap, color_img)

        # Stack images side by side
        combined_1 = np.hstack((color_img, midas_colormap))
        combined_2 = np.hstack((sensor_colormap, diff_colormap))
        combined = np.vstack((combined_1, combined_2))
        scale = 0.5  # or 0.3, 0.75, etc.
        h, w = combined.shape[:2]
        combined_resized = cv2.resize(combined, (int(w * scale), int(h * scale)))
        cv2.imshow("Color | MiDaS Depth | Sensor Depth | Difference", combined_resized)
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        MidasDepthNode()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
