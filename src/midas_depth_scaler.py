import rospy
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError # Added CvBridgeError

# PyTorch and MiDaS imports
try:
    import torch
    import torch.hub
    
    # Load the smallest MiDaS model via torch.hub
    MiDaS_MODEL_TYPE = "MiDaS_small"
    MiDaS_MODEL = torch.hub.load("intel-isl/MiDaS", MiDaS_MODEL_TYPE)
    
    # Use GPU if available, otherwise CPU
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MiDaS_MODEL.to(DEVICE)
    MiDaS_MODEL.eval()
    
    # Load MiDaS transformation function
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if MiDaS_MODEL_TYPE == "DPT_Large" or MiDaS_MODEL_TYPE == "DPT_Hybrid":
        TRANSFORM = midas_transforms.dpt_transform
    else: # Default for small models
        TRANSFORM = midas_transforms.small_transform
    
    print(f"MiDaS model ({MiDaS_MODEL_TYPE}) loaded successfully on {DEVICE}")

except ImportError as e:
    print(f"ERROR: PyTorch or MiDaS dependencies not found: {e}")
    print("Please install torch, torchvision, and ensure MiDaS dependencies are available.")
    # Create mock placeholders to allow ROS node initialization
    MiDaS_MODEL = None
    TRANSFORM = None
    DEVICE = None

class MiDaSDepthEstimator:
    """
    ROS node to perform MiDaS monocular depth estimation, scale it using a true 
    depth sensor, and compare the results.
    """
    def __init__(self):
        rospy.init_node('midas_depth_estimator_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Stores the latest raw MiDaS output (float32, unnormalized) for the sync_callback
        self.last_midas_raw_numpy = None 
        # Stores the last frame's smoothed metric depth map for temporal consistency
        self.last_depth_scaled_smoothed = None
        # Time (seconds) when the U-TURN message is allowed to expire and the arrow can reappear.
        self.uturn_expire_time = 0.0 
        # Exponential Moving Average (EMA) factor for temporal smoothing (e.g., 0.8 = 80% new frame, 20% old frame)
        self.SMOOTHING_ALPHA = 0.8 

        # Publishers
        # This topic now publishes the normalized, viewable 8-bit image
        self.pub_mono_raw = rospy.Publisher('/mono/depth/image_raw', Image, queue_size=1) 
        self.pub_mono_scaled = rospy.Publisher('/mono/depth/image_scaled', Image, queue_size=1)
        self.pub_comparison = rospy.Publisher('/comparison_image', Image, queue_size=1)
        # New publisher for the visualized metric depth map
        self.pub_mono_draw = rospy.Publisher('/mono/depth/image_draw', Image, queue_size=1)

        # ----------------------------------------------------------------------
        # 1. RGB Subscriber for continuous MiDaS Inference (regardless of sync)
        # ----------------------------------------------------------------------
        # This callback performs inference and publishes the raw MiDaS output.
        self.sub_rgb_inference = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_inference_callback, queue_size=1)

        # ----------------------------------------------------------------------
        # 2. Synchronizer for Scaling and Comparison
        # ----------------------------------------------------------------------
        # We synchronize the *output* of the MiDaS pipeline with the true depth sensor data.
        # Note: sub_mono_raw acts as a time trigger, but the data is pulled from self.last_midas_raw_numpy
        sub_mono_raw = message_filters.Subscriber('/mono/depth/image_raw', Image)
        sub_depth_true = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)

        # Synchronize messages. Increased queue_size (10) and slop (0.5s) for better robustness 
        # against timing variations in sensor data.
        ts = message_filters.ApproximateTimeSynchronizer([sub_mono_raw, sub_depth_true], 
                                                         queue_size=10, slop=0.5) # Increased queue and slop
        ts.registerCallback(self.sync_callback)

        rospy.loginfo("MiDaSDepthEstimator node initialized and synchronizing topics...")

    def run_midas_inference(self, cv_image):
        """
        Performs MiDaS inference on the input image.
        Returns the raw MiDaS output (disparity/inverse depth map) as a NumPy array.
        """
        if MiDaS_MODEL is None:
            # Fallback for missing dependency - returns a mock map
            rospy.logwarn_throttle(5, "MiDaS model not loaded. Returning mock depth map.")
            return np.ones(cv_image.shape[:2], dtype=np.float32) * 0.5
        
        # 1. Transform the image
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        input_batch = TRANSFORM(img).to(DEVICE)

        with torch.no_grad():
            # 2. Run inference
            prediction = MiDaS_MODEL(input_batch)

            # 3. Resize and convert to NumPy
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def process_true_depth(self, depth_msg):
        """
        Converts ROS depth image (e.g., 16UC1 or 32FC1) to a float32 depth map (in meters).
        """
        try:
            # Try 32-bit float first (standard for ROS depth in meters)
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if cv_depth.dtype == np.uint16:
                # Convert 16-bit millimeter depth to 32-bit meter depth
                # Assume 0 is invalid/far away for visualization
                cv_depth = cv_depth.astype(np.float32) / 1000.0
                cv_depth[cv_depth == 0] = np.nan  # Mark 0 depth as NaN/invalid
            
            # Ensure it's float32 for calculations
            return cv_depth.astype(np.float32)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return None

    def rgb_inference_callback(self, rgb_msg):
        """
        Callback for /camera/image_raw. Performs MiDaS inference and publishes 
        the normalized disparity map to /mono/depth/image_raw for visualization.
        """
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting RGB image for inference: {e}")
            return

        # 1. MiDaS Inference: Get raw float inverse depth data
        depth_midas_raw_float = self.run_midas_inference(cv_rgb)
        
        # Store the raw float data for the scaling callback (sync_callback)
        self.last_midas_raw_numpy = depth_midas_raw_float

        # 2. Normalize and convert to 8-bit for visualization
        # Find min/max values in the map for normalization
        min_val = depth_midas_raw_float.min()
        max_val = depth_midas_raw_float.max()
        
        if max_val > min_val:
            # Normalize to 0-1, then scale to 0-255 (8-bit grayscale)
            # Farther objects (smaller disparity values) appear darker (closer to 0)
            normalized_map = (depth_midas_raw_float - min_val) / (max_val - min_val)
            depth_midas_vis = (normalized_map * 255).astype(np.uint8)
        else:
            # Handle flat image case
            depth_midas_vis = np.zeros_like(depth_midas_raw_float, dtype=np.uint8)

        # 3. Publish Normalized 8-bit Image for Visualization
        # The topic /mono/depth/image_raw now shows the gradient properly
        self.pub_mono_raw.publish(self.bridge.cv2_to_imgmsg(depth_midas_vis, "mono8", rgb_msg.header))
        rospy.loginfo_once("Started publishing normalized MiDaS depth to /mono/depth/image_raw (8-bit for viewing).")


    def sync_callback(self, mono_raw_msg, depth_true_msg):
        """
        Synchronized callback for incoming MiDaS Raw Depth and True Depth messages.
        It uses the raw float MiDaS data stored in the class member.
        1. Read MiDaS Raw Depth (from class member).
        2. Scale Estimation & Scaling.
        3. Comparison Map Generation.
        4. Publishing results.
        """
        rospy.loginfo("Received synchronized depth messages. Processing scaling and comparison...")

        # 1. Read raw MiDaS data from class member
        depth_midas_raw = self.last_midas_raw_numpy
        if depth_midas_raw is None:
            rospy.logwarn("Raw MiDaS data not available. Skipping sync callback.")
            return

        try:
            # Convert ROS Images to OpenCV format
            depth_true = self.process_true_depth(depth_true_msg)
            
            if depth_true is None:
                return # Skip if depth conversion failed

        except Exception as e:
            rospy.logerr(f"Error converting synchronized images: {e}")
            return

        # ----------------------------------------------------------------------
        # 2. Scale Estimation and Metric Scaling
        # ----------------------------------------------------------------------
        H, W = depth_true.shape
        # Center coordinates
        center_y, center_x = H // 2, W // 2
        
        # Resize MiDaS output to match True Depth resolution (crucial for alignment)
        depth_midas_aligned = cv2.resize(depth_midas_raw, (W, H), 
                                         interpolation=cv2.INTER_LINEAR)

        # Define a proxy for "really far away" in MiDaS disparity space (a very small, non-zero value)
        # This will result in a very large metric depth estimate (1/D)
        FAR_AWAY_PROXY_DISPARITY = 1e-4 

        # Extract center pixels
        midas_center_val = depth_midas_aligned[center_y, center_x]
        true_center_val = depth_true[center_y, center_x]
        
        # Check for invalid TRUE depth values (cannot scale without true reference)
        if not np.isfinite(true_center_val) or true_center_val <= 0:
            rospy.logwarn("Invalid or zero TRUE depth at center pixel. Skipping scaling.")
            return

        # Check for invalid ESTIMATED MiDaS disparity value at the center
        if not np.isfinite(midas_center_val) or midas_center_val <= 0:
            # If the MiDaS estimate is invalid/zero, substitute a very small disparity 
            # (proxy for far away metric distance) for scaling.
            rospy.logwarn(f"Invalid or zero MiDaS output at center pixel ({midas_center_val:.4f}). Using 'far away' proxy ({FAR_AWAY_PROXY_DISPARITY}) for scaling.")
            midas_center_val = FAR_AWAY_PROXY_DISPARITY
        
        # MiDaS output is proportional to inverse depth (disparity).
        # True Depth (Z) is metric depth. 
        # We need Z_scaled = S' / D_midas
        # At the center: Z_true = S' / D_midas
        # Scale Factor S' = Z_true_center * D_midas_center
        scale_factor_s_prime = true_center_val * midas_center_val

        # Apply the scale factor to get metric depth in meters
        # Avoid division by zero by setting near-zero MiDaS values to a small constant
        epsilon = 1e-6
        depth_midas_scaled = scale_factor_s_prime / (depth_midas_aligned + epsilon)
        
        # --- Temporal Smoothing (EMA) ---
        depth_midas_smoothed = depth_midas_scaled.copy()
        
        if self.last_depth_scaled_smoothed is not None:
            # Resize the last smoothed map to match the current frame size (H, W) if necessary
            # This is critical if the input image size can change, though usually fixed in ROS.
            last_smoothed_aligned = cv2.resize(self.last_depth_scaled_smoothed, (W, H), 
                                                interpolation=cv2.INTER_LINEAR)
            
            # Apply EMA: Smoothed = alpha * Current + (1 - alpha) * Previous
            alpha = self.SMOOTHING_ALPHA
            
            # Use a mask to only smooth where both current and previous estimates are valid/finite
            valid_mask = np.isfinite(depth_midas_scaled) & (depth_midas_scaled > 0)
            valid_mask_prev = np.isfinite(last_smoothed_aligned) & (last_smoothed_aligned > 0)
            
            common_valid_mask = valid_mask & valid_mask_prev
            
            # Perform the smoothing calculation only on common valid pixels
            depth_midas_smoothed[common_valid_mask] = (alpha * depth_midas_scaled[common_valid_mask] + 
                                                      (1 - alpha) * last_smoothed_aligned[common_valid_mask])

        # Store the current smoothed map for the next iteration
        self.last_depth_scaled_smoothed = depth_midas_smoothed.copy()
        
        # --- NOTE: All subsequent code now uses depth_midas_smoothed ---
        
        # Publish Scaled Metric Depth (float32, meters)
        # Use the True Depth message header for the output
        self.pub_mono_scaled.publish(self.bridge.cv2_to_imgmsg(depth_midas_smoothed, "32FC1", depth_true_msg.header))
        rospy.loginfo(f"Scale Factor (S'): {scale_factor_s_prime:.4f}. Smoothed scaled depth published.")

        # ----------------------------------------------------------------------
        # 3. Visualization and Drawing (/mono/depth/image_draw)
        # ----------------------------------------------------------------------
        
        # --- Prepare the image for drawing (Normalization + BGR Conversion) ---
        
        # 1. Normalize the smoothed metric depth (in meters) for visualization
        # Use a sensible clipping range (e.g., 0.1m to 10m) for depth visualization
        VIS_MIN_DEPTH = 0.1
        VIS_MAX_DEPTH = 10.0
        
        visual_map = np.clip(depth_midas_smoothed, VIS_MIN_DEPTH, VIS_MAX_DEPTH)
        
        # Convert depth (meters) to 8-bit grayscale for display
        # We invert the colors here so closer objects are brighter (standard depth visualization)
        normalized_vis_map = 1.0 - ((visual_map - VIS_MIN_DEPTH) / (VIS_MAX_DEPTH - VIS_MIN_DEPTH))
        depth_draw_8bit = (normalized_vis_map * 255).astype(np.uint8)
        
        # Convert to BGR so we can draw colored shapes/text
        depth_draw_bgr = cv2.cvtColor(depth_draw_8bit, cv2.COLOR_GRAY2BGR)

        # --- Feature 1 & 3: Find Closest Depth in Radius and Draw Circle ---
        
        RADIUS = 40
        CENTER = (center_x, center_y)

        # Define the region of interest (ROI) for the center area
        y_start = max(0, center_y - RADIUS)
        y_end = min(H, center_y + RADIUS)
        x_start = max(0, center_x - RADIUS)
        x_end = min(W, center_x + RADIUS)

        # Extract ROI from the estimated metric depth map (smoothed)
        roi_depth_estimated = depth_midas_smoothed[y_start:y_end, x_start:x_end]
        closest_depth_m_estimated = np.nanmin(roi_depth_estimated[roi_depth_estimated > 0]) # Find min, ignore NaNs/zeros
        
        # Extract ROI from the true depth map
        roi_depth_true = depth_true[y_start:y_end, x_start:x_end]
        closest_depth_m_true = np.nanmin(roi_depth_true[roi_depth_true > 0]) # Find min, ignore NaNs/zeros


        # Draw blue circle around the radius
        BLUE = (255, 0, 0) # BGR
        cv2.circle(depth_draw_bgr, CENTER, RADIUS, BLUE, 2)

        # --- Feature 2: Display Closest Estimated Depth Text (RED) ---
        RED = (0, 0, 255) # BGR
        estimated_text = f"Est: {closest_depth_m_estimated:.2f} m"
        
        # Position text slightly above the circle center
        text_pos_est = (center_x - 50, center_y - RADIUS - 10) 
        cv2.putText(depth_draw_bgr, estimated_text, text_pos_est, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        
        # --- Feature 5: Display Closest True Depth Text (GREEN) ---
        GREEN = (0, 255, 0) # BGR
        true_text = f"True: {closest_depth_m_true:.2f} m"
        
        # Position text below the estimated depth text (offset by ~25 pixels)
        text_pos_true = (center_x - 50, center_y - RADIUS + 15) 
        cv2.putText(depth_draw_bgr, true_text, text_pos_true, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)

        # --- Feature 4: Arrow/U-TURN Logic ---
        
        # 1. Calculate the average estimated depth in the 40-pixel radius
        valid_roi_depths = roi_depth_estimated[roi_depth_estimated > 0]
        if valid_roi_depths.size > 0:
            avg_depth_in_radius = np.mean(valid_roi_depths)
        else:
            # If no valid depth in radius, fallback to a small threshold
            avg_depth_in_radius = 0.01 
        
        # 2. Find the pixel location (y, x) with the maximum depth that is GREATER than the local average
        
        # Use np.nan_to_num to handle NaNs, setting them to -1.0 so they fail the check
        depths_for_max = np.nan_to_num(depth_midas_smoothed, nan=-1.0)
        
        # Mask the depths: only consider points significantly farther than the local average
        # We use a 5% buffer above average (0.01m minimum)
        min_depth_threshold = max(avg_depth_in_radius * 1.05, 0.01)
        far_depth_mask = depths_for_max > min_depth_threshold
        
        # Apply the mask. Set masked-out values to -1.0 to ensure argmax picks the farthest valid point
        masked_depths = np.where(far_depth_mask, depths_for_max, -1.0)

        max_idx = np.argmax(masked_depths)
        
        # Convert flat index to 2D coordinates (y, x)
        max_y, max_x = np.unravel_index(max_idx, masked_depths.shape)
        
        # Check if the chosen max depth is actually a valid depth (i.e., not -1.0)
        max_depth_val = masked_depths[max_y, max_x]
        
        current_time = rospy.Time.now().to_sec()

        # U-TURN conditions:
        # U-TURN only when the closest estimated depth within the 40-pixel radius is less than or equal to 0.3 meter.
        UTURN_DISTANCE_THRESHOLD = 0.3
        is_uturn_trigger = closest_depth_m_estimated <= UTURN_DISTANCE_THRESHOLD
        
        if is_uturn_trigger:
            # If a U-TURN trigger occurs, ensure the message is displayed for 3 seconds minimum.
            if current_time >= self.uturn_expire_time:
                # New trigger or expired timer: start new 3 second countdown
                self.uturn_expire_time = current_time + 3.0
                rospy.loginfo_throttle(1.0, f"U-TURN triggered: Closest object within {UTURN_DISTANCE_THRESHOLD}m radius. Displaying 'U-TURN'.")
            
            # Draw U-TURN text (Display if triggered OR if the timer is still active)
            U_TURN_TEXT = "U-TURN"
            # Center text placement
            (text_w, text_h), baseline = cv2.getTextSize(U_TURN_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
            text_x = int((W - text_w) / 2)
            text_y = int((H + text_h) / 2)
            cv2.putText(depth_draw_bgr, U_TURN_TEXT, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 3, cv2.LINE_AA)
            
        elif current_time < self.uturn_expire_time:
            # U-TURN is NOT triggered this cycle, but the timer is still running (force display for 3s minimum).
            
            # Draw U-TURN text
            U_TURN_TEXT = "U-TURN"
            (text_w, text_h), baseline = cv2.getTextSize(U_TURN_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
            text_x = int((W - text_w) / 2)
            text_y = int((H + text_h) / 2)
            cv2.putText(depth_draw_bgr, U_TURN_TEXT, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 3, cv2.LINE_AA)

        else:
            # U-TURN is not triggered and timer has expired: Draw the normal arrow.
            
            if max_depth_val > 0:
                # Point to the farthest depth found
                ARROW_TIP = (max_x, max_y)
            else:
                # Fallback direction if no clear path is found.
                ARROW_TIP = (center_x, center_y - 50) 
                rospy.logwarn_throttle(2.0, "No clear far point found. Arrow points up (default).")

            # Draw red arrow from center to max depth point
            cv2.arrowedLine(depth_draw_bgr, CENTER, ARROW_TIP, RED, 3, tipLength=0.1)

        # Publish the drawn image
        self.pub_mono_draw.publish(self.bridge.cv2_to_imgmsg(depth_draw_bgr, "bgr8", depth_true_msg.header))
        rospy.loginfo("Visualized metric depth map published to /mono/depth/image_draw.")

        # ----------------------------------------------------------------------
        # 4. Comparison Image Generation and Publication
        # ----------------------------------------------------------------------
        
        # Use only valid depth regions for comparison
        valid_mask_true = np.isfinite(depth_true) & (depth_true > 0)
        valid_mask_mono = np.isfinite(depth_midas_smoothed) & (depth_midas_smoothed > 0)
        valid_mask = valid_mask_true & valid_mask_mono

        # Calculate absolute difference (using smoothed map)
        diff_map = np.abs(depth_midas_smoothed - depth_true)
        
        # Max error (in meters) to consider for visualization normalization (e.g., 1.0 meter)
        MAX_ERROR_M = 1.0
        
        # Normalize the difference map: 
        # Difference_Norm = min(|Error| / MAX_ERROR, 1.0)
        diff_norm = np.clip(diff_map / MAX_ERROR_M, 0.0, 1.0)

        # Comparison Pixel Value: 
        # White (255) = 0 difference (Error_Norm = 0)
        # Black (0) = Max difference (Error_Norm = 1.0)
        comparison_image_gray = (255 - (diff_norm * 255)).astype(np.uint8)
        
        # Apply the valid mask: pixels where either depth is invalid should be gray (e.g., 127)
        comparison_image_gray[~valid_mask] = 127
        
        # Convert grayscale to BGR for publishing (optional, but often preferred for Image topics)
        comparison_image_bgr = cv2.cvtColor(comparison_image_gray, cv2.COLOR_GRAY2BGR)

        # Publish the Comparison Image
        self.pub_comparison.publish(self.bridge.cv2_to_imgmsg(comparison_image_bgr, "bgr8", depth_true_msg.header))
        rospy.loginfo("Comparison image published.")

if __name__ == '__main__':
    try:
        # Check if MiDaS model loaded correctly before starting the node
        if MiDaS_MODEL is not None:
            node = MiDaSDepthEstimator()
            rospy.spin()
        else:
            rospy.logerr("MiDaS model failed to load. Exiting ROS node.")
    except rospy.ROSInterruptException:
        pass
