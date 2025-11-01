import rospy
import cv2
import numpy as np
import message_filters
import random
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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
    depth sensor (simulated ToF), and compare the results.
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
        # Exponential Moving Average (EMA) factor for temporal smoothing 
        self.SMOOTHING_ALPHA = 0.8 

        # --- Arrow Smoothing Control ---
        self.ARROW_SMOOTHING_BETA = 0.1 
        self.last_arrow_tip_x = -1 
        self.last_arrow_tip_y = -1 
        # -------------------------------
        
        # --- Dynamic Cone Simulation Parameters (Based on Multi-Ranger Deck 27 deg half-angle) ---
        self.CONE_SLOPE_PIXELS_PER_METER = 20.0 
        self.MIN_CONE_RADIUS_PIXELS = 5     
        self.SENSOR_MAX_RANGE = 3.0         # Max range of simulated single-point sensor (meters)
        self.MIN_ACQUISITION_M = 0.03       # Minimum range the sensor can reliably measure (3 cm)
        
        # --- Fixed Max Cone Size ---
        # Calculation for the fixed boundary circle (at 3.0m)
        self.MAX_CONE_RADIUS_PIXELS = int(self.SENSOR_MAX_RANGE * self.CONE_SLOPE_PIXELS_PER_METER)
        # ---------------------------

        # --- Distance-Dependent ToF Noise Parameters & Dropouts ---
        self.TOF_NOISE_BASE_M = 0.01        
        self.TOF_NOISE_FACTOR_M = 0.01      
        self.PROBABILITY_DROP_AT_MAX = 0.15 
        # ----------------------------------------------------------------------

        # Publishers
        self.pub_mono_raw = rospy.Publisher('/mono/depth/image_raw', Image, queue_size=1) 
        self.pub_mono_scaled = rospy.Publisher('/mono/depth/image_scaled', Image, queue_size=1)
        self.pub_comparison = rospy.Publisher('/comparison_image', Image, queue_size=1)
        self.pub_mono_draw = rospy.Publisher('/mono/depth/image_draw', Image, queue_size=1)

        # ----------------------------------------------------------------------
        # 1. RGB Subscriber for continuous MiDaS Inference (regardless of sync)
        # ----------------------------------------------------------------------
        self.sub_rgb_inference = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_inference_callback, queue_size=1)

        # ----------------------------------------------------------------------
        # 2. Synchronizer for Scaling and Comparison
        # ----------------------------------------------------------------------
        sub_mono_raw = message_filters.Subscriber('/camera/depth/image_rect_raw', Image) 
        sub_depth_true = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)

        ts = message_filters.ApproximateTimeSynchronizer([sub_mono_raw, sub_depth_true], 
                                                         queue_size=10, slop=0.5) 
        ts.registerCallback(self.sync_callback)

        rospy.loginfo("MiDaSDepthEstimator node initialized and synchronizing topics...")
        
    def add_simulated_tof_noise(self, depth_value):
        """
        Adds Gaussian noise and models probabilistic dropouts (returning 0.0) 
        as depth approaches max range.
        """
        if not np.isfinite(depth_value) or depth_value <= 0:
            return depth_value
        
        # 1. Probabilistic Dropout (Failure to Acquire)
        dropout_scale = depth_value / self.SENSOR_MAX_RANGE
        dropout_prob = min(self.PROBABILITY_DROP_AT_MAX, dropout_scale * self.PROBABILITY_DROP_AT_MAX)
        
        if random.random() < dropout_prob:
            return 0.0  # Sensor failure (will be filtered as invalid later)

        # 2. Distance-Dependent Gaussian Noise
        std_dev = self.TOF_NOISE_BASE_M + (self.TOF_NOISE_FACTOR_M * depth_value)
        
        noise = random.gauss(0, std_dev)
        noisy_depth = depth_value + noise
        
        return max(0.001, noisy_depth)


    def run_midas_inference(self, cv_image):
        """
        Performs MiDaS inference on the input image.
        Returns the raw MiDaS output (disparity/inverse depth map) as a NumPy array.
        """
        if MiDaS_MODEL is None:
            rospy.logwarn_throttle(5, "MiDaS model not loaded. Returning mock depth map.")
            return np.ones(cv_image.shape[:2], dtype=np.float32) * 0.5
        
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        input_batch = TRANSFORM(img).to(DEVICE)

        with torch.no_grad():
            prediction = MiDaS_MODEL(input_batch)

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
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if cv_depth.dtype == np.uint16:
                cv_depth = cv_depth.astype(np.float32) / 1000.0
                cv_depth[cv_depth == 0] = np.nan
            
            return cv_depth.astype(np.float32)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return None

    def rgb_inference_callback(self, rgb_msg):
        """
        Performs MiDaS inference and publishes the normalized disparity map for visualization.
        """
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting RGB image for inference: {e}")
            return

        depth_midas_raw_float = self.run_midas_inference(cv_rgb)
        self.last_midas_raw_numpy = depth_midas_raw_float

        min_val = depth_midas_raw_float.min()
        max_val = depth_midas_raw_float.max()
        
        if max_val > min_val:
            normalized_map = (depth_midas_raw_float - min_val) / (max_val - min_val)
            depth_midas_vis = (normalized_map * 255).astype(np.uint8)
        else:
            depth_midas_vis = np.zeros_like(depth_midas_raw_float, dtype=np.uint8)

        self.pub_mono_raw.publish(self.bridge.cv2_to_imgmsg(depth_midas_vis, "mono8", rgb_msg.header))
        rospy.loginfo_once("Started publishing normalized MiDaS depth to /mono/depth/image_raw (8-bit for viewing).")


    def sync_callback(self, mono_raw_msg, depth_true_msg):
        """
        Synchronized callback for scaling, smoothing, and visualization.
        """
        rospy.loginfo("Received synchronized depth messages. Processing scaling and comparison...")

        depth_midas_raw = self.last_midas_raw_numpy
        if depth_midas_raw is None:
            rospy.logwarn("Raw MiDaS data not available. Skipping sync callback.")
            return

        try:
            depth_true = self.process_true_depth(depth_true_msg)
            if depth_true is None:
                return 

        except Exception as e:
            rospy.logerr(f"Error converting synchronized images: {e}")
            return

        # ----------------------------------------------------------------------
        # 2. Dynamic Cone Range Check and Metric Scaling
        # ----------------------------------------------------------------------
        H, W = depth_true.shape
        center_y, center_x = H // 2, W // 2
        
        depth_midas_aligned = cv2.resize(depth_midas_raw, (W, H), 
                                         interpolation=cv2.INTER_LINEAR)

        FAR_AWAY_PROXY_DISPARITY = 1e-4 
        MIN_ACQ = self.MIN_ACQUISITION_M
        
        # --- A. Fixed Max Cone Search ROI (Simulates Sensor FoV) ---
        # The sensor can physically see everything within the 27-degree cone at 3.0m.
        # This defines the physical search boundary.
        MAX_RADIUS = self.MAX_CONE_RADIUS_PIXELS 
        y_max_start = max(0, center_y - MAX_RADIUS)
        y_max_end = min(H, center_y + MAX_RADIUS)
        x_max_start = max(0, center_x - MAX_RADIUS)
        x_max_end = min(W, center_x + MAX_RADIUS)

        roi_true_max = depth_true[y_max_start:y_max_end, x_max_start:x_max_end]
        
        # Filter true depth for valid readings within the physical cone area
        valid_true_max = roi_true_max[np.isfinite(roi_true_max) & 
                                      (roi_true_max > MIN_ACQ) &  
                                      (roi_true_max > 0)]

        # Find the absolute closest object the sensor *could* see
        if valid_true_max.size == 0:
            true_min_for_radius_base = self.SENSOR_MAX_RANGE + 1.0 # Assume far away
        else:
            true_min_for_radius_base = np.nanmin(valid_true_max)
        
        # --- B. Add Simulated ToF Noise and Dropout ---
        true_min_val = self.add_simulated_tof_noise(true_min_for_radius_base)

        # 1. Calculate Dynamic Radius (R_cone = MIN_R + Z * Slope)
        CONE_RADIUS_PIXELS = max(self.MIN_CONE_RADIUS_PIXELS, 
                                 int(true_min_val * self.CONE_SLOPE_PIXELS_PER_METER))
        
        MAX_R = min(H, W) // 2 - 1 
        CONE_RADIUS_PIXELS = min(CONE_RADIUS_PIXELS, MAX_R)

        # --- C. Range Check ---
        sensor_is_active = true_min_val <= self.SENSOR_MAX_RANGE and true_min_val > MIN_ACQ

        # --- D. Metric Scaling (Only if sensor is successfully in range) ---
        if sensor_is_active:
            # Re-Extract MDE ROI using the dynamic cone size for the median anchor
            y_start = max(0, center_y - CONE_RADIUS_PIXELS)
            y_end = min(H, center_y + CONE_RADIUS_PIXELS)
            x_start = max(0, center_x - CONE_RADIUS_PIXELS)
            x_end = min(W, center_x + CONE_RADIUS_PIXELS)
            roi_midas = depth_midas_aligned[y_start:y_end, x_start:x_end]
            valid_midas = roi_midas[np.isfinite(roi_midas) & (roi_midas > 0)]
            midas_median_val = np.median(valid_midas) if valid_midas.size > 0 else FAR_AWAY_PROXY_DISPARITY

            # Scale Factor S' = Z_true_min * D_midas_median
            scale_factor_s_prime = true_min_val * midas_median_val

            # Apply the scale factor to get metric depth in meters
            epsilon = 1e-6
            depth_midas_scaled = scale_factor_s_prime / (depth_midas_aligned + epsilon)
            
            # --- Temporal Smoothing (EMA) ---
            depth_midas_smoothed = depth_midas_scaled.copy()
            
            if self.last_depth_scaled_smoothed is not None:
                last_smoothed_aligned = cv2.resize(self.last_depth_scaled_smoothed, (W, H), 
                                                    interpolation=cv2.INTER_LINEAR)
                
                alpha = self.SMOOTHING_ALPHA
                valid_mask = np.isfinite(depth_midas_scaled) & (depth_midas_scaled > 0)
                valid_mask_prev = np.isfinite(last_smoothed_aligned) & (last_smoothed_aligned > 0)
                common_valid_mask = valid_mask & valid_mask_prev
                
                depth_midas_smoothed[common_valid_mask] = (alpha * depth_midas_scaled[common_valid_mask] + 
                                                          (1 - alpha) * last_smoothed_aligned[common_valid_mask])

            self.last_depth_scaled_smoothed = depth_midas_smoothed.copy()
            
            self.pub_mono_scaled.publish(self.bridge.cv2_to_imgmsg(depth_midas_smoothed, "32FC1", depth_true_msg.header))
            rospy.loginfo(f"Scale Factor (S'): {scale_factor_s_prime:.4f}. Smoothed scaled depth published.")
        
        # ----------------------------------------------------------------------
        # 3. Visualization and Drawing (/mono/depth/image_draw)
        # ----------------------------------------------------------------------
        
        smooth_map_for_vis = self.last_depth_scaled_smoothed
        if smooth_map_for_vis is None:
            rospy.logwarn("Skipping visualization: Smoothed depth map not initialized.")
            return

        VIS_MIN_DEPTH = 0.1
        VIS_MAX_DEPTH = 10.0
        
        visual_map = np.clip(smooth_map_for_vis, VIS_MIN_DEPTH, VIS_MAX_DEPTH)
        
        normalized_vis_map = ((visual_map - VIS_MIN_DEPTH) / (VIS_MAX_DEPTH - VIS_MIN_DEPTH)) * 255.0
        depth_draw_8bit = (normalized_vis_map).astype(np.uint8)
        
        depth_draw_bgr = cv2.applyColorMap(depth_draw_8bit, cv2.COLORMAP_VIRIDIS)
        
        CENTER = (center_x, center_y)
        RED = (0, 0, 255) 
        GREEN = (0, 255, 0) 
        WHITE = (255, 255, 255)
        BLUE = (255, 0, 0)
        
        # --- 1. Draw Fixed Max Cone Boundary ---
        cv2.circle(depth_draw_bgr, CENTER, self.MAX_CONE_RADIUS_PIXELS, WHITE, 1)

        if sensor_is_active:
            # 2. Draw Dynamic Cone and calculate estimated depth
            RADIUS = CONE_RADIUS_PIXELS 
            cv2.circle(depth_draw_bgr, CENTER, RADIUS, BLUE, 2) # Blue circle
            
            # Find min estimated depth within the cone ROI
            # Note: y_start/x_start were calculated for this radius in step E.
            roi_depth_estimated = smooth_map_for_vis[y_start:y_end, x_start:x_end]
            
            if roi_depth_estimated[roi_depth_estimated > 0].size > 0:
                closest_depth_m_estimated = np.nanmin(roi_depth_estimated[roi_depth_estimated > 0])
            else:
                closest_depth_m_estimated = VIS_MAX_DEPTH 
            
            # Display Estimated Depth
            estimated_text = f"Est: {closest_depth_m_estimated:.2f} m (R:{RADIUS}px)" 
            text_pos_est = (center_x - 50, center_y - RADIUS - 10) 
            cv2.putText(depth_draw_bgr, estimated_text, text_pos_est, cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
            
            # Display True Depth (Cone Min)
            true_text = f"True: {true_min_val:.2f} m (Noisy Min)"
            text_pos_true = (center_x - 50, center_y - RADIUS + 15) 
            cv2.putText(depth_draw_bgr, true_text, text_pos_true, cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)

            # U-TURN/Arrow Logic 
            UTURN_DISTANCE_THRESHOLD = 0.43 
            is_uturn_trigger = closest_depth_m_estimated <= UTURN_DISTANCE_THRESHOLD
            current_time = rospy.Time.now().to_sec()
            
            if is_uturn_trigger:
                if current_time >= self.uturn_expire_time:
                    self.uturn_expire_time = current_time + 3.0
                
                U_TURN_TEXT = "U-TURN"
                (text_w, text_h), baseline = cv2.getTextSize(U_TURN_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
                text_x = int((W - text_w) / 2)
                text_y = int((H + text_h) / 2)
                cv2.putText(depth_draw_bgr, U_TURN_TEXT, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 3, cv2.LINE_AA)
            
            elif current_time < self.uturn_expire_time:
                U_TURN_TEXT = "U-TURN"
                (text_w, text_h), baseline = cv2.getTextSize(U_TURN_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
                text_x = int((W - text_w) / 2)
                text_y = int((H + text_h) / 2)
                cv2.putText(depth_draw_bgr, U_TURN_TEXT, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 3, cv2.LINE_AA)
            
            else:
                # Arrow Drawing Logic (pointing to max depth/clearance)
                depths_for_max = np.nan_to_num(smooth_map_for_vis, nan=-1.0)
                avg_depth = np.mean(depths_for_max[depths_for_max > 0])
                min_depth_threshold = max(avg_depth * 1.05, 0.01)
                far_depth_mask = depths_for_max > min_depth_threshold
                masked_depths = np.where(far_depth_mask, depths_for_max, -1.0)

                max_idx = np.argmax(masked_depths)
                
                max_y, max_x = np.unravel_index(max_idx, masked_depths.shape)
                max_depth_val = masked_depths[max_y, max_x]
                
                target_tip_x, target_tip_y = (max_x, max_y) if max_depth_val > 0 else (center_x, center_y - 50) 
                
                # --- Apply EMA for Arrow Smoothing ---
                beta = self.ARROW_SMOOTHING_BETA
                if self.last_arrow_tip_x == -1: self.last_arrow_tip_x, self.last_arrow_tip_y = target_tip_x, target_tip_y
                smoothed_tip_x = int(beta * target_tip_x + (1 - beta) * self.last_arrow_tip_x)
                smoothed_tip_y = int(beta * target_tip_y + (1 - beta) * self.last_arrow_tip_y)
                self.last_arrow_tip_x, self.last_arrow_tip_y = smoothed_tip_x, smoothed_tip_y

                cv2.arrowedLine(depth_draw_bgr, CENTER, (smoothed_tip_x, smoothed_tip_y), RED, 3, tipLength=0.1)

        else: # sensor_is_active is False (COAST CLEAR or Dropout)
            COAST_TEXT = f"COAST CLEAR - >{self.SENSOR_MAX_RANGE:.1f}m"
            (text_w, text_h), baseline = cv2.getTextSize(COAST_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = int((W - text_w) / 2)
            text_y = int((H + text_h) / 2)
            cv2.putText(depth_draw_bgr, COAST_TEXT, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2, cv2.LINE_AA)
            
            self.last_arrow_tip_x, self.last_arrow_tip_y = -1, -1


        # Publish the drawn image
        self.pub_mono_draw.publish(self.bridge.cv2_to_imgmsg(depth_draw_bgr, "bgr8", depth_true_msg.header))
        rospy.loginfo("Visualized metric depth map published to /mono/depth/image_draw.")

        # ----------------------------------------------------------------------
        # 4. Comparison Image Generation and Publication
        # ----------------------------------------------------------------------
        
        valid_mask_true = np.isfinite(depth_true) & (depth_true > 0)
        valid_mask_mono = np.isfinite(smooth_map_for_vis) & (smooth_map_for_vis > 0)
        valid_mask = valid_mask_true & valid_mask_mono

        diff_map = np.abs(smooth_map_for_vis - depth_true)
        MAX_ERROR_M = 1.0
        diff_norm = np.clip(diff_map / MAX_ERROR_M, 0.0, 1.0)

        comparison_image_gray = (255 - (diff_norm * 255)).astype(np.uint8)
        
        comparison_image_gray[~valid_mask] = 127
        
        comparison_image_bgr = cv2.cvtColor(comparison_image_gray, cv2.COLOR_GRAY2BGR)

        self.pub_comparison.publish(self.bridge.cv2_to_imgmsg(comparison_image_bgr, "bgr8", depth_true_msg.header))
        rospy.loginfo("Comparison image published.")

if __name__ == '__main__':
    try:
        if MiDaS_MODEL is not None:
            node = MiDaSDepthEstimator()
            rospy.spin()
        else:
            rospy.logerr("MiDaS model failed to load. Exiting ROS node.")
    except rospy.ROSInterruptException:
        pass
