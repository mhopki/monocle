import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# PyTorch and MiDaS imports
try:
    import torch
    import torch.hub
    
    # --- MiDaS Model Setup ---
    MiDaS_MODEL_TYPE = "MiDaS_small"
    MiDaS_MODEL = torch.hub.load("intel-isl/MiDaS", MiDaS_MODEL_TYPE)
    
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MiDaS_MODEL.to(DEVICE)
    MiDaS_MODEL.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if MiDaS_MODEL_TYPE == "DPT_Large" or MiDaS_MODEL_TYPE == "DPT_Hybrid":
        TRANSFORM = midas_transforms.dpt_transform
    else: 
        TRANSFORM = midas_transforms.small_transform
    
    print(f"MiDaS model ({MiDaS_MODEL_TYPE}) loaded successfully on {DEVICE}")

except ImportError as e:
    print(f"ERROR: PyTorch or MiDaS dependencies not found: {e}")
    MiDaS_MODEL = None
    TRANSFORM = None
    DEVICE = None

class MiDaSDisparityChecker:
    """
    ROS node to perform MiDaS monocular depth estimation and use the raw, unit-less
    disparity value to check for close-range obstacles (collision avoidance).
    """
    
    def __init__(self):
        rospy.init_node('midas_disparity_safety_checker', anonymous=True)
        self.bridge = CvBridge()
        
        # --- Configurable Safety Parameter ---
        self.SAFETY_DISPARITY_THRESHOLD = 500 
        self.U_TURN_DISTANCE_THRESHOLD = 0.3 # Nominal distance for text display
        
        # --- State Variables for Smoothing and U-TURN logic ---
        self.last_disparity_smoothed = None 
        self.uturn_expire_time = 0.0 
        
        # --- Adaptive Fusion Parameters ---
        self.ALPHA_FAST = 0.5      # Default alpha for textured scenes (fast reaction)
        self.ALPHA_SLOW = 0.2      # Slow alpha for uniform scenes (stability, less jitter)
        self.UNIFORMITY_THRESHOLD = 50.0 # StdDev below this means the scene is textureless (low stability)
        self.SAFETY_PERCENTILE = 80 # Primary check: 80th percentile of center circle
        
        # --- Multi-Point Check Parameters for Flank Fusion ---
        self.PERIMETER_OFFSET = 190 
        self.SAMPLE_SIZE = 40      
        
        # --- Differential Corner Failsafe Parameter ---
        self.REVERSE_SAFETY_THRESHOLD = 90.0 # Set to 60.0 as requested
        self.CORNER_CHECK_RADIUS = 20    # Sample 40x40 area at the absolute center
        
        # --- Publishers ---
        self.pub_mono_raw = rospy.Publisher('/mono/depth/image_raw', Image, queue_size=1) 
        self.pub_mono_draw = rospy.Publisher('/mono/depth/image_draw', Image, queue_size=1)
        
        # --- Subscriber: MiDaS Inference (RGB -> Raw Disparity) ---
        self.sub_rgb = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_processing_callback, queue_size=1)

        rospy.loginfo("MiDaS Disparity Safety Checker initialized. Awaiting RGB frames.")

    def run_midas_inference(self, cv_image):
        """ Performs MiDaS inference and returns the raw disparity map (NumPy float32). """
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

    def get_disparity_at_point(self, disparity_map, center_x, center_y, size):
        """ Extracts a square ROI and returns the mean disparity. """
        H, W = disparity_map.shape
        half_size = size // 2
        
        x_start = max(0, center_x - half_size)
        x_end = min(W, center_x + half_size)
        y_start = max(0, center_y - half_size)
        y_end = min(H, center_y + half_size)

        roi = disparity_map[y_start:y_end, x_start:x_end]
        valid_roi = roi[np.isfinite(roi) & (roi > 0)]
        
        return np.mean(valid_roi) if valid_roi.size > 0 else 0.0

    def rgb_processing_callback(self, rgb_msg):
        """
        Runs MiDaS inference, applies EMA smoothing to disparity, and performs safety checks.
        """
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting RGB image: {e}")
            return

        depth_midas_raw = self.run_midas_inference(cv_rgb)
        H, W = depth_midas_raw.shape
        center_y, center_x = H // 2, W // 2

        # --- 1a. Calculate ROI for Safety Checks and Adaptive Fusion ---
        RADIUS = 200
        y_start = max(0, center_y - RADIUS)
        y_end = min(H, center_y + RADIUS)
        x_start = max(0, center_x - RADIUS)
        x_end = min(W, center_x + RADIUS)

        # Raw Estimated Disparity in the safety ROI
        roi_disparity_raw = depth_midas_raw[y_start:y_end, x_start:x_end]
        valid_roi_raw = roi_disparity_raw[np.isfinite(roi_disparity_raw) & (roi_disparity_raw > 0)]
        
        # --- 1b. Adaptive Fusion Logic (Dynamic Alpha) ---
        if valid_roi_raw.size > 10:
            std_dev_disp = np.std(valid_roi_raw)
        else:
            std_dev_disp = 0.0
            
        # Select alpha based on texture/stability (low StdDev -> slow alpha)
        textureless_scene = std_dev_disp < self.UNIFORMITY_THRESHOLD
        if textureless_scene:
            alpha = self.ALPHA_SLOW
        else:
            alpha = self.ALPHA_FAST
        
        # --- 1c. Temporal Smoothing (EMA) ---
        disparity_smoothed = depth_midas_raw.copy()
        
        if self.last_disparity_smoothed is not None:
            last_smoothed_aligned = cv2.resize(self.last_disparity_smoothed, (W, H), 
                                                interpolation=cv2.INTER_LINEAR)
            
            valid_mask = np.isfinite(depth_midas_raw) & (depth_midas_raw > 0)
            valid_mask_prev = np.isfinite(last_smoothed_aligned) & (last_smoothed_aligned > 0)
            common_valid_mask = valid_mask & valid_mask_prev
            
            # Fuse the disparities using the calculated dynamic alpha
            disparity_smoothed[common_valid_mask] = (alpha * depth_midas_raw[common_valid_mask] + 
                                                      (1 - alpha) * last_smoothed_aligned[common_valid_mask])

        # Store the current smoothed state
        self.last_disparity_smoothed = disparity_smoothed.copy()
        
        # --- 2. Publish RAW Disparity for visualization (8-bit normalized) ---
        min_val_raw, max_val_raw = disparity_smoothed.min(), disparity_smoothed.max()
        if max_val_raw > min_val_raw:
            normalized_map_raw = (disparity_smoothed - min_val_raw) / (max_val_raw - min_val_raw)
            depth_midas_vis = (normalized_map_raw * 255).astype(np.uint8)
        else:
            depth_midas_vis = np.zeros_like(disparity_smoothed, dtype=np.uint8)
            
        self.pub_mono_raw.publish(self.bridge.cv2_to_imgmsg(depth_midas_vis, "mono8", rgb_msg.header))

        # --- 3. Visualization and Drawing (/mono/depth/image_draw) ---

        CENTER = (center_x, center_y)

        # --- Colorized Map Generation ---
        
        VIS_MIN_DISP = np.nanmin(disparity_smoothed[disparity_smoothed > 0])
        VIS_MAX_DISP = np.nanmax(disparity_smoothed)
        
        if (VIS_MAX_DISP - VIS_MIN_DISP) < 1e-6:
            VIS_MAX_DISP = VIS_MIN_DISP + 1e-6

        visual_map = np.clip(disparity_smoothed, VIS_MIN_DISP, VIS_MAX_DISP)
        
        normalized_vis_map = (visual_map - VIS_MIN_DISP) / (VIS_MAX_DISP - VIS_MIN_DISP)
        depth_draw_8bit = (normalized_vis_map * 255).astype(np.uint8)
        depth_draw_bgr = cv2.applyColorMap(depth_draw_8bit, cv2.COLORMAP_VIRIDIS)
        
        # --- PRIMARY ROBUST SAFETY CHECK (Percentile) ---
        roi_disparity_smoothed = disparity_smoothed[y_start:y_end, x_start:x_end]
        valid_roi_smoothed = roi_disparity_smoothed[np.isfinite(roi_disparity_smoothed)]
        
        if valid_roi_smoothed.size > 0:
            percentile_disparity = np.nanpercentile(valid_roi_smoothed, self.SAFETY_PERCENTILE) 
        else:
            percentile_disparity = 0.0
        
        # --- CORNER VOID DETECTION (Center Check) ---
        y_center_start = max(0, center_y - self.CORNER_CHECK_RADIUS)
        y_center_end = min(H, center_y + self.CORNER_CHECK_RADIUS)
        x_center_start = max(0, center_x - self.CORNER_CHECK_RADIUS)
        x_center_end = min(W, center_x + self.CORNER_CHECK_RADIUS)

        center_roi = disparity_smoothed[y_center_start:y_center_end, x_center_start:x_center_end]
        center_avg_disparity = np.mean(center_roi[np.isfinite(center_roi) & (center_roi > 0)])
        
        # Draw the center void check box (for visualization/debugging)
        RED_LIGHT_GRAY = (127, 127, 255) # Light Red/Orange
        cv2.rectangle(depth_draw_bgr, (x_center_start, y_center_start), (x_center_end, y_center_end), RED_LIGHT_GRAY, 2)
        
        # --- SECONDARY CORNER SAFETY CHECK (Perimeter Average) ---
        
        # Define the four points near the perimeter of the safety circle
        perimeter_points = [
            (center_x + self.PERIMETER_OFFSET, center_y), # Right
            (center_x - self.PERIMETER_OFFSET, center_y), # Left
            (center_x, center_y + self.PERIMETER_OFFSET), # Bottom
            (center_x, center_y - self.PERIMETER_OFFSET), # Top
        ]
        
        perimeter_triggered = False
        max_perimeter_disparity = 0.0
        
        for px, py in perimeter_points:
            avg_disp = self.get_disparity_at_point(disparity_smoothed, px, py, self.SAMPLE_SIZE)
            max_perimeter_disparity = max(max_perimeter_disparity, avg_disp)
            
            # Draw small green circles at the check points
            GREEN = (0, 255, 0)
            cv2.circle(depth_draw_bgr, (px, py), self.SAMPLE_SIZE // 2, GREEN, 5) 
            
            if avg_disp >= self.SAFETY_DISPARITY_THRESHOLD:
                perimeter_triggered = True

        # --- Combine Checks (Spatial Fusion) ---
        
        # 3. DIFFERENTIAL CORNER FAILSAFE:
        disparity_gap = max_perimeter_disparity - center_avg_disparity
        
        # NEW LOGIC: Trigger if the gap is large AND the scene is textureless (low StdDev)
        differential_triggered = (disparity_gap >= self.REVERSE_SAFETY_THRESHOLD) and textureless_scene
             
        # The U-TURN triggers if ANY of the three robust checks fail:
        is_uturn_trigger = (percentile_disparity >= self.SAFETY_DISPARITY_THRESHOLD) or perimeter_triggered or differential_triggered
        
        # Draw blue circle around the radius
        BLUE = (255, 0, 0) # BGR
        cv2.circle(depth_draw_bgr, CENTER, RADIUS, BLUE, 2)

        # --- Display Closest Disparity Text (RED) ---
        RED = (0, 0, 255) # BGR
        
        # Adaptive Fusion Status
        alpha_text = f"Alpha: {alpha:.1f} (StdDev: {std_dev_disp:.2f})"
        
        # Primary decision text
        estimated_text_pct = f"{self.SAFETY_PERCENTILE}th Pctl: {percentile_disparity:.4f}"
        estimated_text_perim = f"Max Perim Avg: {max_perimeter_disparity:.4f}"
        
        # Differential check text
        estimated_text_void = f"Center Avg: {center_avg_disparity:.4f} | Gap (Diff Check): {disparity_gap:.4f}"
        
        # Position text
        text_pos_est = (center_x - 100, center_y - RADIUS - 20) 
        
        cv2.putText(depth_draw_bgr, alpha_text, text_pos_est, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2, cv2.LINE_AA)
        cv2.putText(depth_draw_bgr, estimated_text_pct, (text_pos_est[0], text_pos_est[1] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        cv2.putText(depth_draw_bgr, estimated_text_perim, (text_pos_est[0], text_pos_est[1] + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        cv2.putText(depth_draw_bgr, estimated_text_void, (text_pos_est[0], text_pos_est[1] + 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)


        # --- Arrow/U-TURN Logic ---
        
        current_time = rospy.Time.now().to_sec()
        
        if is_uturn_trigger:
            if current_time >= self.uturn_expire_time:
                self.uturn_expire_time = current_time + 3.0
            
            U_TURN_TEXT = f"U-TURN ({self.U_TURN_DISTANCE_THRESHOLD:.1f}m Risk)"
            (text_w, text_h), baseline = cv2.getTextSize(U_TURN_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
            text_x = int((W - text_w) / 2)
            text_y = int((H + text_h) / 2)
            cv2.putText(depth_draw_bgr, U_TURN_TEXT, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 3, cv2.LINE_AA)
            
        elif current_time < self.uturn_expire_time:
            # Force display of U-TURN until timer expires
            U_TURN_TEXT = f"U-TURN ({self.U_TURN_DISTANCE_THRESHOLD:.1f}m Risk)"
            (text_w, text_h), baseline = cv2.getTextSize(U_TURN_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
            text_x = int((W - text_w) / 2)
            text_y = int((H + text_h) / 2)
            cv2.putText(depth_draw_bgr, U_TURN_TEXT, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 3, cv2.LINE_AA)

        else:
            # Draw the normal arrow pointing to the lowest disparity (farthest point)
            
            # Find the minimum disparity (farthest point) outside the safety radius
            min_disp_idx = np.argmin(disparity_smoothed)
            min_y, min_x = np.unravel_index(min_disp_idx, disparity_smoothed.shape)
            
            ARROW_TIP = (min_x, min_y)
            cv2.arrowedLine(depth_draw_bgr, CENTER, ARROW_TIP, RED, 3, tipLength=0.1)

        # Publish the drawn image
        self.pub_mono_draw.publish(self.bridge.cv2_to_imgmsg(depth_draw_bgr, "bgr8", rgb_msg.header))
        rospy.loginfo_throttle(1.0, "Disparity safety check image published to /mono/depth/image_draw.")

if __name__ == '__main__':
    try:
        if MiDaS_MODEL is not None:
            node = MiDaSDisparityChecker()
            rospy.spin()
        else:
            rospy.logerr("MiDaS model failed to load. Exiting ROS node.")
    except rospy.ROSInterruptException:
        pass
