#!/usr/bin/env python3

# ROS 1 Python Node for Kernel-Based Template Matching Depth Tracking

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from geometry_msgs.msg import Point # Message for publishing x, y coordinates

class TemplateDepthTracker:
    """
    Decoupled tracker that switches from Color-Aligned setup (needed for Zero-Shot) 
    to the faster RAW DEPTH pipeline for continuous tracking.
    
    This version implements a ROTATED KERNEL BANK for rotation robustness.
    """

    MAX_DEPTH_VALUE = 1000.0  

    def __init__(self):
        rospy.init_node('object_tracker_node', anonymous=True)
        self.bridge = CvBridge()
        
        # --- Configuration Parameters ---
        self.publish_depth_vis = rospy.get_param('~publish_depth_vis', True)
        self.publish_color_vis = rospy.get_param('~publish_color_vis', True)
        self.depth_tolerance = rospy.get_param('~depth_tolerance', 0.15) 
        self.tracking_mode = rospy.get_param('~tracking_mode', 'combined').lower()
        
        # NEW VISUALIZATION CONTROLS
        self.VIS_ALWAYS_PUBLISH = rospy.get_param('~vis_always_publish', True)
        self.VIS_MIN_SCORE_THRESHOLD = rospy.get_param('~vis_min_score_threshold', 0.6) # Default threshold for visualization
        
        # Threshold for accepting any match (0.0 to 1.0)
        self.MIN_SCORE_THRESHOLD = rospy.get_param('~score_threshold', 0.6) 
        
        # Rotations to test (e.g., [0, 90, 180, 270])
        self.rotation_degrees = rospy.get_param('~rotation_degrees', [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        rospy.loginfo(f"Rotational Robustness: {len(self.rotation_degrees)} kernels per mode.")
        rospy.loginfo(f"Min Match Score Threshold: {self.MIN_SCORE_THRESHOLD:.2f}")

        # --- Fusion Parameters (for 'combined' mode) ---
        self.WEIGHT_INV = rospy.get_param('~weight_inverted', 0.5) 
        self.WEIGHT_POS = rospy.get_param('~weight_positive', 0.5)
        self.MAX_FUSION_DISTANCE_PIXELS = rospy.get_param('~fusion_distance_pixels', 30)
        self.WEIGHT_INV = self.WEIGHT_INV / (self.WEIGHT_INV + self.WEIGHT_POS)
        self.WEIGHT_POS = 1.0 - self.WEIGHT_INV
        
        # --- Intrinsics State (Required to remap first pixel) ---
        self.color_fx, self.color_fy, self.color_cx, self.color_cy = 0.0, 0.0, 0.0, 0.0
        self.depth_fx, self.depth_fy, self.depth_cx, self.depth_cy = 0.0, 0.0, 0.0, 0.0
        self.intrinsics_ready = False
        
        # --- Tracking State ---
        self.kernel_inv = None # Base kernel for INVERTED mode (32FC1)
        self.kernel_pos = None # Base kernel for POSITIVE mode (8UC1)
        self.kernel_pos_mask = None # Base mask for POSITIVE mode (32FC1)
        
        # NEW: Lists to store all rotated versions
        self.kernel_bank_inv = []
        self.kernel_bank_pos = []
        self.kernel_bank_pos_masks = []
        
        self.kernel_avg_depth = None    
        self.kernel_w = 0               
        self.kernel_h = 0               
        self.setup_complete = False

        # Publishers
        self.tracked_depth_pub = rospy.Publisher("/tracked_raw_depth_visualization", Image, queue_size=1)
        self.tracked_depth_pub_pos = rospy.Publisher("/tracked_raw_depth_visualization_pos", Image, queue_size=1)
        self.tracked_color_pub = rospy.Publisher("/tracked_color_visualization", Image, queue_size=1)
        self.kernel_pub = rospy.Publisher("/tracking_kernel", Image, queue_size=1, latch=True) 
        self.kernel_pos_pub = rospy.Publisher("/tracking_kernel_positive", Image, queue_size=1, latch=True)

        self.location_pub = rospy.Publisher("/tracked_pixel_location", Point, queue_size=1) 

        # STAGE 1: Setup - Subscribe to Info Topics
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self._color_info_callback, queue_size=1)
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self._depth_info_callback, queue_size=1)

        # Start a timer to periodically check if setup is ready
        self.setup_timer = rospy.Timer(rospy.Duration(0.5), self._setup_check_timer)

        rospy.loginfo("STAGE 1: Waiting for Intrinsics and /segmented_depth_image...")

    # --- INTRINSICS CALLBACKS ---

    def _check_intrinsics_ready(self):
        if self.color_fx != 0.0 and self.depth_fx != 0.0:
             self.intrinsics_ready = True
             rospy.loginfo("Intrinsics for Color and Depth frames successfully loaded.")
        
    def _color_info_callback(self, info_msg):
        if self.color_fx == 0.0:
            self.color_fx = info_msg.K[0]
            self.color_fy = info_msg.K[4]
            self.color_cx = info_msg.K[2]
            self.color_cy = info_msg.K[5]
            self._check_intrinsics_ready()

    def _depth_info_callback(self, info_msg):
        if self.depth_fx == 0.0:
            self.depth_fx = info_msg.K[0]
            self.depth_fy = info_msg.K[4]
            self.depth_cx = info_msg.K[2]
            self.depth_cy = info_msg.K[5]
            self._check_intrinsics_ready()

    def _setup_check_timer(self, event):
        """Timer event to poll for readiness and start kernel acquisition."""
        if self.intrinsics_ready and not self.setup_complete:
            self.setup_timer.shutdown()
            self.kernel_acquisition() # Call the blocking acquisition function

    # --- UTILITY METHODS ---

    def convert_to_32fc1_meters(self, img_msg, encoding):
        """Converts image message (16UC1 or 32FC1) to a 32FC1 NumPy array (depth in meters)."""
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

    def _map_color_pixel_to_depth_pixel(self, color_pixel_x, color_pixel_y, depth_z):
        """
        Remaps a pixel from the color frame to the depth frame using intrinsic matrices.
        """
        if not self.intrinsics_ready:
            return None, None

        X_c = (color_pixel_x - self.color_cx) * depth_z / self.color_fx
        Y_c = (color_pixel_y - self.color_cy) * depth_z / self.color_fy
        Z_c = depth_z
        
        depth_pixel_x = (X_c / Z_c) * self.depth_fx + self.depth_cx
        depth_pixel_y = (Y_c / Z_c) * self.depth_fy + self.depth_cy

        return int(depth_pixel_x), int(depth_pixel_y)
        
    def _rotate_and_bank_kernel(self, base_kernel, base_mask, is_32f=False):
        """
        Generates rotated versions of a kernel and its mask for the kernel bank.
        Returns (list_of_kernels, list_of_masks)
        """
        bank = []
        mask_bank = []
        (h, w) = base_kernel.shape
        (cX, cY) = (w // 2, h // 2)

        for angle in self.rotation_degrees:
            # 1. Calculate the rotation matrix
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0) # Negative angle for standard rotation
            
            # 2. Rotate the kernel
            # Use same interpolation for all
            if is_32f: # For Inverted Kernel (32FC1)
                rotated_kernel = cv2.warpAffine(base_kernel, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            else: # For Positive Kernel (8UC1)
                rotated_kernel = cv2.warpAffine(base_kernel, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            bank.append(rotated_kernel)
            
            # 3. Rotate the mask if provided (only needed for positive 8uc1 matching)
            if base_mask is not None:
                 # Masks must be rotated with NEAREST interpolation to preserve binary structure
                 rotated_mask = cv2.warpAffine(base_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                 mask_bank.append(rotated_mask)
                 
        return bank, mask_bank

    def _generate_positive_kernel(self, segmented_depth_array, y_min, y_max, x_min, x_max, valid_object_pixels):
        """
        Generates an 8-bit grayscale kernel for texture matching and its 32FC1 mask.
        """
        kernel_patch = segmented_depth_array[y_min:y_max+1, x_min:x_max+1].copy()
        
        avg_depth = np.nanmean(valid_object_pixels)

        pos_mask = (~np.isnan(kernel_patch)).astype(np.float32)
        
        kernel_patch[np.isnan(kernel_patch)] = self.MAX_DEPTH_VALUE 
        
        valid_depths = kernel_patch[kernel_patch != self.MAX_DEPTH_VALUE]
        
        if valid_depths.size == 0:
            rospy.logwarn("Positive kernel has no valid depth pixels for 8-bit normalization.")
            return None, None, None

        min_d = np.nanmin(valid_depths)
        max_d = np.nanmax(valid_depths)
        
        if max_d > min_d:
            kernel_8u = cv2.normalize(kernel_patch, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask=pos_mask.astype(np.uint8))
            kernel_8u[kernel_8u == 0] = 0
            kernel_8u[pos_mask.astype(np.uint8) == 0] = 1 
        else:
            kernel_8u = np.zeros_like(kernel_patch, dtype=np.uint8)

        return kernel_8u, pos_mask, avg_depth

    def _generate_inverted_kernel(self, segmented_depth_array, raw_depth_array, y_min, y_max, x_min, x_max):
        """Generates a 32FC1 kernel that masks the object and matches the background depth profile."""
        
        kernel_patch_raw = raw_depth_array[y_min:y_max+1, x_min:x_max+1].copy()

        object_mask_patch = ~np.isnan(segmented_depth_array[y_min:y_max+1, x_min:x_max+1])
        
        inverted_kernel = kernel_patch_raw.copy()
        inverted_kernel[object_mask_patch] = np.nan
        
        background_pixels = inverted_kernel[~np.isnan(inverted_kernel)]
        if background_pixels.size == 0:
            rospy.logwarn("Inverted kernel has no valid background pixels.")
            return None, None
            
        avg_depth = np.nanmean(background_pixels)

        inverted_kernel[np.isnan(inverted_kernel)] = 0.0
        
        return inverted_kernel, avg_depth

    def kernel_acquisition(self):
        """
        BLOCKING Acquisition: Fetches the segmented message and the corresponding raw depth message
        to generate the kernel, ensuring the setup phase is robust.
        """
        rospy.loginfo("Waiting for segmented depth message...")
        try:
            # 1. BLOCK: Fetch the segmented message (this is the static kernel source)
            segmented_depth_data = rospy.wait_for_message("/segmented_depth_image", Image, timeout=10.0)
            rospy.loginfo("Received /segmented_depth_image.")
            
            # 2. BLOCK: Fetch the current raw depth image
            raw_depth_data = rospy.wait_for_message("/camera/depth/image_rect_raw", Image, timeout=10.0)
            rospy.loginfo("Received /camera/depth/image_rect_raw.")

        except rospy.ROSException:
            rospy.logerr("Setup failed: Timeout waiting for kernel or raw depth messages. Shutting down.")
            rospy.signal_shutdown("Setup failed.")
            return

        try:
            segmented_depth_array = self.convert_to_32fc1_meters(segmented_depth_data, segmented_depth_data.encoding)
            raw_depth_array = self.convert_to_32fc1_meters(raw_depth_data, raw_depth_data.encoding)
            
            object_mask_full = ~np.isnan(segmented_depth_array)
            valid_object_pixels = segmented_depth_array[object_mask_full]

            if valid_object_pixels.size == 0:
                rospy.logerr("Segmented image is empty. Kernel creation failed.")
                return

            # --- 1. DETERMINE PIXEL REMAPPING FROM ALIGNED/COLOR FRAME TO RAW DEPTH FRAME ---
            y_coords, x_coords = np.where(object_mask_full)
            center_pixel_color_frame_x = np.mean(x_coords)
            center_pixel_color_frame_y = np.mean(y_coords)
            center_depth_aligned = np.mean(valid_object_pixels) 

            # Convert the Center of the Object from the Color Frame to the Raw Depth Frame
            depth_x, depth_y = self._map_color_pixel_to_depth_pixel(
                center_pixel_color_frame_x, 
                center_pixel_color_frame_y, 
                center_depth_aligned
            )
            
            # --- 2. EXTRACT KERNEL BOUNDS ---
            
            mask_w = np.max(x_coords) - np.min(x_coords) + 1
            mask_h = np.max(y_coords) - np.min(y_coords) + 1

            self.kernel_w = mask_w
            self.kernel_h = mask_h
            
            # Calculate CLAMPED, RECENTERED bounds on the RAW image
            raw_h, raw_w = raw_depth_array.shape
            y_min_raw = int(depth_y - mask_h / 2)
            y_max_raw = int(depth_y + mask_h / 2)
            x_min_raw = int(depth_x - mask_w / 2)
            x_max_raw = int(depth_x + mask_w / 2)

            y_min_clamped = max(0, y_min_raw)
            y_max_clamped = min(raw_h, y_max_raw)
            x_min_clamped = max(0, x_min_raw)
            x_max_clamped = min(raw_w, x_max_raw)
            
            if (y_max_clamped - y_min_clamped < 10) or (x_max_clamped - x_min_clamped < 10):
                 rospy.logerr("Clamped kernel is too small. Setup failed.")
                 return
            
            # --- 3. GENERATE BASE KERNELS ---
            avg_depth_inv = None
            avg_depth_pos = None

            if self.tracking_mode in ['inverted', 'combined']:
                 self.kernel_inv, avg_depth_inv = self._generate_inverted_kernel(
                      segmented_depth_array, raw_depth_array, y_min_clamped, y_max_clamped, x_min_clamped, x_max_clamped
                 )
            if self.tracking_mode in ['positive', 'combined']:
                 self.kernel_pos, self.kernel_pos_mask, avg_depth_pos = self._generate_positive_kernel(
                      segmented_depth_array, y_min_clamped, y_max_clamped, x_min_clamped, x_max_clamped, valid_object_pixels
                 )
            
            # Determine overall reference depth
            if self.tracking_mode == 'inverted' and avg_depth_inv is not None:
                 self.kernel_avg_depth = avg_depth_inv 
                 log_type = "INVERTED"
            elif self.tracking_mode == 'positive' and avg_depth_pos is not None:
                 self.kernel_avg_depth = avg_depth_pos
                 log_type = "POSITIVE"
            elif self.tracking_mode == 'combined' and avg_depth_pos is not None:
                 self.kernel_avg_depth = avg_depth_pos # Combined mode uses positive avg depth for filtering
                 log_type = "COMBINED"
            else:
                 rospy.logerr("Kernel generation failed in chosen mode.")
                 return
                 
            # --- 4. POPULATE ROTATED KERNEL BANKS ---
            
            if self.kernel_inv is not None:
                # Bank Inverted Kernels (32FC1, no mask needed)
                self.kernel_bank_inv, _ = self._rotate_and_bank_kernel(
                    self.kernel_inv, None, is_32f=True
                )
                
            if self.kernel_pos is not None:
                # Bank Positive Kernels (8UC1) and their masks (32FC1)
                self.kernel_bank_pos, self.kernel_bank_pos_masks = self._rotate_and_bank_kernel(
                    self.kernel_pos, self.kernel_pos_mask, is_32f=False
                )


            # --- FINAL VISUALIZATION & SWITCH ---
            
            # 1. Publish primary kernel visualization (for /tracking_kernel)
            kernel_to_vis = self.kernel_inv if self.kernel_inv is not None else self.kernel_pos
            kernel_vis_8u = cv2.normalize(kernel_to_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            kernel_msg = self.bridge.cv2_to_imgmsg(kernel_vis_8u, "mono8")
            self.kernel_pub.publish(kernel_msg)
            
            # 2. Publish POSITIVE kernel visualization (for /tracking_kernel_positive)
            if self.kernel_pos is not None:
                 kernel_pos_vis_msg = self.bridge.cv2_to_imgmsg(self.kernel_pos, "mono8")
                 kernel_pos_vis_msg.header = segmented_depth_data.header
                 self.kernel_pos_pub.publish(kernel_pos_vis_msg)

            self.setup_complete = True
            rospy.loginfo("STAGE 1 COMPLETE: Kernel acquired. Starting tracking loop.")
            
            self.start_tracking_loop()

        except Exception as e:
            rospy.logerr(f"Error during critical kernel acquisition phase: {e}")

    def start_tracking_loop(self):
        """
        Initializes the dedicated message filter for the two high-frequency streams.
        """
        rospy.loginfo("STAGE 2 STARTING: Initiating high-speed tracking loop...")
        
        # FIX: Subscribing to RAW DEPTH /image_raw topic
        raw_depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)

        # Synchronize only the two necessary streams (raw depth and color)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [raw_depth_sub, color_sub], 
            queue_size=10, 
            slop=0.05 
        )
        self.ts.registerCallback(self.tracking_callback)
        rospy.loginfo("Tracking loop is active on RAW DEPTH stream.")


    def tracking_callback(self, raw_depth_data, color_data):
        """
        The continuous, high-frequency callback executed upon receiving synchronized
        raw depth and color images. Performs rotationally robust template matching and visualization.
        """
        # Define default center location (mid-screen) for visualization failure mode
        H_def, W_def = (raw_depth_data.height, raw_depth_data.width)
        center_x_def, center_y_def = (W_def // 2, H_def // 2)
        top_left_def = (center_x_def - self.kernel_w // 2, center_y_def - self.kernel_h // 2)
        bottom_right_def = (top_left_def[0] + self.kernel_w, top_left_def[1] + self.kernel_h)
        
        publish_location_update = False
        
        if not self.setup_complete:
            pass # Skip tracking logic
        elif self.kernel_inv is None and self.kernel_pos is None:
            rospy.logwarn_throttle(5.0, "Tracking callback received data, but no kernel is set. Waiting.")
        
        else:
            # --- ATTEMPT ROTATIONALLY ROBUST TRACKING ---
            best_score = -1.0
            best_loc = (0, 0)
            best_kernel = None
            best_angle = 0.0 # Store the best matching angle
            tracking_mode = self.tracking_mode

            try: # <--- Start of main tracking and visualization try block
                # Get the pristine RAW depth array (32FC1 meters)
                raw_depth_array_pristine = self.convert_to_32fc1_meters(raw_depth_data, raw_depth_data.encoding)
                if raw_depth_array_pristine is None:
                    raw_depth_array_pristine = np.zeros((H_def, W_def), dtype=np.float32) # Use blank array for vis
                
                # --- CONTINUOUS TRACKING ---
                tracking_array = raw_depth_array_pristine.copy()

                # 1. Dynamic Depth Thresholding (INVERSE FILTER)
                min_target_depth = self.kernel_avg_depth - self.depth_tolerance
                max_target_depth = self.kernel_avg_depth + self.depth_tolerance
                if max_target_depth > 2.65: max_target_depth = 2.65

                within_bounds_mask = (tracking_array >= min_target_depth) & \
                                   (tracking_array <= max_target_depth)
                
                tracking_array[within_bounds_mask] = 0.0
                tracking_array[np.isnan(tracking_array)] = 0.0
                
                H, W = tracking_array.shape
                
                # Initialize storage for individual match results
                res_pos = {'score': -1.0, 'loc': (0, 0), 'kernel': None, 'angle': 0.0} 
                res_inv = {'score': -1.0, 'loc': (0, 0), 'kernel': None, 'angle': 0.0}
                
                # 2. Perform Matching based on Mode (Rotational Search)
                
                # --- POSITIVE Matching (8UC1 Data) ---
                if tracking_mode in ['positive', 'combined'] and self.kernel_bank_pos:
                    tracking_array_8u = cv2.normalize(tracking_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    for i, kernel in enumerate(self.kernel_bank_pos):
                        h, w = kernel.shape
                        if H >= h and W >= w:
                            result_pos_i = cv2.matchTemplate(
                                tracking_array_8u, 
                                kernel,   
                                cv2.TM_CCOEFF_NORMED,
                                mask=self.kernel_bank_pos_masks[i].astype(np.uint8) 
                            )
                            _, max_val_i, _, max_loc_i = cv2.minMaxLoc(result_pos_i)
                            
                            if max_val_i > res_pos['score']:
                                res_pos['score'] = max_val_i
                                res_pos['loc'] = max_loc_i
                                res_pos['kernel'] = kernel 
                                res_pos['angle'] = self.rotation_degrees[i] # Store the angle

                
                # --- INVERTED Matching (32FC1 Data) ---
                if tracking_mode in ['inverted', 'combined'] and self.kernel_bank_inv:
                    for i, kernel in enumerate(self.kernel_bank_inv):
                        h, w = kernel.shape
                        if H >= h and W >= w:
                            result_inv_i = cv2.matchTemplate(tracking_array, kernel, cv2.TM_CCOEFF_NORMED)
                            _, max_val_i, _, max_loc_i = cv2.minMaxLoc(result_inv_i)
                            
                            if max_val_i > res_inv['score']:
                                res_inv['score'] = max_val_i
                                res_inv['loc'] = max_loc_i
                                res_inv['kernel'] = kernel 
                                res_inv['angle'] = self.rotation_degrees[i] # Store the angle
                
                
                # 3. Determine Final Location (Fusion/Fallback)
                
                best_match = res_pos if res_pos['score'] > res_inv['score'] else res_inv
                
                # Use the best single result as the fallback location
                best_score = best_match['score']
                best_loc = best_match['loc']
                best_kernel = best_match['kernel']
                best_angle = best_match['angle'] # Initialize best angle
                
                if tracking_mode == 'combined':
                    score_pos = res_pos['score']
                    score_inv = res_inv['score']
                    
                    valid_pos = score_pos > self.MIN_SCORE_THRESHOLD # Use threshold for validity check
                    valid_inv = score_inv > self.MIN_SCORE_THRESHOLD # Use threshold for validity check
                    
                    if valid_pos and valid_inv:
                        center_pos_x = res_pos['loc'][0] + self.kernel_w // 2
                        center_pos_y = res_pos['loc'][1] + self.kernel_h // 2
                        center_inv_x = res_inv['loc'][0] + self.kernel_w // 2
                        center_inv_y = res_inv['loc'][1] + self.kernel_h // 2
                        
                        distance = math.sqrt(
                            (center_pos_x - center_inv_x)**2 + 
                            (center_pos_y - center_inv_y)**2
                        )

                        if distance <= self.MAX_FUSION_DISTANCE_PIXELS:
                            rospy.logdebug(f"Fusion accepted. Distance: {distance:.1f}px")
                            
                            fused_x = int(self.WEIGHT_POS * res_pos['loc'][0] + self.WEIGHT_INV * res_inv['loc'][0])
                            fused_y = int(self.WEIGHT_POS * res_pos['loc'][1] + self.WEIGHT_INV * res_inv['loc'][1])
                            
                            best_loc = (fused_x, fused_y)
                            best_kernel = res_pos['kernel'] if res_pos['score'] > res_inv['score'] else res_inv['kernel']
                            best_score = (score_pos * self.WEIGHT_POS) + (score_inv * self.WEIGHT_INV) 
                            publish_location_update = True
                            
                            # Use the angle from the highest scoring kernel in the fused result
                            best_angle = res_pos['angle'] if res_pos['score'] > res_inv['score'] else res_inv['angle']
                            
                        else:
                            rospy.logwarn_throttle(1.0, f"Fusion rejected. Distance: {distance:.1f}px. Publishing visualization only.")
                            # Do not publish pixel update
                            
                    elif valid_pos or valid_inv:
                         # One kernel passed the score threshold but they weren't close enough for fusion.
                         # Fall back to the single best-scoring kernel that passed the threshold.
                         best_match = res_pos if res_pos['score'] > res_inv['score'] and res_pos['score'] > self.MIN_SCORE_THRESHOLD else res_inv
                         if best_match['score'] > self.MIN_SCORE_THRESHOLD:
                            best_loc = best_match['loc']
                            best_kernel = best_match['kernel']
                            best_score = best_match['score']
                            best_angle = best_match['angle']
                            publish_location_update = True
                         else:
                            rospy.logwarn_throttle(1.0, "Individual kernel match failed minimum score threshold.")
                         
                    else:
                         rospy.logwarn_throttle(1.0, "Combined tracking failed: neither kernel yielded a valid match (score < threshold). Publishing visualization only.")
                         # Do not publish pixel update
                
                # --- Individual Mode Publication Check ---
                elif best_score > self.MIN_SCORE_THRESHOLD:
                    publish_location_update = True

                else: # Default if single mode match is below threshold
                     rospy.logwarn_throttle(1.0, f"Tracking mode {tracking_mode} failed to find a valid match (score < {self.MIN_SCORE_THRESHOLD:.2f}). Publishing visualization only.")
                     # Do not publish pixel update


                # --- Final Coordinate Assignment (for visualization) ---
                if best_kernel is not None and best_loc != (0,0):
                    top_left = best_loc
                    h, w = self.kernel_h, self.kernel_w 
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    center_x = top_left[0] + w // 2
                    center_y = top_left[1] + h // 2
                else:
                    # Use default center/box for visualization if tracking failed
                    center_x, center_y = center_x_def, center_y_def
                    top_left = top_left_def
                    bottom_right = bottom_right_def

                # --- PIXEL LOCATION PUBLICATION (Only if successful match/fusion was accepted) ---
                if publish_location_update:
                    location_msg = Point()
                    location_msg.x = center_x
                    location_msg.y = center_y
                    self.location_pub.publish(location_msg)


                # 5. Visualization on Depth (Always published for monitoring)
                vis_pub_inv = self.tracked_depth_pub 
                vis_pub_pos = self.tracked_depth_pub_pos
                
                if self.publish_depth_vis:
                    
                    # Helper function to prepare and publish a visualization for a specific depth array
                    def _publish_depth_visualization(depth_array_32f, publisher, header, current_center_x, current_center_y, current_top_left, current_bottom_right, angle_deg):
                        # Determine if we should publish based on the toggle settings
                        should_publish = self.VIS_ALWAYS_PUBLISH or (publish_location_update and best_score >= self.VIS_MIN_SCORE_THRESHOLD)

                        if not should_publish:
                            return

                        # 1. Normalize to 8-bit grayscale for display
                        vis_copy = depth_array_32f.copy()
                        min_depth_vis = np.nanmin(vis_copy[vis_copy > 0]) if np.sum(vis_copy > 0) > 0 else 0
                        max_depth_vis = np.nanmax(vis_copy) if np.sum(vis_copy > 0) > 0 else 1

                        if max_depth_vis > min_depth_vis:
                            normalized_depth = cv2.normalize(vis_copy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        else:
                            normalized_depth = np.zeros_like(vis_copy, dtype=np.uint8)

                        tracked_depth_cv = cv2.cvtColor(normalized_depth, cv2.COLOR_GRAY2BGR)

                        # 2. Draw Markers and Rotated Box
                        
                        # Define the center and box parameters
                        rect_center = (float(current_center_x), float(current_center_y))
                        rect_size = (float(self.kernel_w), float(self.kernel_h)) # <-- FIX: Ensure size is float
                        rect_angle = -angle_deg # OpenCV requires angle in degrees, often negated

                        # Draw Rotated Box (using RotatedRect definition)
                        rect = (rect_center, rect_size, rect_angle)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        
                        # Change box color if location update was suppressed
                        box_color = (255, 255, 0) if publish_location_update else (0, 255, 255) # Cyan if failed, Yellow if active
                        cv2.polylines(tracked_depth_cv, [box], True, box_color, 2)
                        
                        # Draw Markers (standard cross and center dot)
                        marker_size = 20
                        cv2.line(tracked_depth_cv, (current_center_x, current_center_y - marker_size), (current_center_x, current_center_y + marker_size), (0, 0, 255), 2)
                        cv2.line(tracked_depth_cv, (current_center_x - marker_size, current_center_y), (current_center_x + marker_size, current_center_y), (0, 0, 255), 2)
                        cv2.circle(tracked_depth_cv, (current_center_x, current_center_y), 5, (0, 255, 0), -1)

                        tracked_depth_msg = self.bridge.cv2_to_imgmsg(tracked_depth_cv, "bgr8")
                        tracked_depth_msg.header = header
                        publisher.publish(tracked_depth_msg)


                    # A. Calculate Direct Filtered Array (ONLY OBJECT VISIBLE)
                    min_target_depth = self.kernel_avg_depth - self.depth_tolerance
                    max_target_depth = self.kernel_avg_depth + self.depth_tolerance

                    direct_mask = (raw_depth_array_pristine >= min_target_depth) & \
                                  (raw_depth_array_pristine <= max_target_depth)
                    
                    direct_filtered_array = raw_depth_array_pristine * direct_mask.astype(np.float32)

                    
                    # B. Publish INVERTED (Default) Visualization
                    # Uses the inverse filtered array (object is black hole)
                    _publish_depth_visualization(
                        tracking_array, 
                        vis_pub_inv, 
                        raw_depth_data.header,
                        center_x, center_y, top_left, bottom_right, best_angle
                    )

                    # C. Publish POSITIVE Visualization
                    # Uses the direct filtered array (only object visible)
                    _publish_depth_visualization(
                        direct_filtered_array, 
                        vis_pub_pos, 
                        raw_depth_data.header,
                        center_x, center_y, top_left, bottom_right, best_angle
                    )

                # 6. Visualization on Color Image (Always published for monitoring)
                if self.publish_color_vis:
                    
                    # Determine if we should publish color vis based on the same score check
                    should_publish_color = self.VIS_ALWAYS_PUBLISH or (publish_location_update and best_score >= self.VIS_MIN_SCORE_THRESHOLD)

                    if should_publish_color:
                        color_cv = self.bridge.imgmsg_to_cv2(color_data, "bgr8")
                        tracked_color_cv = color_cv.copy() 
                        
                        # Draw the SAME markers on the color image
                        marker_size = 20
                        box_color = (255, 255, 0) if publish_location_update else (0, 255, 255) # Yellow if active, Cyan if failed
                        
                        # Redraw Rotated Box on color image
                        rect_center = (float(center_x), float(center_y))
                        rect_size = (float(self.kernel_w), float(self.kernel_h)) # Ensure size is float
                        rect_angle = -best_angle 

                        rect = (rect_center, rect_size, rect_angle)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.polylines(tracked_color_cv, [box], True, box_color, 2)
                        
                        # Draw center markers
                        cv2.line(tracked_color_cv, (center_x, center_y - marker_size), (center_x, center_y + marker_size), (0, 0, 255), 2)
                        cv2.line(tracked_color_cv, (center_x - marker_size, center_y), (center_x + marker_size, center_y), (0, 0, 255), 2)
                        cv2.circle(tracked_color_cv, (center_x, center_y), 5, (0, 255, 0), -1)

                        tracked_color_msg = self.bridge.cv2_to_imgmsg(tracked_color_cv, "bgr8")
                        tracked_color_msg.header = color_data.header
                        self.tracked_color_pub.publish(tracked_color_msg)
            
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
            except Exception as e:
                rospy.logerr(f"An unexpected error occurred during tracking: {e}")


if __name__ == '__main__':
    try:
        tracker = TemplateDepthTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
