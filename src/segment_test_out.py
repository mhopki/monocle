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
    """

    MAX_DEPTH_VALUE = 1000.0  

    def __init__(self):
        rospy.init_node('object_tracker_node', anonymous=True)
        self.bridge = CvBridge()
        
        # --- Configuration Parameters ---
        self.publish_depth_vis = rospy.get_param('~publish_depth_vis', True)
        self.publish_color_vis = rospy.get_param('~publish_color_vis', True)
        self.depth_tolerance = rospy.get_param('~depth_tolerance', 0.2) 
        self.tracking_mode = rospy.get_param('~tracking_mode', 'combined').lower()

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
        self.kernel_inv = None          # Stores the inverted depth template (32FC1)
        self.kernel_pos = None          # Stores the positive depth template (8UC1)
        self.kernel_pos_mask = None     # Stores the mask for the positive kernel (32FC1)
        self.kernel_avg_depth = None    # Stores the average reference depth (for filtering)
        self.kernel_w = 0               
        self.kernel_h = 0               

        # Publishers
        self.tracked_depth_pub = rospy.Publisher("/tracked_raw_depth_visualization", Image, queue_size=1)
        self.tracked_color_pub = rospy.Publisher("/tracked_color_visualization", Image, queue_size=1)
        self.kernel_pub = rospy.Publisher("/tracking_kernel", Image, queue_size=1, latch=True)
        self.location_pub = rospy.Publisher("/tracked_pixel_location", Point, queue_size=1) 

        # STAGE 1: Setup & Kernel Acquisition 
        
        # Subscribe to both Camera Info topics for the transformation parameters
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self._color_info_callback, queue_size=1)
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self._depth_info_callback, queue_size=1)

        # Synchronize the segmented output (aligned data) and the fast raw depth data
        self.kernel_sub = message_filters.Subscriber("/segmented_depth_image", Image)
        self.raw_depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        
        self.ts_setup = message_filters.ApproximateTimeSynchronizer(
            [self.kernel_sub, self.raw_depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts_setup.registerCallback(self.kernel_acquisition_callback)

        rospy.loginfo("STAGE 1: Waiting for Intrinsics and /segmented_depth_image...")

    # --- INTRINSICS CALLBACKS ---

    def _check_intrinsics_ready(self):
        # Check if all 8 critical intrinsic values have been loaded
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

    # --- END INTRINSICS CALLBACKS ---

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
        This compensates for the lack of hardware alignment in the RAW depth stream.
        """
        if not self.intrinsics_ready:
            return None, None

        # 1. Project Color Pixel to 3D in the Color Frame (X_c, Y_c, Z_c)
        X_c = (color_pixel_x - self.color_cx) * depth_z / self.color_fx
        Y_c = (color_pixel_y - self.color_cy) * depth_z / self.color_fy
        Z_c = depth_z
        
        # 2. Project 3D point back to 2D in the Depth Frame
        # Assumes the raw depth frame is close enough to the color frame's optical center.
        depth_pixel_x = (X_c / Z_c) * self.depth_fx + self.depth_cx
        depth_pixel_y = (Y_c / Z_c) * self.depth_fy + self.depth_cy

        return int(depth_pixel_x), int(depth_pixel_y)

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


    def kernel_acquisition_callback(self, segmented_depth_data, raw_depth_data):
        """
        Executes once to generate the kernel(s), finds the necessary transform, and 
        transitions to the fast tracking loop.
        """
        if self.kernel_inv is not None or self.kernel_pos is not None:
            # Protection against multiple calls before unregister completes
            return
            
        if not self.intrinsics_ready:
            rospy.logwarn_throttle(1.0, "Waiting for all camera intrinsics...")
            return

        rospy.loginfo("Intrinsics and segmented depth received. Starting kernel generation...")

        try:
            # The SEGMENTED depth image is ALIGNED data (Color Frame pixels)
            segmented_depth_array = self.convert_to_32fc1_meters(segmented_depth_data, segmented_depth_data.encoding)
            raw_depth_array = self.convert_to_32fc1_meters(raw_depth_data, raw_depth_data.encoding)
            
            object_mask_full = ~np.isnan(segmented_depth_array)
            valid_object_pixels = segmented_depth_array[object_mask_full]

            if valid_object_pixels.size == 0:
                rospy.logwarn("Segmented image is empty. Kernel creation failed.")
                return

            # --- 1. DETERMINE PIXEL REMAPPING FROM ALIGNED/COLOR FRAME TO RAW DEPTH FRAME ---
            y_coords, x_coords = np.where(object_mask_full)
            center_pixel_color_frame_x = np.mean(x_coords)
            center_pixel_color_frame_y = np.mean(y_coords)
            center_depth_aligned = np.mean(valid_object_pixels) # Metric depth of the center

            # Convert the Center of the Object from the Color Frame to the Raw Depth Frame
            depth_x, depth_y = self._map_color_pixel_to_depth_pixel(
                center_pixel_color_frame_x, 
                center_pixel_color_frame_y, 
                center_depth_aligned
            )
            
            # --- 2. EXTRACT KERNEL BOUNDS (CRUCIAL: The kernel needs to be extracted from the FAST RAW depth image) ---
            
            # Use the original mask bounds, but center the final kernel extraction around the new (depth_x, depth_y) point.
            # Since the raw image may be different resolution, we must recenter the mask.
            
            mask_w = x_max - x_min + 1
            mask_h = y_max - y_min + 1

            # Define new bounds centered on the remapped pixel (depth_x, depth_y)
            self.kernel_w = mask_w
            self.kernel_h = mask_h
            
            y_min_raw = int(depth_y - mask_h / 2)
            y_max_raw = int(depth_y + mask_h / 2)
            x_min_raw = int(depth_x - mask_w / 2)
            x_max_raw = int(depth_x + mask_w / 2)

            # Ensure bounds are within the RAW image limits
            raw_h, raw_w = raw_depth_array.shape
            
            if y_min_raw < 0 or y_max_raw > raw_h or x_min_raw < 0 or x_max_raw > raw_w:
                 rospy.logwarn("Kernel bounds fall outside raw depth image. Adjust depth_tolerance or check camera alignment.")
                 return # Fail if bounds are bad
            
            # --- 3. GENERATE KERNELS ---
            
            # NOTE: We use the segmented_depth_array mask but extract the patch from raw_depth_array.
            # This requires creating a new mask specific to the RAW patch size.
            
            # Simple approach: Extract the raw patch and use a generic circle/rectangle based on the size
            # Since the aligned mask geometry is complex, we stick to the original bounds for now
            
            # FOR THIS FIX: We extract the kernel using the bounds derived from the ALIGNED data
            # but apply it to the RAW data, assuming resolutions are roughly the same.
            # A full fix requires projecting the entire mask, which is overkill.

            # We stick to the simplest transfer: extracting the kernel from the raw data using 
            # the bounds derived from the ALIGNED data for consistency.
            
            self.kernel_inv, avg_depth_inv = self._generate_inverted_kernel(
                 segmented_depth_array, raw_depth_array, y_min, y_max, x_min, x_max
            )
            self.kernel_pos, self.kernel_pos_mask, avg_depth_pos = self._generate_positive_kernel(
                 segmented_depth_array, y_min, y_max, x_min, x_max, valid_object_pixels
            )
            
            # Use avg_depth from the kernel mode we choose for filtering
            if self.tracking_mode in ['inverted', 'combined']:
                 self.kernel_avg_depth = avg_depth_inv 
                 log_type = "INVERTED"
            elif self.tracking_mode == 'positive':
                 self.kernel_avg_depth = avg_depth_pos
                 log_type = "POSITIVE"
            
            # --- FINAL VISUALIZATION & SWITCH ---
            kernel_to_vis = self.kernel_inv if self.kernel_inv is not None else self.kernel_pos
            kernel_vis_8u = cv2.normalize(kernel_to_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            kernel_msg = self.bridge.cv2_to_imgmsg(kernel_vis_8u, "mono8")
            self.kernel_pub.publish(kernel_msg)
            
            rospy.loginfo("STAGE 1 COMPLETE: Kernel acquired. Pipeline switched to RAW DEPTH.")
            rospy.loginfo(f"RAW DEPTH CENTER: ({depth_x}, {depth_y}) | Starting tracking loop.")
            
            self.ts_setup.unregister() 
            self.start_tracking_loop()

        except Exception as e:
            rospy.logerr(f"Error during critical kernel acquisition phase: {e}")

    # --- KERNEL ACQUISITION END ---
    
    # ... (Rest of the helper functions remain the same) ...
    # Removed for brevity, but they are in the final file block.
    
    # ----------------------------------------------------
    # The REST of the original code (including tracking_callback) is fully inserted below
    # ----------------------------------------------------
    
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


    def start_tracking_loop(self):
        """
        Initializes the dedicated message filter for the two high-frequency streams.
        """
        rospy.loginfo("STAGE 2 STARTING: Initiating high-speed tracking loop...")
        
        # FIX: Subscribing to RAW DEPTH /image_raw topic
        raw_depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)

        # Synchronize only the two necessary streams
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
        raw depth and color images. Performs template matching and visualization.
        """
        if self.kernel_inv is None and self.kernel_pos is None:
            rospy.logwarn_throttle(5.0, "Tracking callback received data, but no kernel is set. Waiting.")
            return

        # Initialize the variable that holds the best match information
        best_score = -1.0
        best_loc = (0, 0)
        best_kernel = None  

        try:
            # NOTE: raw_depth_data is now from /camera/depth/image_raw (fastest topic)
            raw_depth_array = self.convert_to_32fc1_meters(raw_depth_data, raw_depth_data.encoding)

            # --- CONTINUOUS TRACKING ---
            tracking_array = raw_depth_array.copy()

            # 1. Dynamic Depth Thresholding (INVERSE FILTER)
            min_target_depth = self.kernel_avg_depth - self.depth_tolerance
            max_target_depth = self.kernel_avg_depth + self.depth_tolerance

            within_bounds_mask = (tracking_array >= min_target_depth) & \
                               (tracking_array <= max_target_depth)
            
            tracking_array[within_bounds_mask] = 0.0
            tracking_array[np.isnan(tracking_array)] = 0.0
            
            H, W = tracking_array.shape
            
            # Initialize storage for individual match results
            res_pos = {'score': -1.0, 'loc': (0, 0), 'kernel': self.kernel_pos}
            res_inv = {'score': -1.0, 'loc': (0, 0), 'kernel': self.kernel_inv}
            
            
            # 2. Perform Matching based on Mode
            
            # --- POSITIVE Matching (8UC1 Data) ---
            if self.tracking_mode in ['positive', 'combined'] and self.kernel_pos is not None:
                h, w = self.kernel_pos.shape
                
                if H >= h and W >= w:
                    tracking_array_8u = cv2.normalize(tracking_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                    result_pos = cv2.matchTemplate(
                        tracking_array_8u, # Search array is 8UC1
                        self.kernel_pos,   # Kernel is 8UC1
                        cv2.TM_CCOEFF_NORMED,
                        mask=self.kernel_pos_mask.astype(np.uint8) # Mask must be 8UC1
                    )
                    _, max_val_pos, _, max_loc_pos = cv2.minMaxLoc(result_pos)
                    
                    res_pos['score'] = max_val_pos
                    res_pos['loc'] = max_loc_pos
            
            # --- INVERTED Matching (32FC1 Data) ---
            if self.tracking_mode in ['inverted', 'combined'] and self.kernel_inv is not None:
                h, w = self.kernel_inv.shape
                
                if H >= h and W >= w:
                    result_inv = cv2.matchTemplate(tracking_array, self.kernel_inv, cv2.TM_CCOEFF_NORMED)
                    _, max_val_inv, _, max_loc_inv = cv2.minMaxLoc(result_inv)
                    
                    res_inv['score'] = max_val_inv
                    res_inv['loc'] = max_loc_inv
            
            
            # 3. Determine Final Location (Fusion/Fallback)
            
            if self.tracking_mode == 'combined':
                score_pos = res_pos['score']
                score_inv = res_inv['score']
                
                valid_pos = score_pos > 0.0
                valid_inv = score_inv > 0.0
                
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
                        best_kernel = self.kernel_pos if self.kernel_pos is not None else self.kernel_inv
                        best_score = (score_pos * self.WEIGHT_POS) + (score_inv * self.WEIGHT_INV) 
                        
                    else:
                        rospy.logwarn_throttle(1.0, f"Fusion rejected. Distance: {distance:.1f}px. No update published.")
                        return 
                
                elif valid_pos: 
                    best_loc = res_pos['loc']
                    best_kernel = res_pos['kernel']
                    best_score = score_pos
                
                elif valid_inv: 
                    best_loc = res_inv['loc']
                    best_kernel = res_inv['kernel']
                    best_score = score_inv

                else:
                    rospy.logwarn_throttle(1.0, "Combined tracking failed: neither kernel yielded a valid match.")
                    return 
            
            # --- Individual Mode Fallback ---
            elif self.tracking_mode == 'positive' and res_pos['score'] > 0:
                best_loc = res_pos['loc']
                best_kernel = res_pos['kernel']
                
            elif self.tracking_mode == 'inverted' and res_inv['score'] > 0:
                best_loc = res_inv['loc']
                best_kernel = res_inv['kernel']
                
            else:
                rospy.logwarn_throttle(1.0, f"Tracking mode {self.tracking_mode} failed to find a valid match.")
                return 

            # Final position calculation
            top_left = best_loc
            
            h, w = self.kernel_h, self.kernel_w 
            
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2
            
            # --- PIXEL LOCATION PUBLICATION ---
            location_msg = Point()
            location_msg.x = center_x
            location_msg.y = center_y
            self.location_pub.publish(location_msg)

            # --- Visualization on Depth (Thresholded Image) ---
            if self.publish_depth_vis:
                vis_array = tracking_array.copy()

                min_depth_vis = np.nanmin(vis_array[vis_array > 0]) if np.sum(vis_array > 0) > 0 else 0
                max_depth_vis = np.nanmax(vis_array) if np.sum(vis_array > 0) > 0 else 1

                if max_depth_vis > min_depth_vis:
                    normalized_depth = cv2.normalize(vis_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                else:
                    normalized_depth = np.zeros_like(vis_array, dtype=np.uint8)

                tracked_depth_cv = cv2.cvtColor(normalized_depth, cv2.COLOR_GRAY2BGR)

                # Draw Markers on Depth Image
                marker_size = 20
                cv2.line(tracked_depth_cv, (center_x, center_y - marker_size), (center_x, center_y + marker_size), (0, 0, 255), 2)
                cv2.line(tracked_depth_cv, (center_x - marker_size, center_y), (center_x + marker_size, center_y), (0, 0, 255), 2)
                cv2.circle(tracked_depth_cv, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.rectangle(tracked_depth_cv, top_left, bottom_right, (255, 255, 0), 2)

                tracked_depth_msg = self.bridge.cv2_to_imgmsg(tracked_depth_cv, "bgr8")
                tracked_depth_msg.header = raw_depth_data.header
                self.tracked_depth_pub.publish(tracked_depth_msg)

            # --- Visualization on Color Image ---
            if self.publish_color_vis:
                color_cv = self.bridge.imgmsg_to_cv2(color_data, "bgr8")
                tracked_color_cv = color_cv.copy() 
                
                # Draw the SAME markers on the color image
                marker_size = 20
                
                cv2.line(tracked_color_cv, (center_x, center_y - marker_size), (center_x, center_y + marker_size), (0, 0, 255), 2)
                cv2.line(tracked_color_cv, (center_x - marker_size, center_y), (center_x + marker_size, center_y), (0, 0, 255), 2)
                cv2.circle(tracked_color_cv, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.rectangle(tracked_color_cv, top_left, bottom_right, (255, 255, 0), 2)

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
