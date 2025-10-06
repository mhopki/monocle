#!/usr/bin/env python3

# ROS 1 Python Node for Kernel-Based Template Matching Depth Tracking

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from geometry_msgs.msg import Point # Message for publishing x, y coordinates

class TemplateDepthTracker:
    """
    Decoupled tracker with dynamic kernel selection:
    1. Subscribes once to /segmented_depth_image to create the kernel(s).
    2. Switches to a dedicated synchronization loop for raw depth and color images only.
    3. Supports 'inverted' (32FC1) and 'positive' (8UC1 masked) tracking modes, and a 
       'combined' mode with proximity-based weighted fusion.
    
    The tracking callback uses an inverse depth filter on the search image.
    """

    # Define a maximum possible depth value for use in the positive kernel mask
    MAX_DEPTH_VALUE = 1000.0  

    def __init__(self):
        """Initializes the node, bridge, publisher, and sets up the initial kernel acquisition subscriber."""
        rospy.init_node('object_tracker_node', anonymous=True)
        self.bridge = CvBridge()
        
        # --- Configuration Parameters ---
        self.publish_depth_vis = rospy.get_param('~publish_depth_vis', True)
        self.publish_color_vis = rospy.get_param('~publish_color_vis', True)
        self.depth_tolerance = rospy.get_param('~depth_tolerance', 0.05) # Tolerance in meters
        self.tracking_mode = rospy.get_param('~tracking_mode', 'combined').lower()

        # --- Fusion Parameters (for 'combined' mode) ---
        self.WEIGHT_INV = rospy.get_param('~weight_inverted', 0.5) 
        self.WEIGHT_POS = rospy.get_param('~weight_positive', 0.5)
        self.MAX_FUSION_DISTANCE_PIXELS = rospy.get_param('~fusion_distance_pixels', 30)

        # Normalize weights
        total_weight = self.WEIGHT_INV + self.WEIGHT_POS
        if total_weight == 0:
            rospy.logerr("Total fusion weight is zero. Defaulting to 50/50 split.")
            self.WEIGHT_INV = 0.5
            self.WEIGHT_POS = 0.5
            total_weight = 1.0
        
        self.WEIGHT_INV /= total_weight
        self.WEIGHT_POS /= total_weight
        
        rospy.loginfo(f"Tracking Mode: {self.tracking_mode.upper()} | Tolerance: +/- {self.depth_tolerance:.2f}m")
        rospy.loginfo(f"Fusion Weights (INV/POS): {self.WEIGHT_INV:.2f} / {self.WEIGHT_POS:.2f} | Fusion Distance: {self.MAX_FUSION_DISTANCE_PIXELS}px")

        
        # State variables
        self.kernel_inv = None          # Stores the inverted depth template (32FC1)
        self.kernel_pos = None          # Stores the positive depth template (8UC1)
        self.kernel_pos_mask = None     # Stores the mask for the positive kernel (32FC1)
        self.kernel_avg_depth = None    # Stores the average reference depth (for filtering)
        self.kernel_w = 0               # Kernel width
        self.kernel_h = 0               # Kernel height
        
        # Publishers
        self.tracked_depth_pub = rospy.Publisher("/tracked_raw_depth_visualization", Image, queue_size=1)
        self.tracked_color_pub = rospy.Publisher("/tracked_color_visualization", Image, queue_size=1)
        self.kernel_pub = rospy.Publisher("/tracking_kernel", Image, queue_size=1, latch=True)
        self.location_pub = rospy.Publisher("/tracked_pixel_location", Point, queue_size=1) 

        # STAGE 1: Kernel Acquisition
        self.kernel_sub = rospy.Subscriber(
            "/segmented_depth_image", 
            Image, 
            self.kernel_acquisition_callback, 
            queue_size=1
        )
        rospy.loginfo("STAGE 1: Waiting for /segmented_depth_image to build the kernel...")


    def convert_to_32fc1_meters(self, img_msg, encoding):
        """Converts an image message (16UC1 or 32FC1) to a 32FC1 NumPy array (depth in meters)."""
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, encoding)
        depth_array = np.array(cv_image, dtype=np.float32)

        if encoding == "16UC1":
            depth_array /= 1000.0
        
        # In case of invalid/missing depth data, set to NaN
        depth_array[depth_array == 0] = np.nan
        
        return depth_array

    def _generate_positive_kernel(self, segmented_depth_array, y_min, y_max, x_min, x_max, valid_object_pixels):
        """
        Generates an 8-bit grayscale kernel for texture matching and its 32FC1 mask.
        """
        kernel_patch = segmented_depth_array[y_min:y_max+1, x_min:x_max+1].copy()
        
        # 1. Calculate the average depth of the object (foreground)
        avg_depth = np.nanmean(valid_object_pixels)

        # 2. Create the binary mask (32FC1 type required for cv2.matchTemplate mask argument)
        pos_mask = (~np.isnan(kernel_patch)).astype(np.float32)
        
        # 3. Fill NaN/invalid areas in the float kernel patch (MAX_DEPTH_VALUE is irrelevant now, but for visualization)
        kernel_patch[np.isnan(kernel_patch)] = self.MAX_DEPTH_VALUE 
        
        # 4. Convert the object's depth profile to 8-bit grayscale for robust texture matching
        # Normalize between min and max depth within the patch range
        valid_depths = kernel_patch[kernel_patch != self.MAX_DEPTH_VALUE]
        
        if valid_depths.size == 0:
            rospy.logwarn("Positive kernel has no valid depth pixels for 8-bit normalization.")
            return None, None, None

        # Normalize 32FC1 depths to 8UC1 (0-255) based on the object's depth range
        min_d = np.nanmin(valid_depths)
        max_d = np.nanmax(valid_depths)
        
        if max_d > min_d:
            # Map the float range (min_d to max_d) onto 0-255
            kernel_8u = cv2.normalize(kernel_patch, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask=pos_mask.astype(np.uint8))
            # Set the masked area (background/MAX_DEPTH_VALUE) to 0 in the 8-bit kernel
            kernel_8u[kernel_8u == 0] = 0
            
            # Repopulate the background with a high penalty value for visual output, although the mask handles the matching
            kernel_8u[pos_mask.astype(np.uint8) == 0] = 1 # Use 1 as minimum non-zero value for visibility
        else:
            kernel_8u = np.zeros_like(kernel_patch, dtype=np.uint8)

        return kernel_8u, pos_mask, avg_depth

    def _generate_inverted_kernel(self, segmented_depth_array, raw_depth_array, y_min, y_max, x_min, x_max):
        """Generates a 32FC1 kernel that masks the object and matches the background depth profile."""
        
        # 1. Extract the raw depth patch corresponding to the bounding box
        kernel_patch_raw = raw_depth_array[y_min:y_max+1, x_min:x_max+1].copy()

        # 2. Extract the mask of the object within the patch
        object_mask_patch = ~np.isnan(segmented_depth_array[y_min:y_max+1, x_min:x_max+1])
        
        # 3. Create the INVERTED kernel: Mask out the object pixels
        inverted_kernel = kernel_patch_raw.copy()
        inverted_kernel[object_mask_patch] = np.nan
        
        # 4. Calculate the average depth of the background for filtering
        background_pixels = inverted_kernel[~np.isnan(inverted_kernel)]
        if background_pixels.size == 0:
            rospy.logwarn("Inverted kernel has no valid background pixels.")
            return None, None
            
        avg_depth = np.nanmean(background_pixels)

        # 5. Replace NaN (background/invalid/object area) with 0 for template matching
        inverted_kernel[np.isnan(inverted_kernel)] = 0.0
        
        return inverted_kernel, avg_depth


    def kernel_acquisition_callback(self, segmented_depth_data):
        """
        Executed once upon receiving the segmented depth image.
        Generates the selected kernel(s) and transitions to STAGE 2 (Tracking).
        """
        if self.kernel_inv is not None or self.kernel_pos is not None:
            return

        try:
            segmented_depth_array = self.convert_to_32fc1_meters(segmented_depth_data, segmented_depth_data.encoding)
            object_mask_full = ~np.isnan(segmented_depth_array)
            valid_object_pixels = segmented_depth_array[object_mask_full]

            if valid_object_pixels.size == 0:
                rospy.logwarn("Segmented image is empty. Kernel creation failed.")
                return

            y_coords, x_coords = np.where(object_mask_full)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            
            if (x_max - x_min < 10) or (y_max - y_min < 10):
                rospy.logwarn("Segmented object is too small to form a reliable kernel.")
                return
                
            self.kernel_w = x_max - x_min + 1
            self.kernel_h = y_max - y_min + 1

            # --- FETCH RAW DEPTH FOR INVERSION/DEPTH REFERENCE ---
            raw_depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image, timeout=5.0)
            raw_depth_array = self.convert_to_32fc1_meters(raw_depth_msg, raw_depth_msg.encoding)

            # --- KERNEL GENERATION BASED ON MODE ---
            avg_depth_inv = None
            avg_depth_pos = None

            if self.tracking_mode in ['inverted', 'combined']:
                self.kernel_inv, avg_depth_inv = self._generate_inverted_kernel(
                    segmented_depth_array, raw_depth_array, y_min, y_max, x_min, x_max
                )
                if self.tracking_mode == 'inverted' and avg_depth_inv is not None:
                    self.kernel_avg_depth = avg_depth_inv # Use background avg depth for filtering
                    log_type = "INVERTED (32FC1)"
                
            if self.tracking_mode in ['positive', 'combined']:
                self.kernel_pos, self.kernel_pos_mask, avg_depth_pos = self._generate_positive_kernel(
                    segmented_depth_array, y_min, y_max, x_min, x_max, valid_object_pixels
                )
                if self.tracking_mode == 'positive' and avg_depth_pos is not None:
                    self.kernel_avg_depth = avg_depth_pos # Use object avg depth for filtering
                    log_type = "POSITIVE (8UC1)"

            if self.tracking_mode == 'combined':
                if avg_depth_pos is not None:
                    # For combined mode, use the average of the positive kernel (object depth) for filtering
                    self.kernel_avg_depth = avg_depth_pos 
                    log_type = "COMBINED"
                else:
                    rospy.logerr("Combined mode requires a valid Positive Kernel. Initialization failed.")
                    return

            if self.kernel_inv is None and self.kernel_pos is None:
                rospy.logerr("Kernel generation failed for all selected modes.")
                return
            
            # --- Visualize and Publish Kernel (Primary Kernel) ---
            # Prioritize the visualization that matches the primary tracking mode
            if self.tracking_mode == 'positive' and self.kernel_pos is not None:
                kernel_to_vis = self.kernel_pos # Already 8UC1
                vis_encoding = "mono8"
            elif self.kernel_inv is not None:
                # Inverted/Combined mode visualization (requires normalization of 32FC1)
                kernel_to_vis = self.kernel_inv
                kernel_vis_32f = cv2.normalize(kernel_to_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                kernel_to_vis = kernel_vis_32f
                vis_encoding = "mono8"
            else:
                 rospy.logwarn("No kernel available for visualization.")
                 return

            kernel_msg = self.bridge.cv2_to_imgmsg(kernel_to_vis, vis_encoding)
            kernel_msg.header = segmented_depth_data.header
            self.kernel_pub.publish(kernel_msg)
            rospy.loginfo(f"Kernel extracted! Mode: {log_type}, Ref Depth: {self.kernel_avg_depth:.3f}m")

            # --- STAGE 2: Transition to Tracking ---
            self.kernel_sub.unregister()
            rospy.loginfo("STAGE 1 COMPLETE: Kernel acquired. Shutting down acquisition subscriber.")
            
            self.start_tracking_loop()

        except rospy.ROSException:
            rospy.logerr("Timeout waiting for raw depth frame. Cannot build kernel.")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error during kernel acquisition: {e}")
        except Exception as e:
            rospy.logerr(f"Error during kernel acquisition: {e}")


    def start_tracking_loop(self):
        """
        Initializes the dedicated message filter for the two high-frequency streams.
        """
        rospy.loginfo("STAGE 2 STARTING: Initiating high-speed tracking loop...")
        
        raw_depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)

        # Synchronize only the two necessary streams
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [raw_depth_sub, color_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.tracking_callback)
        rospy.loginfo("Tracking loop is active. Only subscribing to raw depth and color topics.")


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
            raw_depth_array = self.convert_to_32fc1_meters(raw_depth_data, raw_depth_data.encoding)

            # --- CONTINUOUS TRACKING ---
            tracking_array = raw_depth_array.copy()

            # 1. Dynamic Depth Thresholding (INVERSE FILTER)
            min_target_depth = self.kernel_avg_depth - self.depth_tolerance
            max_target_depth = self.kernel_avg_depth + self.depth_tolerance

            # Create mask for pixels *WITHIN* the target depth slice
            within_bounds_mask = (tracking_array >= min_target_depth) & \
                               (tracking_array <= max_target_depth)
            
            # Mask out the target depth slice: set pixels *within* the slice to 0.0
            tracking_array[within_bounds_mask] = 0.0

            # Also set invalid (NaN) depth readings to 0 for template matching
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
                
                # Check for valid matches
                valid_pos = score_pos > 0.0
                valid_inv = score_inv > 0.0
                
                if valid_pos and valid_inv:
                    # Calculate center coordinates for proximity check
                    center_pos_x = res_pos['loc'][0] + self.kernel_w // 2
                    center_pos_y = res_pos['loc'][1] + self.kernel_h // 2
                    center_inv_x = res_inv['loc'][0] + self.kernel_w // 2
                    center_inv_y = res_inv['loc'][1] + self.kernel_h // 2
                    
                    # Calculate pixel distance
                    distance = math.sqrt(
                        (center_pos_x - center_inv_x)**2 + 
                        (center_pos_y - center_inv_y)**2
                    )

                    if distance <= self.MAX_FUSION_DISTANCE_PIXELS:
                        rospy.logdebug(f"Fusion accepted. Distance: {distance:.1f}px")
                        
                        # Weighted Average of the Top-Left Corners (loc[0]=x, loc[1]=y)
                        fused_x = int(self.WEIGHT_POS * res_pos['loc'][0] + self.WEIGHT_INV * res_inv['loc'][0])
                        fused_y = int(self.WEIGHT_POS * res_pos['loc'][1] + self.WEIGHT_INV * res_inv['loc'][1])
                        
                        best_loc = (fused_x, fused_y)
                        best_kernel = self.kernel_pos if self.kernel_pos is not None else self.kernel_inv
                        best_score = (score_pos * self.WEIGHT_POS) + (score_inv * self.WEIGHT_INV) # Fused score
                        
                    else:
                        # FALLBACK: Distance exceeded threshold. 
                        rospy.logwarn_throttle(1.0, f"Fusion rejected. Distance: {distance:.1f}px. No update published.")
                        return # <--- MODIFIED: Exit callback if fusion is rejected
                
                elif valid_pos: # Only positive valid
                    best_loc = res_pos['loc']
                    best_kernel = res_pos['kernel']
                    best_score = score_pos
                
                elif valid_inv: # Only inverted valid
                    best_loc = res_inv['loc']
                    best_kernel = res_inv['kernel']
                    best_score = score_inv

                else:
                    rospy.logwarn_throttle(1.0, "Combined tracking failed: neither kernel yielded a valid match.")
                    return # Exit callback if no valid match
            
            # --- Individual Mode Fallback (Original logic for 'positive' or 'inverted' mode) ---
            elif self.tracking_mode == 'positive' and res_pos['score'] > 0:
                best_loc = res_pos['loc']
                best_kernel = res_pos['kernel']
                
            elif self.tracking_mode == 'inverted' and res_inv['score'] > 0:
                best_loc = res_inv['loc']
                best_kernel = res_inv['kernel']
                
            else:
                rospy.logwarn_throttle(1.0, f"Tracking mode {self.tracking_mode} failed to find a valid match.")
                return # Exit callback if individual mode fails

            # Final position calculation
            top_left = best_loc
            
            # Use the dimensions of the kernel that yielded the best score
            h, w = self.kernel_h, self.kernel_w # Use cached dimensions for drawing
            
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

                # Normalize to 8-bit grayscale for display
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
