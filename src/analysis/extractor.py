"""
Pose & Metric Extraction Module
Extracts body measurements for fashion body shape analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, List, Optional, Tuple, Any
import urllib.request
import os


class PoseExtractor:
    """
    Extracts body pose landmarks and calculates fashion-relevant metrics
    (shoulder width, hip width, waist width) from input images.
    """
    
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    MODEL_PATH = "models/pose_landmarker_heavy.task"
    
    def __init__(self) -> None:
        """Initialize MediaPipe Pose with high accuracy settings."""
        # Download model if not exists
        self._ensure_model_downloaded()
        
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
    
    def _ensure_model_downloaded(self) -> None:
        """Download the pose landmarker model if it doesn't exist."""
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(os.path.dirname(self.MODEL_PATH) or "models", exist_ok=True)
            print(f"Downloading pose detection model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")
    
    def __del__(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
    
    def _calculate_euclidean_distance(
        self, 
        point1: Tuple[float, float], 
        point2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: (x, y) coordinates of first point
            point2: (x, y) coordinates of second point
            
        Returns:
            Euclidean distance in pixels
        """
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def _estimate_waist_position(
        self,
        shoulder_y: float,
        hip_y: float
    ) -> float:
        """
        Estimate natural waistline position using fashion heuristics.
        
        Args:
            shoulder_y: Y-coordinate of shoulder midpoint
            hip_y: Y-coordinate of hip midpoint
            
        Returns:
            Estimated Y-coordinate of natural waist
        """
        # Calculate torso length
        torso_length = hip_y - shoulder_y
        
        # Natural waist is approximately 15% above hip center from the shoulder-hip midpoint
        # This is a fashion-based heuristic
        waist_y = hip_y - (torso_length * 0.15)
        
        return waist_y
    
    def _get_torso_width_from_mask(
        self,
        segmentation_mask: np.ndarray,
        y_level: int,
        center_x: int
    ) -> float:
        """
        Calculate torso width using center-out scan to exclude arms.
        
        This method scans left and right from the torso center, stopping at
        the first gap (0 pixel) to avoid including arms hanging at the sides.
        
        Args:
            segmentation_mask: Body segmentation mask from MediaPipe
            y_level: Y-coordinate (pixel row) to measure
            center_x: X-coordinate of torso center
            
        Returns:
            Torso width in pixels (excludes arms)
        """
        image_height, image_width = segmentation_mask.shape
        
        # Clamp y_level to valid range
        y_level = max(0, min(y_level, image_height - 1))
        center_x = max(0, min(center_x, image_width - 1))
        
        # Extract the row at this Y level
        row = segmentation_mask[y_level, :]
        
        # Fallback: If center pixel is 0, find nearest body pixel
        if row[center_x] == 0:
            # Search nearby for a body pixel (within 50px radius)
            search_radius = 50
            for offset in range(1, search_radius):
                # Try left
                if center_x - offset >= 0 and row[center_x - offset] > 0:
                    center_x = center_x - offset
                    break
                # Try right
                if center_x + offset < image_width and row[center_x + offset] > 0:
                    center_x = center_x + offset
                    break
            
            # If still no body pixel found, return 0
            if row[center_x] == 0:
                return 0.0
        
        # Scan Left: Start at center and move backwards until hitting gap or edge
        left_edge = center_x
        for x in range(center_x, -1, -1):
            if row[x] == 0:
                left_edge = x + 1  # Last valid body pixel
                break
            left_edge = x
        
        # Scan Right: Start at center and move forwards until hitting gap or edge
        right_edge = center_x
        for x in range(center_x, image_width):
            if row[x] == 0:
                right_edge = x - 1  # Last valid body pixel
                break
            right_edge = x
        
        # Calculate width
        torso_width = right_edge - left_edge + 1  # +1 because edges are inclusive
        
        return float(torso_width)
    
    def extract_metrics(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract body measurements from an image.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Dictionary containing:
                - shoulder_px: Shoulder width in pixels
                - hip_px: Hip width in pixels
                - waist_px: Estimated waist width in pixels
                - landmarks: List of all pose landmarks
                - shoulder_coords: Coordinates of shoulder points
                - hip_coords: Coordinates of hip points
                - waist_coords: Estimated coordinates of waist points
            Returns None if pose detection fails
        """
        # Get original image dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Pad image to multiple of 32 to avoid MediaPipe/XNNPACK memory alignment issues
        # This prevents "Check failed: 1 == ChannelSize()" errors with segmentation masks
        pad_h = (32 - (orig_height % 32)) % 32
        pad_w = (32 - (orig_width % 32)) % 32
        
        # Add padding to bottom and right (keeps top-left origin at 0,0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(
                image,
                top=0,
                bottom=pad_h,
                left=0,
                right=pad_w,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get padded image dimensions (used for all coordinate calculations)
        image_height, image_width = image.shape[:2]
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process the image
        results = self.pose.detect(mp_image)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None
        
        # Check if segmentation masks are available
        if not results.segmentation_masks or len(results.segmentation_masks) == 0:
            return None
        
        # Get the first pose's landmarks and segmentation mask
        landmarks_list = results.pose_landmarks[0]
        
        # Convert segmentation mask to numpy array and threshold to binary
        # Make a copy immediately to avoid MediaPipe internal processing issues
        segmentation_mask_raw = results.segmentation_masks[0].numpy_view().copy()
        
        # Convert to single channel if needed and create binary mask (0 or 1)
        if len(segmentation_mask_raw.shape) > 2:
            segmentation_mask_raw = segmentation_mask_raw[:, :, 0]
        
        # Threshold to create binary mask (values > 0.5 are body)
        segmentation_mask = (segmentation_mask_raw > 0.5).astype(np.uint8)
        
        # Extract key landmark indices
        # Shoulders: 11 (Left), 12 (Right)
        # Hips: 23 (Left), 24 (Right)
        left_shoulder = landmarks_list[11]
        right_shoulder = landmarks_list[12]
        left_hip = landmarks_list[23]
        right_hip = landmarks_list[24]
        
        # Convert normalized coordinates to pixel coordinates
        left_shoulder_px = (left_shoulder.x * image_width, left_shoulder.y * image_height)
        right_shoulder_px = (right_shoulder.x * image_width, right_shoulder.y * image_height)
        left_hip_px = (left_hip.x * image_width, left_hip.y * image_height)
        right_hip_px = (right_hip.x * image_width, right_hip.y * image_height)
        
        # Calculate waist and hip positions
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        waist_y_normalized = self._estimate_waist_position(shoulder_center_y, hip_center_y)
        
        # Calculate center X coordinates for torso measurements
        hip_center_x = (left_hip.x + right_hip.x) / 2
        waist_center_x = hip_center_x  # Use same center as hips for consistency
        
        # Convert to pixel coordinates
        hip_center_x_px = int(hip_center_x * image_width)
        waist_center_x_px = int(waist_center_x * image_width)
        hip_y_px = int(hip_center_y * image_height)
        waist_y_px = int(waist_y_normalized * image_height)
        
        # Step 1: Calculate Skeletal Widths (bone-to-bone distances)
        shoulder_skeletal = self._calculate_euclidean_distance(
            left_shoulder_px, 
            right_shoulder_px
        )
        hip_skeletal = self._calculate_euclidean_distance(left_hip_px, right_hip_px)
        
        # Step 2: Define Visual Width Multipliers
        # These account for flesh, deltoids, and clothing extending past joints
        SHOULDER_MULTIPLIER = 1.18  # Deltoids extend past shoulder joint
        HIP_MULTIPLIER = 1.15       # Flesh/clothing extends past hip joint
        
        # Step 3: Get Mask Widths using center-out scan (excludes arms)
        hip_mask_width = self._get_torso_width_from_mask(
            segmentation_mask,
            hip_y_px,
            hip_center_x_px
        )
        
        waist_mask_width = self._get_torso_width_from_mask(
            segmentation_mask,
            waist_y_px,
            waist_center_x_px
        )
        
        # Step 4: Robust Decision Tree for Final Measurements
        
        # Check 1: Oval/Apple Shape Detection (waist >= hips indicates belly fat)
        is_oval_shape = waist_mask_width >= hip_mask_width * 0.95
        
        if is_oval_shape:
            # Trust mask for torso, inflate skeletal for shoulders
            shoulder_width = shoulder_skeletal * SHOULDER_MULTIPLIER
            hip_width = hip_mask_width
            waist_width = waist_mask_width
        else:
            # Check 2: Arm Interference Detection
            # If mask hip > 1.35x skeletal, arms are merged with torso
            arm_interference = hip_mask_width > hip_skeletal * 1.35
            
            if arm_interference:
                # Fallback to inflated skeletal for all measurements
                shoulder_width = shoulder_skeletal * SHOULDER_MULTIPLIER
                hip_width = hip_skeletal * HIP_MULTIPLIER
                waist_width = hip_width * 0.92  # 92% avoids forcing hourglass on rectangles
            else:
                # Clean segmentation: inflate skeletal shoulders, trust mask for torso
                shoulder_width = shoulder_skeletal * SHOULDER_MULTIPLIER
                hip_width = hip_mask_width
                waist_width = waist_mask_width
        
        # Safety check: If measurements are suspiciously small, fall back to skeletal
        min_reasonable_width = image_width * 0.20  # 20% of image width
        
        if hip_width < min_reasonable_width:
            hip_width = hip_skeletal * HIP_MULTIPLIER
        
        if waist_width < min_reasonable_width:
            waist_width = hip_skeletal * 0.92
        
        # Calculate waist point coordinates for visualization
        waist_center_x = (left_hip.x + right_hip.x) / 2
        waist_y_px = waist_y_normalized * image_height
        waist_x_px = waist_center_x * image_width
        
        # Estimate left and right waist points
        waist_half_width = waist_width / 2
        left_waist_px = (waist_x_px - waist_half_width, waist_y_px)
        right_waist_px = (waist_x_px + waist_half_width, waist_y_px)
        
        # Convert landmarks to list format
        landmarks_output = [
            {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
            }
            for lm in landmarks_list
        ]
        
        return {
            'shoulder_px': float(shoulder_width),
            'hip_px': float(hip_width),
            'waist_px': float(waist_width),
            'landmarks': landmarks_output,
            'shoulder_coords': {
                'left': left_shoulder_px,
                'right': right_shoulder_px
            },
            'hip_coords': {
                'left': left_hip_px,
                'right': right_hip_px
            },
            'waist_coords': {
                'left': left_waist_px,
                'right': right_waist_px
            }
        }
