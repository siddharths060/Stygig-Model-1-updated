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
    
    def _estimate_waist_width(
        self,
        segmentation_mask: np.ndarray,
        waist_y: float,
        image_height: int
    ) -> float:
        """
        Calculate true waist width using segmentation mask.
        
        Args:
            segmentation_mask: Body segmentation mask from MediaPipe
            waist_y: Y-coordinate of waist position (normalized)
            image_height: Image height in pixels
            
        Returns:
            True waist width in pixels from segmentation mask
        """
        # Convert normalized Y to pixel coordinate
        waist_y_px = int(waist_y * image_height)
        
        # Clamp to valid range
        waist_y_px = max(0, min(waist_y_px, image_height - 1))
        
        # Get the row of pixels at waist level
        waist_row = segmentation_mask[waist_y_px, :]
        
        # Count non-zero pixels (body pixels) in this row
        waist_width_px = np.count_nonzero(waist_row)
        
        return float(waist_width_px)
    
    def _calculate_hip_width_from_mask(
        self,
        segmentation_mask: np.ndarray,
        hip_y: float,
        image_height: int
    ) -> float:
        """
        Calculate true hip width using segmentation mask.
        
        Args:
            segmentation_mask: Body segmentation mask from MediaPipe
            hip_y: Y-coordinate of hip position (normalized)
            image_height: Image height in pixels
            
        Returns:
            True hip width in pixels from segmentation mask
        """
        # Convert normalized Y to pixel coordinate
        hip_y_px = int(hip_y * image_height)
        
        # Clamp to valid range
        hip_y_px = max(0, min(hip_y_px, image_height - 1))
        
        # Get the row of pixels at hip level
        hip_row = segmentation_mask[hip_y_px, :]
        
        # Count non-zero pixels (body pixels) in this row
        hip_width_px = np.count_nonzero(hip_row)
        
        return float(hip_width_px)
    
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
        
        # Calculate shoulder width (skeletal measurement is still accurate)
        shoulder_width = self._calculate_euclidean_distance(
            left_shoulder_px, 
            right_shoulder_px
        )
        
        # Calculate waist and hip positions
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        waist_y_normalized = self._estimate_waist_position(shoulder_center_y, hip_center_y)
        
        # Use segmentation mask for true body width measurements
        hip_width = self._calculate_hip_width_from_mask(
            segmentation_mask,
            hip_center_y,
            image_height
        )
        
        waist_width = self._estimate_waist_width(
            segmentation_mask,
            waist_y_normalized,
            image_height
        )
        
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
