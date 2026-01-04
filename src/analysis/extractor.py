"""
Pose & Metric Extraction Module
Extracts body measurements for fashion body shape analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Any


class PoseExtractor:
    """
    Extracts body pose landmarks and calculates fashion-relevant metrics
    (shoulder width, hip width, waist width) from input images.
    """
    
    def __init__(self) -> None:
        """Initialize MediaPipe Pose with high accuracy settings."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # High accuracy model
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    
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
        landmarks: Any,
        waist_y: float,
        image_width: int,
        image_height: int
    ) -> float:
        """
        Estimate waist width at the calculated waist height.
        
        Args:
            landmarks: MediaPipe pose landmarks
            waist_y: Y-coordinate of waist position (normalized)
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Estimated waist width in pixels
        """
        # Get left and right hip landmarks (indices 23, 24)
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        
        # For MVP, estimate waist width as a proportion of hip width
        # Fashion heuristic: waist is typically 70-90% of hip width depending on body shape
        # We use 80% as a reasonable middle estimate
        hip_width_normalized = abs(right_hip.x - left_hip.x)
        
        # Apply a slight taper factor (waist is typically narrower)
        waist_width_normalized = hip_width_normalized * 0.85
        
        # Convert to pixels
        waist_width_px = waist_width_normalized * image_width
        
        return waist_width_px
    
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
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        image_height, image_width = image.shape[:2]
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks
        
        # Extract key landmark indices
        # Shoulders: 11 (Left), 12 (Right)
        # Hips: 23 (Left), 24 (Right)
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        
        # Convert normalized coordinates to pixel coordinates
        left_shoulder_px = (left_shoulder.x * image_width, left_shoulder.y * image_height)
        right_shoulder_px = (right_shoulder.x * image_width, right_shoulder.y * image_height)
        left_hip_px = (left_hip.x * image_width, left_hip.y * image_height)
        right_hip_px = (right_hip.x * image_width, right_hip.y * image_height)
        
        # Calculate widths
        shoulder_width = self._calculate_euclidean_distance(
            left_shoulder_px, 
            right_shoulder_px
        )
        hip_width = self._calculate_euclidean_distance(
            left_hip_px, 
            right_hip_px
        )
        
        # Calculate waist position
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        waist_y_normalized = self._estimate_waist_position(shoulder_center_y, hip_center_y)
        
        # Estimate waist width
        waist_width = self._estimate_waist_width(
            landmarks,
            waist_y_normalized,
            image_width,
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
        landmarks_list = [
            {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }
            for lm in landmarks.landmark
        ]
        
        return {
            'shoulder_px': float(shoulder_width),
            'hip_px': float(hip_width),
            'waist_px': float(waist_width),
            'landmarks': landmarks_list,
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
