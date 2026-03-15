"""
Visualization Script for Feature Extraction Debugging
Displays body measurements and skin tone analysis results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.extractor import PoseExtractor
from src.analysis.skin_tone import SkinToneAnalyzer
from src.analysis.classifier import BodyShapeClassifier


class FeatureVisualizer:
    """
    Visualizes body shape and skin tone analysis results on images.
    """
    
    # Color definitions (BGR format for OpenCV)
    COLOR_NEON_GREEN = (57, 255, 20)  # Neon green for shoulder/hip lines
    COLOR_RED = (0, 0, 255)           # Red for waist line
    COLOR_WHITE = (255, 255, 255)     # White for text
    
    # Line and text properties
    LINE_THICKNESS = 3
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    TEXT_PADDING = 20
    
    def __init__(self) -> None:
        """Initialize the visualizer with analysis modules."""
        self.pose_extractor = PoseExtractor()
        self.skin_analyzer = SkinToneAnalyzer()
        self.body_classifier = BodyShapeClassifier()
    
    def _draw_measurement_lines(
        self,
        image: np.ndarray,
        metrics: dict
    ) -> np.ndarray:
        """
        Draw measurement lines on the image.
        
        Args:
            image: Input image
            metrics: Extracted metrics with coordinates
            
        Returns:
            Image with drawn lines
        """
        # Create a copy to avoid modifying the original
        annotated = image.copy()
        
        # Draw shoulder line (Neon Green)
        shoulder_left = tuple(map(int, metrics['shoulder_coords']['left']))
        shoulder_right = tuple(map(int, metrics['shoulder_coords']['right']))
        cv2.line(
            annotated,
            shoulder_left,
            shoulder_right,
            self.COLOR_NEON_GREEN,
            self.LINE_THICKNESS
        )
        
        # Draw hip line (Neon Green)
        hip_left = tuple(map(int, metrics['hip_coords']['left']))
        hip_right = tuple(map(int, metrics['hip_coords']['right']))
        cv2.line(
            annotated,
            hip_left,
            hip_right,
            self.COLOR_NEON_GREEN,
            self.LINE_THICKNESS
        )
        
        # Draw waist line (Red)
        waist_left = tuple(map(int, metrics['waist_coords']['left']))
        waist_right = tuple(map(int, metrics['waist_coords']['right']))
        cv2.line(
            annotated,
            waist_left,
            waist_right,
            self.COLOR_RED,
            self.LINE_THICKNESS
        )
        
        # Add small circles at measurement points for clarity
        for point in [shoulder_left, shoulder_right, hip_left, hip_right]:
            cv2.circle(annotated, point, 5, self.COLOR_NEON_GREEN, -1)
        
        for point in [waist_left, waist_right]:
            cv2.circle(annotated, point, 5, self.COLOR_RED, -1)
        
        return annotated
    
    def _draw_face_analysis(
        self,
        image: np.ndarray,
        skin_data: dict
    ) -> np.ndarray:
        """
        Draw face analysis details including bounding box, color swatch, and undertone.
        
        Args:
            image: Input image
            skin_data: Dictionary containing skin tone analysis results
            
        Returns:
            Image with face analysis visualization
        """
        annotated = image.copy()
        
        # Extract face bounding box and color data
        face_bbox = skin_data.get('face_bbox')
        if face_bbox is None:
            return annotated
        
        x, y, w, h = face_bbox
        undertone = skin_data['undertone']
        bgr_values = skin_data['bgr_values']
        
        # Draw face bounding box (cyan color for visibility)
        box_color = (255, 255, 0)  # Cyan in BGR
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        
        # Draw color swatch (30x30 filled square) next to the face box
        swatch_size = 30
        swatch_x = x + w + 10  # 10px to the right of face box
        swatch_y = y
        
        # Extract BGR color as integers
        swatch_color = (
            int(bgr_values['B']),
            int(bgr_values['G']),
            int(bgr_values['R'])
        )
        
        # Draw filled rectangle for color swatch
        cv2.rectangle(
            annotated,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            swatch_color,
            -1  # Filled
        )
        
        # Draw border around swatch for visibility
        cv2.rectangle(
            annotated,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            self.COLOR_WHITE,
            2
        )
        
        # Add undertone text above the face box
        undertone_text = f"Undertone: {undertone}"
        text_y = max(y - 10, 20)  # Position above box, but not off-screen
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            undertone_text,
            self.FONT,
            self.FONT_SCALE * 0.8,  # Slightly smaller font
            self.FONT_THICKNESS
        )
        
        # Draw semi-transparent background for text
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (x - 2, text_y - text_height - 2),
            (x + text_width + 2, text_y + baseline + 2),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Draw text
        cv2.putText(
            annotated,
            undertone_text,
            (x, text_y),
            self.FONT,
            self.FONT_SCALE * 0.8,
            self.COLOR_WHITE,
            self.FONT_THICKNESS,
            cv2.LINE_AA
        )
        
        return annotated
    
    def _add_text_overlay(
        self,
        image: np.ndarray,
        body_shape: str,
        undertone: Optional[str],
        metrics: dict
    ) -> np.ndarray:
        """
        Add text overlay with detailed analysis results.
        
        Args:
            image: Input image
            body_shape: Body shape classification
            undertone: Skin tone undertone (or None if not available)
            metrics: Dictionary containing body measurements
            
        Returns:
            Image with text overlay
        """
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Calculate ratios
        shoulder_width = metrics['shoulder_px']
        hip_width = metrics['hip_px']
        waist_width = metrics['waist_px']
        
        sh_ratio = shoulder_width / hip_width if hip_width > 0 else 0.0
        wh_ratio = waist_width / hip_width if hip_width > 0 else 0.0
        
        # Prepare multi-line text
        undertone_text = undertone if undertone else "N/A"
        lines = [
            f"Shape: {body_shape}",
            f"Undertone: {undertone_text}",
            f"S/H Ratio: {sh_ratio:.2f}",
            f"W/H Ratio: {wh_ratio:.2f}"
        ]
        
        # Calculate maximum text width and total height
        max_text_width = 0
        line_heights = []
        
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line,
                self.FONT,
                self.FONT_SCALE,
                self.FONT_THICKNESS
            )
            max_text_width = max(max_text_width, text_width)
            line_heights.append(text_height + baseline)
        
        # Calculate total height with spacing between lines
        line_spacing = 10
        total_height = sum(line_heights) + (len(lines) - 1) * line_spacing
        
        # Draw semi-transparent background rectangle
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (self.TEXT_PADDING - 5, self.TEXT_PADDING - 5),
            (self.TEXT_PADDING + max_text_width + 10, self.TEXT_PADDING + total_height + 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Draw each line of text
        y_offset = self.TEXT_PADDING
        for i, line in enumerate(lines):
            cv2.putText(
                annotated,
                line,
                (self.TEXT_PADDING, y_offset + line_heights[i]),
                self.FONT,
                self.FONT_SCALE,
                self.COLOR_WHITE,
                self.FONT_THICKNESS,
                cv2.LINE_AA
            )
            y_offset += line_heights[i] + line_spacing
        
        return annotated
    
    def visualize(self, image_path: str) -> Optional[np.ndarray]:
        """
        Complete visualization pipeline.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Annotated image or None if processing fails
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        print("Processing image...")
        
        # Extract pose metrics
        print("  - Extracting body measurements...")
        metrics = self.pose_extractor.extract_metrics(image)
        
        if metrics is None:
            print("Error: Could not detect pose in image")
            return None
        
        print(f"    Shoulder width: {metrics['shoulder_px']:.1f}px")
        print(f"    Hip width: {metrics['hip_px']:.1f}px")
        print(f"    Waist width: {metrics['waist_px']:.1f}px")
        
        # Classify body shape
        print("  - Classifying body shape...")
        body_shape = self.body_classifier.classify(metrics)
        print(f"    Body shape: {body_shape}")
        
        # Analyze skin tone
        print("  - Analyzing skin tone...")
        skin_tone = self.skin_analyzer.get_skin_tone(image)
        
        if skin_tone:
            undertone = skin_tone['undertone']
            print(f"    Undertone: {undertone}")
            print(f"    LAB values: L={skin_tone['lab_values']['L']:.1f}, "
                  f"A={skin_tone['lab_values']['A']:.1f}, "
                  f"B={skin_tone['lab_values']['B']:.1f}")
        else:
            undertone = None
            print("    Warning: Could not detect face for skin tone analysis")
        
        # Draw measurement lines
        print("  - Creating visualization...")
        annotated = self._draw_measurement_lines(image, metrics)
        
        # Draw face analysis details if skin tone was detected
        if skin_tone:
            annotated = self._draw_face_analysis(annotated, skin_tone)
        
        # Add text overlay with detailed metrics
        annotated = self._add_text_overlay(annotated, body_shape, undertone, metrics)
        
        print("Visualization complete!")
        return annotated
    
    def display(self, annotated_image: np.ndarray, title: str = "StyGig Feature Analysis") -> None:
        """
        Display the annotated image using matplotlib.
        
        Args:
            annotated_image: Annotated image to display
            title: Window title
        """
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run the visualization pipeline.
    """
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python visualize_features.py <image_path>")
        print("\nExample:")
        print("  python visualize_features.py test_image.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    print("=" * 60)
    print("StyGig Fashion Feature Extraction")
    print("=" * 60)
    print(f"Input image: {image_path}")
    print()
    
    # Create visualizer
    visualizer = FeatureVisualizer()
    
    # Process and visualize
    annotated = visualizer.visualize(image_path)
    
    if annotated is not None:
        print()
        print("Displaying results...")
        visualizer.display(annotated)
    else:
        print("Failed to process image.")


if __name__ == "__main__":
    main()
