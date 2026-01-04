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
    
    def _add_text_overlay(
        self,
        image: np.ndarray,
        body_shape: str,
        undertone: Optional[str]
    ) -> np.ndarray:
        """
        Add text overlay with analysis results.
        
        Args:
            image: Input image
            body_shape: Body shape classification
            undertone: Skin tone undertone (or None if not available)
            
        Returns:
            Image with text overlay
        """
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Prepare text
        if undertone:
            text = f"Shape: {body_shape} | Undertone: {undertone}"
        else:
            text = f"Shape: {body_shape} | Undertone: N/A"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.FONT,
            self.FONT_SCALE,
            self.FONT_THICKNESS
        )
        
        # Draw semi-transparent background rectangle
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (self.TEXT_PADDING - 5, self.TEXT_PADDING - 5),
            (self.TEXT_PADDING + text_width + 5, self.TEXT_PADDING + text_height + baseline + 5),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Draw text
        cv2.putText(
            annotated,
            text,
            (self.TEXT_PADDING, self.TEXT_PADDING + text_height),
            self.FONT,
            self.FONT_SCALE,
            self.COLOR_WHITE,
            self.FONT_THICKNESS,
            cv2.LINE_AA
        )
        
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
        
        # Add text overlay
        annotated = self._add_text_overlay(annotated, body_shape, undertone)
        
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
