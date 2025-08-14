#!/usr/bin/env python3
"""Faxitron Final Detector - Working version that detects all annotations including text."""

import cv2
import numpy as np
from PIL import Image
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import re
import easyocr
import time

class FaxitronFinalDetector:
    """Final working version that detects all annotations including text."""
    
    def __init__(self):
        # Multiple HSV color ranges for yellow detection
        self.yellow_ranges = [
            (np.array([15, 100, 100]), np.array([35, 255, 255])),  # Standard yellow
            (np.array([20, 80, 80]), np.array([40, 255, 255])),    # Lighter yellow
            (np.array([10, 120, 120]), np.array([30, 255, 255])),  # Darker yellow
            (np.array([25, 70, 70]), np.array([45, 255, 255]))     # Very light yellow
        ]
        
        # Initialize EasyOCR reader for text extraction
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)  # Use CPU mode
            print("EasyOCR initialized successfully")
        except Exception as e:
            print(f"Warning: EasyOCR initialization failed: {e}")
            self.reader = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    def detect_yellow_regions(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect yellow regions using multiple HSV ranges."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Combine multiple yellow masks
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.yellow_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Minimal cleaning to preserve small details
        kernel = np.ones((2, 2), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask, hsv
    
    def calculate_overlap(self, rect1: Dict[str, Any], rect2: Dict[str, Any]) -> float:
        """Calculate overlap ratio between two rectangles."""
        # Get coordinates
        x1, y1, w1, h1 = rect1['x'], rect1['y'], rect1['width'], rect1['height']
        x2, y2, w2, h2 = rect2['x'], rect2['y'], rect2['width'], rect2['height']
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Return overlap ratio (intersection over smaller rectangle)
        smaller_area = min(area1, area2)
        return intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    def filter_redundant_regions(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out redundant main regions that contain smaller rectangles."""
        # Sort by area (largest first)
        sorted_annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        
        filtered_annotations = []
        redundant_ids = set()
        
        for i, ann1 in enumerate(sorted_annotations):
            if ann1['id'] in redundant_ids:
                continue
                
            is_redundant = False
            
            # Check if this main region contains smaller rectangles
            if ann1['type'] == 'main_region':
                for j, ann2 in enumerate(sorted_annotations):
                    if i != j and ann2['id'] not in redundant_ids:
                        # Check if ann2 is contained within ann1
                        if (ann2['x'] >= ann1['x'] and 
                            ann2['y'] >= ann1['y'] and 
                            ann2['x'] + ann2['width'] <= ann1['x'] + ann1['width'] and 
                            ann2['y'] + ann2['height'] <= ann1['y'] + ann1['height']):
                            
                            # If the smaller region is a large_rectangle, mark main region as redundant
                            if ann2['type'] == 'large_rectangle':
                                is_redundant = True
                                redundant_ids.add(ann1['id'])
                                break
            
            if not is_redundant:
                filtered_annotations.append(ann1)
        
        return filtered_annotations
    
    def remove_duplicate_regions(self, annotations: List[Dict[str, Any]], overlap_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Remove duplicate regions that have high overlap."""
        if len(annotations) <= 1:
            return annotations
        
        print("Removing duplicate regions with high overlap...")
        
        # Sort by area (largest first)
        sorted_annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        
        unique_annotations = []
        used_indices = set()
        
        for i, ann1 in enumerate(sorted_annotations):
            if i in used_indices:
                continue
            
            unique_annotations.append(ann1)
            used_indices.add(i)
            
            # Check for overlapping regions
            for j in range(i + 1, len(sorted_annotations)):
                if j in used_indices:
                    continue
                
                ann2 = sorted_annotations[j]
                
                # Calculate overlap
                overlap = self.calculate_overlap(ann1, ann2)
                
                if overlap > overlap_threshold:
                    print(f"  Removing duplicate: {ann2['type']} {ann2['id']} (overlap: {overlap:.2f})")
                    used_indices.add(j)
        
        print(f"Removed {len(annotations) - len(unique_annotations)} duplicate regions")
        return unique_annotations
    
    def extract_text_from_region(self, image: np.ndarray, annotation: Dict[str, Any]) -> str:
        """Extract actual text content from a detected text annotation region using EasyOCR."""
        if self.reader is None:
            # Fallback to estimation if EasyOCR is not available
            area = annotation['area']
            if area < 200:
                return "single_char"
            elif area < 800:
                return "short_text"
            elif area < 2000:
                return "medium_text"
            else:
                return "long_text"
        
        try:
            # Extract the region from the image
            x, y, w, h = annotation['x'], annotation['y'], annotation['width'], annotation['height']
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return "invalid_region"
            
            # Extract the region
            region = image[y:y+h, x:x+w]
            
            if region.size == 0:
                return "empty_region"
            
            # Convert BGR to RGB for EasyOCR
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Perform OCR on the region
            results = self.reader.readtext(region_rgb)
            
            if not results:
                return "no_text_detected"
            
            # Extract text from results
            extracted_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter by confidence threshold
                    extracted_texts.append(text.strip())
            
            if extracted_texts:
                # Join multiple text lines if found
                final_text = " ".join(extracted_texts)
                # Clean up the text
                final_text = re.sub(r'[^\w\s\-\.]', '', final_text)  # Remove special characters except hyphens and dots
                final_text = final_text.strip()
                return final_text if final_text else "text_detected"
            else:
                return "low_confidence"
                
        except Exception as e:
            print(f"Error extracting text from region {annotation['id']}: {e}")
            return f"error_{str(e)[:20]}"
    
    def extract_text_from_basic_rectangle(self, image: np.ndarray, annotation: Dict[str, Any]) -> str:
        """Extract text content from a detected basic rectangle (main region or large rectangle)."""
        if self.reader is None:
            return "ocr_not_available"
        
        try:
            # Extract the region from the image
            x, y, w, h = annotation['x'], annotation['y'], annotation['width'], annotation['height']
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return "invalid_region"
            
            # Extract the region
            region = image[y:y+h, x:x+w]
            
            if region.size == 0:
                return "empty_region"
            
            # Convert BGR to RGB for EasyOCR
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Perform OCR on the basic rectangle
            print(f"  Processing {annotation['type']} {annotation['id']} ({w}x{h}) at ({x},{y})...", end=" ")
            
            results = self.reader.readtext(region_rgb)
            
            if not results:
                print("No text detected")
                return "no_text_detected"
            
            # Extract text from results
            extracted_texts = []
            total_confidence = 0
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter by confidence threshold
                    extracted_texts.append(text.strip())
                    total_confidence += confidence
            
            if extracted_texts:
                # Join multiple text lines if found
                final_text = " ".join(extracted_texts)
                # Clean up the text - keep hyphens and dots for text like "E21-E24"
                final_text = re.sub(r'[^\w\s\-\.]', '', final_text)
                final_text = final_text.strip()
                
                avg_confidence = total_confidence / len(extracted_texts)
                print(f"Extracted: '{final_text}' (avg confidence: {avg_confidence:.2f})")
                
                # Validate that this looks like expected text format (E followed by numbers)
                if re.match(r'^E\d+(\-\d+)?$', final_text):
                    print(f"    ✓ Perfect match: {final_text}")
                elif re.match(r'^E.*\d+', final_text):
                    print(f"    ✓ Good match: {final_text}")
                elif 'E' in final_text and any(c.isdigit() for c in final_text):
                    print(f"    ⚠ Partial match: {final_text}")
                else:
                    print(f"    ? Unexpected format: {final_text}")
                
                return final_text if final_text else "text_detected"
            else:
                print("Low confidence results")
                return "low_confidence"
                
        except Exception as e:
            print(f"Error: {e}")
            return f"error_{str(e)[:20]}"
    
    def detect_all_annotations(self, yellow_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Detect all annotations using all contour detection methods."""
        # Use RETR_LIST to get all contours (including nested ones)
        contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        annotations = []
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Approximate contour to see shape complexity
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_corners = len(approx)
            
            # Classify the annotation
            if area > 500000 and approx_corners == 4:
                annotation_type = "main_region"
            elif area > 100000 and approx_corners == 4:
                annotation_type = "large_rectangle"
            elif area > 10000 and approx_corners == 4:
                annotation_type = "medium_rectangle"
            elif area > 1000 and approx_corners == 4:
                annotation_type = "small_rectangle"
            elif area < 5000 and area > 50:
                annotation_type = "text_annotation"
            elif area < 50:
                annotation_type = "tiny_region"
            else:
                annotation_type = "irregular_shape"
            
            # Calculate center
            center = (int(x + w/2), int(y + h/2))
            
            annotations.append({
                'id': i,
                'type': annotation_type,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': float(area),
                'aspect_ratio': float(aspect_ratio),
                'center': center,
                'contour': contour.tolist(),
                'approx_corners': approx_corners,
                'perimeter': float(cv2.arcLength(contour, True))
            })
        
        return annotations
    
    def analyze_annotations(self, image_path: str) -> Dict[str, Any]:
        """Main function to analyze all annotations."""
        # Load image
        image = self.load_image(image_path)
        
        # Detect yellow regions
        yellow_mask, hsv = self.detect_yellow_regions(image)
        
        # Detect all annotations
        all_annotations = self.detect_all_annotations(yellow_mask)
        
        # Filter out redundant main regions
        filtered_annotations = self.filter_redundant_regions(all_annotations)
        
        # Find basic rectangles (main regions and large rectangles) FIRST
        basic_rectangles = []
        for ann in filtered_annotations:
            if ann['type'] in ['main_region', 'large_rectangle']:
                basic_rectangles.append(ann)
        
        # Remove duplicate regions with high overlap BEFORE text extraction
        print(f"Found {len(basic_rectangles)} basic rectangles, removing duplicates...")
        unique_basic_rectangles = self.remove_duplicate_regions(basic_rectangles)
        print(f"After duplicate removal: {len(unique_basic_rectangles)} unique basic rectangles")
        
        # NOW extract text from unique basic rectangles only
        for ann in unique_basic_rectangles:
            print(f"Extracting text from {ann['type']} {ann['id']}:")
            extracted_text = self.extract_text_from_basic_rectangle(image, ann)
            ann['extracted_text'] = extracted_text
        
        # Find text annotations and extract text content (keeping existing functionality)
        text_annotations = []
        for ann in filtered_annotations:
            if ann['type'] == 'text_annotation':
                # Extract actual text content using OCR
                extracted_text = self.extract_text_from_region(image, ann)
                ann['extracted_text'] = extracted_text
                text_annotations.append(ann)
        
        # Create simplified output for basic rectangles
        simplified_basic_rectangles = []
        for rect in unique_basic_rectangles:
            simplified_rect = {
                'id': rect['id'],
                'type': rect['type'],
                'x': rect['x'],
                'y': rect['y'],
                'extracted_text': rect.get('extracted_text', 'unknown')
            }
            simplified_basic_rectangles.append(simplified_rect)
        
        # Compile results
        results = {
            'image_path': image_path,
            'image_dimensions': {
                'width': image.shape[1],
                'height': image.shape[0]
            },
            'annotations': {
                'all': filtered_annotations,
                'text': text_annotations,
                'basic_rectangles': unique_basic_rectangles
            },
            'simplified_basic_rectangles': simplified_basic_rectangles,
            'total_annotations': len(filtered_annotations),
            'text_annotations_count': len(text_annotations),
            'basic_rectangles_count': len(unique_basic_rectangles),
            'filtered_out': len(all_annotations) - len(filtered_annotations),
            'yellow_mask_stats': {
                'total_yellow_pixels': int(np.sum(yellow_mask > 0)),
                'yellow_percentage': float(np.sum(yellow_mask > 0) / (image.shape[0] * image.shape[1]) * 100)
            }
        }
        
        return results
    
    def create_qc_visualization(self, image_path: str, results: Dict[str, Any], 
                              output_path: Optional[str] = None) -> None:
        """Create comprehensive QC visualization showing all annotation types."""
        # Load image with PIL for matplotlib
        pil_image = Image.open(image_path)
        image_array = np.array(pil_image)
        
        # Create figure with multiple subplots for comprehensive QC
        fig = plt.figure(figsize=(24, 20))
        
        # Define grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Original image with all annotations (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_array)
        ax1.set_title("Original Image", fontsize=14, weight='bold')
        
        # 2. Yellow mask (top center left)
        ax2 = fig.add_subplot(gs[0, 1])
        yellow_mask, _ = self.detect_yellow_regions(self.load_image(image_path))
        ax2.imshow(yellow_mask, cmap='gray')
        ax2.set_title(f"Yellow Mask\n{results['yellow_mask_stats']['total_yellow_pixels']} yellow pixels", 
                     fontsize=12, weight='bold')
        
        # 3. All annotations color-coded by type (top center right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image_array)
        ax3.set_title(f"All Annotations: {results['total_annotations']} total", fontsize=12, weight='bold')
        
        # Color scheme for different annotation types
        colors = {
            'main_region': 'red',
            'large_rectangle': 'darkred', 
            'medium_rectangle': 'orange',
            'small_rectangle': 'yellow',
            'text_annotation': 'blue',
            'tiny_region': 'green',
            'irregular_shape': 'purple'
        }
        
        # Draw all annotations with color coding
        for ann in results['annotations']['all']:
            color = colors.get(ann['type'], 'black')
            rect_patch = Rectangle(
                (ann['x'], ann['y']), 
                ann['width'], 
                ann['height'],
                linewidth=1, 
                edgecolor=color, 
                facecolor='none',
                alpha=0.8
            )
            ax3.add_patch(rect_patch)
            
            # Add center point for larger annotations
            if ann['area'] > 1000:
                ax3.plot(ann['center'][0], ann['center'][1], 'ko', markersize=3, alpha=0.7)
        
        # 4. Text annotations only (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(image_array)
        ax4.set_title(f"Text Annotations: {results['text_annotations_count']} found", fontsize=12, weight='bold')
        
        for ann in results['annotations']['text']:
            rect_patch = Rectangle(
                (ann['x'], ann['y']), 
                ann['width'], 
                ann['height'],
                linewidth=2, 
                edgecolor='blue', 
                facecolor='none',
                alpha=0.8
            )
            ax4.add_patch(rect_patch)
                    # Add text label with extracted text content
        text_content = ann.get('extracted_text', 'unknown')
        # Truncate long text for display
        display_text = text_content[:15] + "..." if len(text_content) > 15 else text_content
        ax4.text(ann['center'][0], ann['center'][1] - 5, 
                f"{display_text}", 
                color='blue', fontsize=6, ha='center', weight='bold')
        
        # 5. Main regions and large rectangles (middle left)
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(image_array)
        ax5.set_title("Main Regions & Large Rectangles", fontsize=12, weight='bold')
        
        for ann in results['annotations']['all']:
            if ann['type'] in ['main_region', 'large_rectangle']:
                color = colors.get(ann['type'], 'red')
                rect_patch = Rectangle(
                    (ann['x'], ann['y']), 
                    ann['width'], 
                    ann['height'],
                    linewidth=3, 
                    edgecolor=color, 
                    facecolor='none',
                    alpha=0.8
                )
                ax5.add_patch(rect_patch)
                # Add ID label
                ax5.text(ann['center'][0], ann['center'][1], 
                        f"{ann['type'][0].upper()}{ann['id']}", 
                        color=color, fontsize=10, ha='center', weight='bold')
        
        # 6. Medium and small rectangles (middle center left)
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(image_array)
        ax6.set_title("Medium & Small Rectangles", fontsize=12, weight='bold')
        
        for ann in results['annotations']['all']:
            if ann['type'] in ['medium_rectangle', 'small_rectangle']:
                color = colors.get(ann['type'], 'orange')
                rect_patch = Rectangle(
                    (ann['x'], ann['y']), 
                    ann['width'], 
                    ann['height'],
                    linewidth=2, 
                    edgecolor=color, 
                    facecolor='none',
                    alpha=0.8
                )
                ax6.add_patch(rect_patch)
                ax6.text(ann['center'][0], ann['center'][1], 
                        f"{ann['type'][0].upper()}{ann['id']}", 
                        color=color, fontsize=8, ha='center', weight='bold')
        
        # 7. Tiny regions (middle center right)
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(image_array)
        ax7.set_title("Tiny Regions (Noise)", fontsize=12, weight='bold')
        
        for ann in results['annotations']['all']:
            if ann['type'] == 'tiny_region':
                rect_patch = Rectangle(
                    (ann['x'], ann['y']), 
                    ann['width'], 
                    ann['height'],
                    linewidth=1, 
                    edgecolor='green', 
                    facecolor='none',
                    alpha=0.6
                )
                ax7.add_patch(rect_patch)
        
        # 8. Irregular shapes (middle right)
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.imshow(image_array)
        ax8.set_title("Irregular Shapes", fontsize=12, weight='bold')
        
        for ann in results['annotations']['all']:
            if ann['type'] == 'irregular_shape':
                rect_patch = Rectangle(
                    (ann['x'], ann['y']), 
                    ann['width'], 
                    ann['height'],
                    linewidth=2, 
                    edgecolor='purple', 
                    facecolor='none',
                    alpha=0.8
                )
                ax8.add_patch(rect_patch)
                ax8.text(ann['center'][0], ann['center'][1], 
                        f"I{ann['id']}", 
                        color='purple', fontsize=8, ha='center', weight='bold')
        
        # 9. Size distribution heatmap (bottom left)
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.imshow(image_array)
        ax9.set_title("Size Distribution", fontsize=12, weight='bold')
        
        for ann in results['annotations']['all']:
            # Color by size
            if ann['area'] < 100:
                color = 'red'  # Very small
            elif ann['area'] < 500:
                color = 'orange'  # Small
            elif ann['area'] < 2000:
                color = 'yellow'  # Medium
            elif ann['area'] < 10000:
                color = 'lightgreen'  # Large
            else:
                color = 'darkgreen'  # Very large
            
            rect_patch = Rectangle(
                (ann['x'], ann['y']), 
                ann['width'], 
                ann['height'],
                linewidth=1, 
                edgecolor=color, 
                facecolor='none',
                alpha=0.7
            )
            ax9.add_patch(rect_patch)
        
        # 10. Annotation count by type (bottom center left)
        ax10 = fig.add_subplot(gs[2, 1])
        ax10.axis('off')
        
        # Count annotations by type
        type_counts = {}
        for ann in results['annotations']['all']:
            ann_type = ann['type']
            type_counts[ann_type] = type_counts.get(ann_type, 0) + 1
        
        # Create bar chart
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors_list = [colors.get(t, 'gray') for t in types]
        
        bars = ax10.bar(types, counts, color=colors_list, alpha=0.7)
        ax10.set_title("Annotation Count by Type", fontsize=12, weight='bold')
        ax10.set_ylabel("Count")
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{count}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax10.get_xticklabels(), rotation=45, ha='right')
        
        # 11. Area distribution histogram (bottom center right)
        ax11 = fig.add_subplot(gs[2, 2])
        areas = [ann['area'] for ann in results['annotations']['all']]
        ax11.hist(areas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax11.set_title("Area Distribution", fontsize=12, weight='bold')
        ax11.set_xlabel("Area (pixels)")
        ax11.set_ylabel("Frequency")
        ax11.set_yscale('log')  # Log scale for better visualization
        
        # 12. Summary statistics (bottom right)
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis('off')
        
        # Create summary text
        summary_text = f"""
FAXITRON ANNOTATION ANALYSIS
        
Image: {Path(results['image_path']).name}
Dimensions: {results['image_dimensions']['width']} × {results['image_dimensions']['height']}

DETECTION RESULTS:
• Total Annotations: {results['total_annotations']}
• Text Annotations: {results['text_annotations_count']}
• Filtered Out: {results['filtered_out']}
• Yellow Pixels: {results['yellow_mask_stats']['total_yellow_pixels']:,}
• Yellow Percentage: {results['yellow_mask_stats']['yellow_percentage']:.2f}%

ANNOTATION TYPES:
"""
        
        for ann_type, count in type_counts.items():
            summary_text += f"• {ann_type.replace('_', ' ').title()}: {count}\n"
        
        summary_text += f"""
QC NOTES:
• Text annotations are highlighted in BLUE with extracted text content
• Main regions are highlighted in RED
• Large rectangles are highlighted in DARK RED
• Medium/small rectangles are highlighted in ORANGE/YELLOW
• Tiny regions (noise) are highlighted in GREEN
• Irregular shapes are highlighted in PURPLE
• Redundant main regions have been filtered out
• OCR confidence threshold: 0.3
"""
        
        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"QC Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_simplified_qc_visualization(self, image_path: str, results: Dict[str, Any], 
                                        output_path: Optional[str] = None) -> None:
        """Create simplified QC visualization showing basic rectangles and text parsing."""
        # Load image with PIL for matplotlib
        pil_image = Image.open(image_path)
        image_array = np.array(pil_image)
        
        # Create simplified figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 1. Original image with basic rectangles highlighted
        ax1.imshow(image_array)
        ax1.set_title("Basic Rectangles Detection", fontsize=16, weight='bold')
        
        # Color scheme for different types
        colors = {
            'main_region': 'red',
            'large_rectangle': 'blue'
        }
        
        # Draw basic rectangles with color coding
        for rect in results['annotations']['basic_rectangles']:
            color = colors.get(rect['type'], 'green')
            rect_patch = Rectangle(
                (rect['x'], rect['y']), 
                rect['width'], 
                rect['height'],
                linewidth=3, 
                edgecolor=color, 
                facecolor='none',
                alpha=0.8
            )
            ax1.add_patch(rect_patch)
            
            # Add ID label
            ax1.text(rect['center'][0], rect['center'][1], 
                    f"{rect['type'][0].upper()}{rect['id']}", 
                    color=color, fontsize=10, ha='center', weight='bold')
        
        # 2. Text parsing results
        ax2.imshow(image_array)
        ax2.set_title("Text Parsing Results", fontsize=16, weight='bold')
        
        # Draw rectangles and show extracted text
        for rect in results['annotations']['basic_rectangles']:
            color = colors.get(rect['type'], 'green')
            rect_patch = Rectangle(
                (rect['x'], rect['y']), 
                rect['width'], 
                rect['height'],
                linewidth=2, 
                edgecolor=color, 
                facecolor='none',
                alpha=0.8
            )
            ax2.add_patch(rect_patch)
            
            # Add extracted text
            text_content = rect.get('extracted_text', 'unknown')
            # Truncate long text for display
            display_text = text_content[:15] + "..." if len(text_content) > 15 else text_content
            ax2.text(rect['center'][0], rect['center'][1], 
                    f"{display_text}", 
                    color=color, fontsize=8, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add summary text
        summary_text = f"""
BASIC RECTANGLES & TEXT PARSING
        
Image: {Path(results['image_path']).name}
Dimensions: {results['image_dimensions']['width']} × {results['image_dimensions']['height']}

DETECTION RESULTS:
• Total Basic Rectangles: {results['basic_rectangles_count']}
• Yellow Pixels: {results['yellow_mask_stats']['total_yellow_pixels']:,}
• Yellow Percentage: {results['yellow_mask_stats']['yellow_percentage']:.2f}%

TEXT EXTRACTION:
• Main Regions (M): {len([r for r in results['annotations']['basic_rectangles'] if r['type'] == 'main_region'])}
• Large Rectangles (L): {len([r for r in results['annotations']['basic_rectangles'] if r['type'] == 'large_rectangle'])}
"""
        
        # Add summary as text box
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Simplified QC Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save analysis results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    def save_simplified_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save simplified analysis results to JSON file (basic rectangles only)."""
        simplified_results = {
            'image_path': results['image_path'],
            'image_dimensions': results['image_dimensions'],
            'basic_rectangles': results['simplified_basic_rectangles'],
            'total_basic_rectangles': results['basic_rectangles_count'],
            'yellow_mask_stats': results['yellow_mask_stats']
        }
        
        with open(output_path, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        print(f"Simplified results saved to: {output_path}")
    
    def get_text_annotations_summary(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get a clean summary of all text annotations with coordinates and extracted text."""
        text_summary = []
        
        for ann in results['annotations']['text']:
            text_summary.append({
                'id': ann['id'],
                'x': ann['x'],
                'y': ann['y'],
                'center_x': ann['center'][0],
                'center_y': ann['center'][1],
                'extracted_text': ann.get('extracted_text', 'unknown'),
                'area': ann['area'],
                'confidence': 'high' if ann.get('extracted_text') not in ['no_text_detected', 'low_confidence', 'error'] else 'low'
            })
        
        return text_summary
    
    def get_basic_rectangles_summary(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get a clean summary of all basic rectangles with coordinates and extracted text."""
        basic_rectangles_summary = []
        
        for ann in results['annotations']['basic_rectangles']:
            basic_rectangles_summary.append({
                'id': ann['id'],
                'type': ann['type'],
                'x': ann['x'],
                'y': ann['y'],
                'center_x': ann['center'][0],
                'center_y': ann['center'][1],
                'extracted_text': ann.get('extracted_text', 'unknown'),
                'area': ann['area'],
                'confidence': 'high' if ann.get('extracted_text') not in ['no_text_detected', 'low_confidence', 'error', 'ocr_not_available'] else 'low'
            })
        
        return basic_rectangles_summary
    
    def export_to_csv(self, results: Dict[str, Any], output_path: str) -> None:
        """Export annotation results to CSV format."""
        import pandas as pd
        
        csv_data = []
        
        for ann in results['annotations']['all']:
            row = {
                'id': ann['id'],
                'type': ann['type'],
                'x': ann['x'],
                'y': ann['y'],
                'width': ann['width'],
                'height': ann['height'],
                'area': ann['area'],
                'aspect_ratio': ann['aspect_ratio'],
                'center_x': ann['center'][0],
                'center_y': ann['center'][1],
                'approx_corners': ann['approx_corners'],
                'perimeter': ann['perimeter']
            }
            
            # Add extracted text for text annotations
            if ann['type'] == 'text_annotation':
                row['extracted_text'] = ann.get('extracted_text', 'unknown')
            else:
                row['extracted_text'] = 'N/A'
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        print(f"CSV exported to: {output_path}")

def main():
    """Main function to demonstrate the final detector."""
    # Initialize detector
    detector = FaxitronFinalDetector()
    
    # Test image path
    image_path = "/drive3/tnoorden/breast_NAT/faxitron/case_01/01_SBS-19-00098_E_Image[a].JPEG"
    
    try:
        print(f"Final detection of all annotations: {image_path}")
        
        # Analyze annotations
        results = detector.analyze_annotations(image_path)
        
        # Print results
        print(f"\nAnalysis Results:")
        print(f"Image dimensions: {results['image_dimensions']['width']}x{results['image_dimensions']['height']}")
        print(f"Total annotations: {results['total_annotations']}")
        print(f"Text annotations: {results['text_annotations_count']}")
        print(f"Basic rectangles: {results['basic_rectangles_count']}")
        print(f"Filtered out: {results['filtered_out']}")
        print(f"Yellow pixels: {results['yellow_mask_stats']['total_yellow_pixels']} ({results['yellow_mask_stats']['yellow_percentage']:.2f}%)")
        
        # Print basic rectangles with extracted content
        if results['basic_rectangles_count'] > 0:
            print(f"\nBasic Rectangles Found:")
            for ann in results['annotations']['basic_rectangles']:
                text_content = ann.get('extracted_text', 'unknown')
                print(f"  {ann['type']} {ann['id']}: '{text_content}' at ({ann['center'][0]}, {ann['center'][1]})")
            
            # Get and display basic rectangles summary
            basic_rectangles_summary = detector.get_basic_rectangles_summary(results)
            print(f"\nBasic Rectangles Summary (Coordinates and Extracted Text):")
            for rect_summary in basic_rectangles_summary:
                print(f"  {rect_summary['type']} {rect_summary['id']}: '{rect_summary['extracted_text']}' at ({rect_summary['x']}, {rect_summary['y']}) - Center: ({rect_summary['center_x']}, {rect_summary['center_y']}) - Confidence: {rect_summary['confidence']}")
        
        # Print annotation types
        type_counts = {}
        for ann in results['annotations']['all']:
            ann_type = ann['type']
            type_counts[ann_type] = type_counts.get(ann_type, 0) + 1
        
        print(f"\nAnnotation Types:")
        for ann_type, count in type_counts.items():
            print(f"  {ann_type}: {count}")
        
        # Print text annotations with extracted content
        if results['text_annotations_count'] > 0:
            print(f"\nText Annotations Found:")
            for ann in results['annotations']['text'][:10]:  # Show first 10
                text_content = ann.get('extracted_text', 'unknown')
                print(f"  Text {ann['id']}: '{text_content}' at ({ann['center'][0]}, {ann['center'][1]})")
            
            # Get and display text summary
            text_summary = detector.get_text_annotations_summary(results)
            print(f"\nText Summary (Coordinates and Extracted Text):")
            for text_ann in text_summary:
                print(f"  ID {text_ann['id']}: '{text_ann['extracted_text']}' at ({text_ann['x']}, {text_ann['y']}) - Center: ({text_ann['center_x']}, {text_ann['center_y']}) - Confidence: {text_ann['confidence']}")
        
        # Save results
        output_dir = Path("faxitron_final_output")
        output_dir.mkdir(exist_ok=True)
        
        json_path = output_dir / "final_detection_results.json"
        detector.save_results(results, str(json_path))
        
        # Save simplified results (basic rectangles only)
        simplified_json_path = output_dir / "simplified_basic_rectangles.json"
        detector.save_simplified_results(results, str(simplified_json_path))
        
        # Create comprehensive QC visualization
        vis_path = output_dir / "qc_visualization.png"
        detector.create_qc_visualization(image_path, results, str(vis_path))
        
        # Create simplified QC visualization
        simplified_vis_path = output_dir / "simplified_qc_visualization.png"
        detector.create_simplified_qc_visualization(image_path, results, str(simplified_vis_path))
        
        print(f"\nProcessing complete! Check the 'faxitron_final_output' directory for results.")
        print(f"Full results saved to: {json_path}")
        print(f"Simplified results saved to: {simplified_json_path}")
        print(f"QC Visualization saved to: {vis_path}")
        print(f"Simplified QC Visualization saved to: {simplified_vis_path}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()

def process_single_image(detector: FaxitronFinalDetector, image_path: str, output_dir: Path) -> bool:
    """Process a single image and save results."""
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        print(f"📁 Input image: {image_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"❌ Error: Image file {image_path} does not exist")
            return False
        
        # Get image size
        image_size = Path(image_path).stat().st_size
        print(f"📏 Image size: {image_size:,} bytes ({image_size/1024/1024:.1f} MB)")
        
        print(f"\n🔍 Starting analysis...")
        start_time = time.time()
        
        # Analyze annotations
        results = detector.analyze_annotations(image_path)
        
        analysis_time = time.time() - start_time
        print(f"✅ Analysis completed in {analysis_time:.1f}s")
        print(f"📊 Found {results['basic_rectangles_count']} basic rectangles")
        
        # Create image-specific output directory
        image_name = Path(image_path).stem
        image_output_dir = output_dir / image_name
        print(f"📂 Creating output directory: {image_output_dir}")
        image_output_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving results...")
        save_start = time.time()
        
        # Save results
        json_path = image_output_dir / "detection_results.json"
        detector.save_results(results, str(json_path))
        print(f"  ✅ Full results: {json_path}")
        
        # Save simplified results
        simplified_json_path = image_output_dir / "simplified_basic_rectangles.json"
        detector.save_simplified_results(results, str(simplified_json_path))
        print(f"  ✅ Simplified results: {simplified_json_path}")
        
        save_time = time.time() - save_start
        print(f"💾 Files saved in {save_time:.1f}s")
        
        print(f"\n🎨 Creating visualizations...")
        vis_start = time.time()
        
        # Create visualizations
        vis_path = image_output_dir / "qc_visualization.png"
        detector.create_qc_visualization(image_path, results, str(vis_path))
        print(f"  ✅ QC visualization: {vis_path}")
        
        simplified_vis_path = image_output_dir / "simplified_qc_visualization.png"
        detector.create_simplified_qc_visualization(image_path, results, str(simplified_vis_path))
        print(f"  ✅ Simplified QC: {simplified_vis_path}")
        
        vis_time = time.time() - vis_start
        print(f"🎨 Visualizations created in {vis_time:.1f}s")
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\n📊 Processing Summary:")
        print(f"  ✅ Successfully processed {Path(image_path).name}")
        print(f"  📊 Basic rectangles found: {results['basic_rectangles_count']}")
        print(f"  📁 Results saved to: {image_output_dir}")
        print(f"  ⏱️  Total processing time: {total_time:.1f}s")
        print(f"  🚀 Processing speed: {1/total_time:.2f} images/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {Path(image_path).name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_folder(detector: FaxitronFinalDetector, folder_path: str, output_dir: Path, verbose: bool = False) -> None:
    """Process all images in a folder sequentially."""
    print(f"\n{'='*80}")
    print(f"STARTING FOLDER PROCESSING")
    print(f"{'='*80}")
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Error: Folder {folder_path} does not exist")
        return
    
    print(f"📁 Processing folder: {folder_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Find all image files
    print(f"\n🔍 Scanning for image files...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        jpeg_files = list(folder_path.glob(f"*{ext}"))
        jpeg_upper_files = list(folder_path.glob(f"*{ext.upper()}"))
        image_files.extend(jpeg_files)
        image_files.extend(jpeg_upper_files)
        if verbose:
            print(f"  Found {len(jpeg_files)} {ext} files")
            print(f"  Found {len(jpeg_upper_files)} {ext.upper()} files")
    
    # Remove duplicates and sort
    image_files = list(set(image_files))
    image_files.sort()
    
    if not image_files:
        print(f"❌ No image files found in {folder_path}")
        return
    
    print(f"\n✅ Found {len(image_files)} total images to process")
    if verbose:
        print(f"📋 Image list:")
        for i, img in enumerate(image_files[:10]):  # Show first 10
            print(f"  {i+1:3d}. {img.name}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more images")
    
    # Create output directory
    print(f"\n📂 Creating output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True)
    
    # Process each image
    successful = 0
    failed = 0
    start_time = time.time()
    
    print(f"\n🚀 Starting image processing...")
    print(f"{'='*80}")
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n📸 Processing image {i}/{len(image_files)}: {image_file.name}")
        print(f"⏱️  Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
        
        try:
            if process_single_image(detector, str(image_file), output_dir):
                successful += 1
                print(f"✅ Successfully processed {image_file.name}")
            else:
                failed += 1
                print(f"❌ Failed to process {image_file.name}")
        except Exception as e:
            failed += 1
            print(f"💥 Exception while processing {image_file.name}: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
        
        # Show time estimates
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / i
        remaining_images = len(image_files) - i
        estimated_remaining = remaining_images * avg_time_per_image
        
        print(f"⏱️  Elapsed: {elapsed_time:.1f}s, Avg per image: {avg_time_per_image:.1f}s")
        print(f"⏱️  Estimated remaining time: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} minutes)")
    
    # Print final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"🎉 PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Results Summary:")
    print(f"  ✅ Successfully processed: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Total images: {len(image_files)}")
    print(f"  ⏱️  Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  📂 Results saved to: {output_dir}")
    print(f"  🚀 Average speed: {len(image_files)/total_time:.2f} images/minute")
    
    if failed > 0:
        print(f"\n⚠️  {failed} images failed to process. Check the output above for details.")
    
    print(f"{'='*80}")

def cli_main():
    """CLI entry point for the Faxitron Final Detector."""
    parser = argparse.ArgumentParser(
        description="Faxitron Final Detector - Detect and extract text from yellow annotations in medical images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python faxitron_final_detector.py -i /path/to/image.jpg
  
  # Process all images in a folder
  python faxitron_final_detector.py -f /path/to/folder
  
  # Specify custom output directory
  python faxitron_final_detector.py -i /path/to/image.jpg -o /custom/output
  
  # Process folder with verbose output
  python faxitron_final_detector.py -f /path/to/folder -v
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--image',
        type=str,
        help='Path to single image file to process'
    )
    input_group.add_argument(
        '-f', '--folder',
        type=str,
        help='Path to folder containing images to process'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='faxitron_output',
        help='Output directory for results (default: faxitron_output)'
    )
    
    # Processing options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip creating QC visualizations (faster processing)'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    print("🚀 Initializing Faxitron Final Detector...")
    if args.verbose:
        print(f"📋 Arguments:")
        print(f"  Input: {'Image: ' + args.image if args.image else 'Folder: ' + args.folder}")
        print(f"  Output: {args.output}")
        print(f"  Verbose: {args.verbose}")
        print(f"  No visualization: {args.no_visualization}")
    
    detector = FaxitronFinalDetector()
    
    # Create output directory
    output_dir = Path(args.output)
    print(f"📂 Setting up output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True)
    
    # Process based on input type
    if args.image:
        # Single image processing
        print(f"📸 Processing single image: {args.image}")
        if not Path(args.image).exists():
            print(f"❌ Error: Image file {args.image} does not exist")
            return 1
        
        process_single_image(detector, args.image, output_dir)
        
    elif args.folder:
        # Folder processing
        print(f"📁 Processing folder: {args.folder}")
        process_folder(detector, args.folder, output_dir, args.verbose)
    
    print(f"\n🎉 Processing complete!")
    return 0

if __name__ == "__main__":
    # Check if running as CLI or demo
    import sys
    
    if len(sys.argv) > 1:
        # CLI mode
        exit(cli_main())
    else:
        # Demo mode (original main function)
        main()
