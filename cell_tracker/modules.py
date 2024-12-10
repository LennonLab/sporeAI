"""
Cell Tracker Modules
------------------
Core functionality modules for cell tracking and segmentation.

This module provides the following key components:
- ImageLoader: Handles image input/output operations
- Preprocessor: Image preprocessing and enhancement
- Segmenter: Cell segmentation using cellSAM
- QuadrantSegmenter: Large image processing through quadrant-based segmentation
"""

import numpy as np
import cv2
import torch
from cellSAM import segment_cellular_image, get_model
from skimage import measure, morphology, io
from scipy.ndimage import find_objects
from pathlib import Path
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImageLoader:
    """
    Handles image loading and validation operations.
    
    Provides utilities for:
    - Loading various image formats
    - Input validation
    - Error handling for corrupt or missing files
    """

    @staticmethod
    def load_image(image_path):
        """
        Load and validate an image from the specified path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Loaded image array
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: For other loading errors
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            img = io.imread(str(path))
            print(f"Successfully loaded image: {path.name}")
            return img
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

class Preprocessor:
    """
    Handles image preprocessing operations.
    
    Features:
    - Artifact removal through inpainting
    - Noise reduction via Gaussian blur
    - Contrast adjustment using gamma correction
    """

    def __init__(self, *,
                 inpaint=True,
                 gaussian_blur_enabled=False,
                 gaussian_blur_sigma=1.0
                 ):
        """
        Initialize preprocessor with specific parameters.
        
        Args:
            inpaint (bool): Whether to remove artifacts using inpainting
            gaussian_blur_enabled (bool): Whether to apply Gaussian blur
            gaussian_blur_sigma (float): Sigma value for Gaussian blur
        """
        self.inpaint = inpaint
        self.gaussian_blur_enabled = gaussian_blur_enabled
        self.gaussian_blur_sigma = gaussian_blur_sigma

    def process(self, img):
        """
        Apply configured preprocessing steps to the image.
        
        Args:
            img (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if self.inpaint:
            img = self.inpaint_image(img)

        if self.gaussian_blur_enabled:
            img = cv2.GaussianBlur(img, (0, 0), self.gaussian_blur_sigma)

        return img

    def inpaint_image(self, img, inpaint_radius=5):
        """
        Remove artifacts using inpainting.
        
        Args:
            img (np.ndarray): Input image
            inpaint_radius (int): Radius of pixels to consider for inpainting
            
        Returns:
            np.ndarray: Image with artifacts removed
        """
        # Create mask where RGB channels differ (indicating artifacts)
        mask = (img[:, :, 0] != img[:, :, 1]) | (img[:, :, 0] != img[:, :, 2])
        mask = mask.astype(np.uint8) * 255
        return cv2.inpaint(img, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

class Segmenter:
    """
    Handles single-image segmentation using cellSAM.
    
    Features:
    - Cell detection and segmentation
    - Automatic device selection (CPU/GPU)
    - Configurable confidence thresholds
    - Visualization capabilities
    """

    def __init__(self, *, bbox_threshold=0.14):
        """
        Initialize segmenter with specific parameters.
        
        Args:
            bbox_threshold (float): Confidence threshold for bounding boxes
        """
        self.bbox_threshold = bbox_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        """
        Load the cellSAM model if not already loaded.
        
        Returns:
            model: Loaded cellSAM model
        """
        if self.model is None:
            self.model = get_model()
        return self.model

    def segment(self, img):
        """
        Segment a single image using cellSAM.
        
        Args:
            img (np.ndarray): Input image
            
        Returns:
            tuple: (mask, bbox)
                - mask (np.ndarray): Segmentation mask
                - bbox (np.ndarray): Bounding boxes for detected cells
        """
        self.load_model()
        mask, _, bbox = segment_cellular_image(
            img,
            device=str(self.device),
            model=self.model,
            normalize=True,
            fast=True,
            bbox_threshold=self.bbox_threshold
        )
        boxes = bbox[0].cpu().numpy()
        return mask, boxes

    def get_cell_properties(self, mask):
        """
        Extract cell properties from segmentation mask.
        
        Args:
            mask (np.ndarray): Labeled segmentation mask
            
        Returns:
            tuple: (centers, boxes)
                - centers: List of cell centroids
                - boxes: List of bounding boxes
        """
        props = measure.regionprops(mask)
        centers = [prop.centroid for prop in props]
        bboxes = [prop.bbox for prop in props]
        boxes = [[float(min_x), float(min_y), float(max_x), float(max_y)] 
                for (min_y, min_x, max_y, max_x) in bboxes]
        return centers, boxes

    def visualize(self, img, mask, boxes):
        """
        Visualize segmentation results.
        
        Args:
            img (np.ndarray): Original image
            mask (np.ndarray): Segmentation mask
            boxes (np.ndarray): Bounding boxes
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        ax1.imshow(img)
        ax1.set_title("Original Image")
        ax1.axis('off')

        # Mask overlay
        ax2.imshow(img)
        colored_mask = np.ma.masked_where(mask == 0, mask)
        ax2.imshow(colored_mask, cmap='hsv', alpha=0.5)
        ax2.set_title("Segmentation Mask")
        ax2.axis('off')

        # Bounding boxes
        ax3.imshow(img)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            ax3.add_patch(rect)
        ax3.set_title("Bounding Boxes")
        ax3.axis('off')

        plt.tight_layout()
        plt.show()

class QuadrantSegmenter:
    """
    Processes images by splitting them into quadrants for efficient processing.
    
    Features:
    - Automatic image splitting into quadrants
    - Parallel processing of quadrants
    - Intelligent boundary merging
    - Visualization of quadrant processing
    - Small object removal
    """

    def __init__(self, base_segmenter):
        """
        Initialize quadrant segmenter with base segmentation model.
        
        Args:
            base_segmenter (Segmenter): Instance of base segmentation model
        """
        self.base_segmenter = base_segmenter
        self.model = self.base_segmenter.load_model()
        self.bounding_boxes = None
        self.quadrant_results = {}  # Stores results for visualization

    def segment_by_quadrants(self, img, visualize_quadrants=False, remove_small_obj=False, min_size=20):
        """
        Process image by dividing into quadrants and merging results.
        
        Args:
            img (np.ndarray): Input image
            visualize_quadrants (bool): Whether to show quadrant processing
            remove_small_obj (bool): Whether to filter small objects
            min_size (int): Minimum object size in pixels
            
        Returns:
            tuple: (final_mask, final_boxes)
                - final_mask: Combined segmentation mask
                - final_boxes: List of all detected cell boxes
        """
        # Split image into quadrants
        H, W = img.shape[:2]
        mid_h, mid_w = H // 2, W // 2
        
        # Extract quadrants
        quadrants = {
            'top_left': img[0:mid_h, 0:mid_w],
            'top_right': img[0:mid_h, mid_w:W],
            'bottom_left': img[mid_h:H, 0:mid_w],
            'bottom_right': img[mid_h:H, mid_w:W]
        }
        
        # Process each quadrant
        masks, boxes = {}, {}
        for name, quad in quadrants.items():
            mask, box = self.base_segmenter.segment(quad)
            masks[name] = mask.astype(np.int32)
            boxes[name] = box
            self.quadrant_results[name] = (quad, mask, box)
            
        # Adjust labels to ensure uniqueness
        max_label = 0
        for name in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            if masks[name].max() > 0:
                masks[name][masks[name] > 0] += max_label
                max_label = masks[name].max()
                
        # Combine masks
        final_mask = np.zeros((H, W), dtype=np.int32)
        final_mask[0:mid_h, 0:mid_w] = masks['top_left']
        final_mask[0:mid_h, mid_w:W] = masks['top_right']
        final_mask[mid_h:H, 0:mid_w] = masks['bottom_left']
        final_mask[mid_h:H, mid_w:W] = masks['bottom_right']
        
        # Handle boundary objects and relabel
        final_mask, _ = measure.label(final_mask, background=0, return_num=True)
        
        # Remove small objects if requested
        if remove_small_obj:
            final_mask = self.remove_small_objects(final_mask, min_size)
            
        # Get final properties
        _, final_boxes = self.get_cell_properties(final_mask)
        
        if visualize_quadrants:
            self.visualize_quadrants()
            
        return final_mask, final_boxes

    def remove_small_objects(self, mask, min_size=20):
        """
        Filter out objects smaller than specified size.
        
        Args:
            mask (np.ndarray): Labeled mask
            min_size (int): Minimum object size in pixels
            
        Returns:
            np.ndarray: Filtered mask
        """
        props = measure.regionprops(mask)
        for prop in props:
            if prop.area < min_size:
                mask[mask == prop.label] = 0
        
        # Relabel after removal
        mask, _ = measure.label(mask, background=0, return_num=True)
        return mask

    def get_cell_properties(self, mask):
        """
        Extract properties of detected cells.
        
        Args:
            mask (np.ndarray): Labeled mask
            
        Returns:
            tuple: (centers, boxes)
                - centers: List of cell centroids
                - boxes: List of bounding boxes
        """
        props = measure.regionprops(mask)
        centers = [prop.centroid for prop in props]
        bboxes = [prop.bbox for prop in props]
        boxes = [[float(min_x), float(min_y), float(max_x), float(max_y)] 
                for (min_y, min_x, max_y, max_x) in bboxes]
        return centers, boxes

    def visualize_quadrants(self):
        """
        Display segmentation results for each quadrant.
        Shows original image, mask overlay, and bounding boxes
        for each processed quadrant.
        """
        for quadrant_name, (q_img, q_mask, q_boxes) in self.quadrant_results.items():
            print(f"\nVisualizing {quadrant_name} quadrant:")
            self.visualize(q_img, q_mask, q_boxes)

    def visualize(self, img, mask, boxes):
        """
        Visualize segmentation results for a single quadrant.
        
        Args:
            img (np.ndarray): Original quadrant image
            mask (np.ndarray): Segmentation mask
            boxes (list): Bounding boxes
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        ax1.imshow(img)
        ax1.set_title("Original Quadrant")
        ax1.axis('off')

        # Mask overlay
        ax2.imshow(img)
        colored_mask = np.ma.masked_where(mask == 0, mask)
        ax2.imshow(colored_mask, cmap='hsv', alpha=0.5)
        ax2.set_title("Segmentation Mask")
        ax2.axis('off')

        # Bounding boxes
        ax3.imshow(img)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            ax3.add_patch(rect)
        ax3.set_title("Detected Cells")
        ax3.axis('off')

        plt.tight_layout()
        plt.show()
