"""
Cell Tracker Orchestrator
------------------------
This module serves as the main entry point for the cell tracking pipeline.
It coordinates all processing steps and provides both a command-line interface
and a programmatic API for cell segmentation tasks.

Key Components:
- CellTrackingPipeline: Main class that orchestrates the entire pipeline
- Configuration management through YAML files
- Command-line interface for easy usage
- Support for both regular and quadrant-based segmentation
"""

import argparse
import yaml
from pathlib import Path
from modules import (
    ImageLoader,
    Preprocessor,
    Segmenter,
    QuadrantSegmenter
)

class CellTrackingPipeline:
    """
    Orchestrates the complete cell tracking pipeline.
    
    This class coordinates all processing steps including:
    1. Loading and validating configuration
    2. Image preprocessing
    3. Cell segmentation (regular or quadrant-based)
    4. Result visualization and saving
    
    The pipeline can be configured through a YAML file or use default settings.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path (str, optional): Path to YAML configuration file.
                If not provided, default configuration will be used.
        """
        # Load configuration from file or use defaults
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._get_default_config()

        self.config = config
        
        # Initialize pipeline components with configured parameters
        self.preprocessor = Preprocessor(
            **config['preprocessing']
        )
        
        self.segmenter = Segmenter(
            bbox_threshold=config['segmentation']['bbox_threshold']
        )
        
        # Initialize quadrant segmenter with base segmenter
        self.quadrant_segmenter = QuadrantSegmenter(self.segmenter)
        
    def _get_default_config(self):
        """
        Provide default configuration settings.
        
        Returns:
            dict: Default configuration parameters for all pipeline components
        """
        return {
            'preprocessing': {
                'inpaint': True,  # Remove imaging artifacts
                'gaussian_blur_enabled': False,  # Optional noise reduction
                'gaussian_blur_sigma': 1.0,  # Blur intensity if enabled
                'gamma_correction_enabled': False,  # Optional contrast adjustment
                'gamma': 1.2  # Gamma correction value if enabled
            },
            'segmentation': {
                'bbox_threshold': 0.4,  # Confidence threshold for cell detection
                'quadrant_mode': False,  # Whether to use quadrant processing
                'device': 'auto'  # Auto-select processing device (CPU/GPU)
            },
            'postprocessing': {
                'min_object_size': 200,  # Minimum cell size in pixels
                'merge_overlap_threshold': 0.5,  # Threshold for merging overlapping cells
                'remove_small_objects': True,  # Filter out small objects
                'min_size': 20  # Minimum size threshold for filtering
            }
        }
        
    def process_image(self, image_path, output_dir="data/output", quadrant_mode=False, visualize=False):
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path (str): Path to input image file
            output_dir (str): Directory for saving results
            quadrant_mode (bool): Whether to use quadrant-based processing
            visualize (bool): Whether to display visualization plots
            
        Returns:
            dict: Results containing:
                - original_image: Raw input image
                - preprocessed_image: After preprocessing
                - segmentation_mask: Binary mask of detected cells
                - bounding_boxes: Coordinates of detected cells
        """
        # Create output directory if needed
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        img = ImageLoader.load_image(image_path)
        preprocessed_img = self.preprocessor.process(img)
        
        # Perform segmentation based on selected mode
        if quadrant_mode:
            mask, boxes = self.quadrant_segmenter.segment_by_quadrants(
                preprocessed_img,
                visualize_quadrants=visualize,
                remove_small_obj=self.config['postprocessing']['remove_small_objects'],
                min_size=self.config['postprocessing']['min_size']
            )
        else:
            mask, boxes = self.segmenter.segment(preprocessed_img)
            
        # Visualize results if requested
        if visualize:
            if quadrant_mode:
                self.quadrant_segmenter.visualize(preprocessed_img, mask, boxes)
            else:
                self.segmenter.visualize(preprocessed_img, mask, boxes)
        
        return {
            'original_image': img,
            'preprocessed_image': preprocessed_img,
            'segmentation_mask': mask,
            'bounding_boxes': boxes
        }

def main():
    """
    Command-line interface entry point.
    
    Provides command-line options for:
    - Input image path
    - Quadrant mode toggle
    - Configuration file path
    - Visualization toggle
    - Output directory
    """
    parser = argparse.ArgumentParser(description='Cell Tracking Pipeline')
    parser.add_argument('input_file', help='Path to input TIFF file')
    parser.add_argument('-q', '--quadrant', action='store_true', help='Enable quadrant segmentation')
    parser.add_argument('-c', '--config', help='Path to config file', default=None)
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize results')
    parser.add_argument('-o', '--output', help='Output directory', default='data/output')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = CellTrackingPipeline(config_path=args.config)
    results = pipeline.process_image(
        args.input_file,
        output_dir=args.output,
        quadrant_mode=args.quadrant,
        visualize=args.visualize
    )
    
    print(f"Processing complete. Results saved to {args.output}")
    
if __name__ == "__main__":
    main()