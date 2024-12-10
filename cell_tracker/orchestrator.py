"""
Cell Tracker Orchestrator
------------------------
Provides a high-level interface to coordinate all cell tracking operations.
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
    Orchestrates the cell tracking pipeline by coordinating all processing steps.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the pipeline with configuration from yaml file.
        
        Args:
            config_path (str): Path to configuration yaml file
        """
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._get_default_config()

        self.config = config
        
        # Initialize components with parameters
        self.preprocessor = Preprocessor(
            **config['preprocessing']
        )
        
        self.segmenter = Segmenter(
            bbox_threshold=config['segmentation']['bbox_threshold']
        )
        
        # Initialize quadrant segmenter
        self.quadrant_segmenter = QuadrantSegmenter(self.segmenter)
        
    def _get_default_config(self):
        """Return default configuration if no config file provided."""
        return {
            'preprocessing': {
                'inpaint': True,
                'gaussian_blur_enabled': False,
                'gaussian_blur_sigma': 1.0,
                'gamma_correction_enabled': False,
                'gamma': 1.2
            },
            'segmentation': {
                'bbox_threshold': 0.4,
                'quadrant_mode': False,
                'device': 'auto'
            },
            'postprocessing': {
                'min_object_size': 200,
                'merge_overlap_threshold': 0.5,
                'remove_small_objects': True,
                'min_size': 20
            }
        }
        
    def process_image(self, image_path, output_dir="data/output", quadrant_mode=False, visualize=False):
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory for output files
            quadrant_mode (bool): Whether to use quadrant segmentation
            visualize (bool): Whether to visualize results
            
        Returns:
            dict: Results containing masks and cell properties
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        img = ImageLoader.load_image(image_path)
        preprocessed_img = self.preprocessor.process(img)
        
        # Perform segmentation
        if quadrant_mode:
            mask, boxes = self.quadrant_segmenter.segment_by_quadrants(
                preprocessed_img,
                visualize_quadrants=visualize,
                remove_small_obj=self.config['postprocessing']['remove_small_objects'],
                min_size=self.config['postprocessing']['min_size']
            )
        else:
            mask, boxes = self.segmenter.segment(preprocessed_img)
            
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
    parser = argparse.ArgumentParser(description='Cell Tracking Pipeline')
    parser.add_argument('input_file', help='Path to input TIFF file')
    parser.add_argument('-q', '--quadrant', action='store_true', help='Enable quadrant segmentation')
    parser.add_argument('-c', '--config', help='Path to config file', default=None)
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize results')
    parser.add_argument('-o', '--output', help='Output directory', default='data/output')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CellTrackingPipeline(config_path=args.config)
    
    # Process image
    results = pipeline.process_image(
        args.input_file,
        output_dir=args.output,
        quadrant_mode=args.quadrant,
        visualize=args.visualize
    )
    
    print(f"Processing complete. Results saved to {args.output}")
    
if __name__ == "__main__":
    main()