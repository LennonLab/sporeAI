"""
Cell Tracker Main Script
----------------------
Command-line interface for the cell tracking pipeline.

This script provides a simple interface for:
- Processing individual images
- Saving results to specified output directory
- Configuring processing parameters
"""

import argparse
from pathlib import Path
from modules import (
    ImageLoader,
    Preprocessor,
    Segmenter,
    PostProcessor,
    Recombiner
)
import os
from skimage import io
import json
import numpy as np

def save_output(img, filename, output_dir):
    """
    Save processed image to output directory.
    
    Args:
        img (np.ndarray): Image to save
        filename (str): Output filename
        output_dir (str/Path): Output directory path
    """
    output_path = Path(output_dir) / filename
    io.imsave(str(output_path), img)
    print(f"Saved output to: {output_path}")

def main(image_path, output_dir="data/output"):
    """
    Main processing pipeline.
    
    Workflow:
    1. Load and validate input image
    2. Preprocess image
    3. Perform segmentation
    4. Post-process results
    5. Save outputs
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory for saving results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and process image
        img = ImageLoader.load_image(image_path)
        
        # Preprocess
        preprocessor = Preprocessor()
        img_preprocessed = preprocessor.process(img)
        save_output(img_preprocessed, 'preprocessed.tif', output_dir)
        
        # Segment
        segmenter = Segmenter()
        mask = segmenter.segment(img_preprocessed)
        save_output(mask, 'raw_segmentation.tif', output_dir)
        
        # Post-process
        postprocessor = PostProcessor()
        final_mask = postprocessor.process(mask)
        
        # Save results
        save_output(final_mask, 'final_segmentation.tif', output_dir)
        properties = postprocessor.get_object_properties(final_mask)
        
        # Save properties as JSON
        properties_path = Path(output_dir) / 'cell_properties.json'
        with open(properties_path, 'w') as f:
            json.dump(properties, f, indent=2, cls=NumpyEncoder)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    
    def default(self, obj):
        """Convert NumPy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cell Segmentation using cellSAM')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--output', type=str, default='data/output',
                      help='Path to output directory')
    
    args = parser.parse_args()
    main(args.image, args.output) 