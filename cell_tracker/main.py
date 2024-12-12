"""
Cell Tracker Main Script
----------------------
Command-line interface for cell segmentation.
"""

import argparse
from pathlib import Path
from modules import CellSeg

def main(image_path, output_dir="data/output"):
    """
    Main processing pipeline.
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory for saving results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize and run cell segmentation
        cell_seg = CellSeg()
        
        # Load and process image
        cell_seg.load_image(image_path)
        
        # Remove distance metric if present
        cell_seg.remove_distance_metric()
        
        # Optional: Apply Gaussian blur
        # cell_seg.apply_gaussian_blur()
        
        # Segment the image (choose one method)
        # Option 1: Direct segmentation
        # mask, boxes = cell_seg.segment_image()
        
        # Option 2: Quadrant segmentation
        cell_seg.segment_quadrants(min_size=200)
        
        # Visualize results
        cell_seg.visualize()
        
        print("Processing completed successfully!")
        print(f"Number of cells detected: {cell_seg.num_cells}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cell Segmentation using cellSAM')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--output', type=str, default='data/output',
                      help='Path to output directory')
    
    args = parser.parse_args()
    main(args.image, args.output) 