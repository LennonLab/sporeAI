# Cell Tracker

A Python package for automated cell segmentation and tracking using cellSAM (Segment Anything Model adapted for cellular images). The package provides efficient processing of microscopy images with support for large image processing through quadrant-based segmentation.

## Features

- Automated cell segmentation using cellSAM
- Support for high-resolution TIFF images
- Quadrant-based processing for large images with intelligent boundary merging
- Automatic artifact removal and preprocessing
- Visualization tools for segmentation results
- Configurable via YAML files
- Command-line interface for batch processing

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for optimal performance)


# Basic usage
python orchestrator.py input_file [options]

Options:
  -q, --quadrant         Enable quadrant segmentation
  -c, --config CONFIG    Path to config file
  -v, --visualize       Visualize results
  -o, --output OUTPUT   Output directory (default: data/output)

python orchestrator.py path/to/image.tif

# With all options
python orchestrator.py path/to/image.tif -q -v -c custom_config.yaml -o custom/output/dir