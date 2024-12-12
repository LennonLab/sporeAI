# Cell Tracker

A Python package for automated cell segmentation using cellSAM (Segment Anything Model adapted for cellular images). This package provides efficient processing of microscopy images with support for both direct and quadrant-based segmentation.

## Features

- Automated cell segmentation using cellSAM
- Support for high-resolution TIFF images
- Two segmentation modes:
  - Direct segmentation for smaller images
  - Quadrant-based processing for large images
- Automatic artifact removal (10µm scale bar removal)
- Optional Gaussian blur preprocessing
- Built-in visualization tools
- Simple command-line interface

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for optimal performance)

### Using Conda (Recommended)