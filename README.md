# SporeAI

A machine learning toolkit for analyzing microscopy images of bacterial spores and cells.

## Components

### Cell Tracker
The primary component is a cell tracking and segmentation pipeline that uses cellSAM (Segment Anything Model adapted for cellular images) to detect and analyze cells in microscopy images. See the [cell_tracker documentation](cell_tracker/README.md) for details.

## Features

- Automated cell detection and segmentation
- Support for high-resolution microscopy images
- Efficient processing of large images through quadrant-based segmentation
- Configurable preprocessing and post-processing steps
- Command-line interface for batch processing

## Installation

1. Clone the repository:
