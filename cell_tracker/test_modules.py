import unittest
import numpy as np
from pathlib import Path
import torch
from cell_tracker.modules import CellSeg

class TestCellSeg(unittest.TestCase):
    """Test cases for CellSeg class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cell_seg = CellSeg()
        
        # Create a simple test image (RGB)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some "cells" (white circles)
        self.test_image[20:30, 20:30] = 255  # Cell 1
        self.test_image[60:70, 60:70] = 255  # Cell 2
        
        # Add a "scale bar" artifact
        self.test_image[90:95, 10:30, 0] = 255  # Red channel only
        
        # Save test image
        self.test_image_path = Path("test_image.tif")
        try:
            from skimage import io
            io.imsave(str(self.test_image_path), self.test_image)
        except Exception as e:
            print(f"Failed to save test image: {e}")
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_image_path.exists():
            self.test_image_path.unlink()
    
    def test_load_image(self):
        """Test image loading functionality."""
        # Test loading valid image
        self.cell_seg.load_image(str(self.test_image_path))
        self.assertIsNotNone(self.cell_seg.img)
        self.assertEqual(self.cell_seg.img.shape, (100, 100, 3))
        
        # Test loading non-existent image
        with self.assertRaises(FileNotFoundError):
            self.cell_seg.load_image("nonexistent.tif")
    
    def test_remove_distance_metric(self):
        """Test scale bar removal."""
        self.cell_seg.load_image(str(self.test_image_path))
        original_artifact = self.cell_seg.img[90:95, 10:30, 0].copy()
        
        self.cell_seg.remove_distance_metric()
        
        # Check if the artifact was modified
        modified_region = self.cell_seg.img[90:95, 10:30, 0]
        self.assertFalse(np.array_equal(original_artifact, modified_region))
    
    def test_apply_gaussian_blur(self):
        """Test Gaussian blur application."""
        self.cell_seg.load_image(str(self.test_image_path))
        original_img = self.cell_seg.img.copy()
        
        self.cell_seg.apply_gaussian_blur(kernel=(5,5), sigma=1)
        
        # Check if image was modified
        self.assertFalse(np.array_equal(original_img, self.cell_seg.img))
    
    def test_segment_image(self):
        """Test direct image segmentation."""
        self.cell_seg.load_image(str(self.test_image_path))
        
        mask, boxes = self.cell_seg.segment_image(bounding_box_threshold=0.4)
        
        # Basic checks on the output
        self.assertIsNotNone(mask)
        self.assertIsNotNone(boxes)
        self.assertEqual(mask.shape[:2], self.test_image.shape[:2])
    
    def test_segment_quadrants(self):
        """Test quadrant-based segmentation."""
        self.cell_seg.load_image(str(self.test_image_path))
        
        self.cell_seg.segment_quadrants(min_size=20)
        
        # Check results
        self.assertIsNotNone(self.cell_seg.mask)
        self.assertIsNotNone(self.cell_seg.bounding_boxes)
        self.assertIsNotNone(self.cell_seg.num_cells)
        self.assertIsNotNone(self.cell_seg.centroids)
        
        # Check mask dimensions
        self.assertEqual(
            self.cell_seg.mask.shape[:2], 
            self.test_image.shape[:2]
        )
    
    def test_cuda_availability(self):
        """Test CUDA device handling."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertIsNotNone(device)

def create_test_suite():
    """Create a test suite with all test cases."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCellSeg))
    return suite

if __name__ == '__main__':
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    runner.run(test_suite) 