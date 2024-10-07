# image processing
import cv2 as cv

# reading images
from skimage import io

# numerical operations on arrays
import numpy as np

# displaying images
import matplotlib.pyplot as plt

# advanced image processing operations
from scipy import ndimage

import math


def showImages(images_dict):
    n = len(images_dict)
    cols = math.ceil(np.sqrt(n))
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*rows, 5*cols))
    axes = axes.ravel()  # flatten array of axes for easy indexing

    for i, (name, image) in enumerate(images_dict.items()):
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(name)
        axes[i].axis('off')

    for j in range(n, rows*cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("images.png")

###################################################
############# Loading Step#########################
img_path = "../images/20240917_delta6_5.tif"
# load microscopy image in grayscale
image_0 = io.imread(img_path, as_gray=True)

# normalize image to [0,1] -> [0,255] and convert to uint8 for opencv compatability
image_0 = (image_0*255).astype(np.uint8)


# Blurring
# Gaussian blurring - applies a gaussian kernal to smooth the image
# kernel size (5,5) determines the extent of the blurring
blurred = cv.GaussianBlur(image_0, (5,5), 0)


############ THRESHOLDING #############################
# Otsu Thresholding
# automatically determines optimal threshold value to separate background and foreground
# produces binary image where pixels are either 0 or 255
# used equalized or blurred
_, otsu_image = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Manual Thresholding
# uses a fixed threshold to binarize the image
thres_value = 90
_, manual_image = cv.threshold(blurred, thres_value, 255, cv.THRESH_BINARY)

# Adaptive Thresholding
adaptive_image = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


############ WATERSHED ###############################
# used to separate overlapping or touching cells by treating the image as a topographical map

# Apply distance transform to highlight cell centers
# computes the distance from every pixel to the nearest zero pixel (background) in the binary image
# results in a distance map where the centers of cells appear as peaks
distance_transform = ndimage.distance_transform_edt(otsu_image, cv.DIST_L2, 5)

# identify the sure foreground (cell centers) by thresholding the distance map
_, sure_fg = cv.threshold(distance_transform, 0.6*distance_transform.max(), 255, 0)
print(f'max distance: {distance_transform.max()}')
# convert the result to 8bit unsigned integers
sure_fg = np.uint8(sure_fg)

# perform dilation on the binary image to obtain the sure background areas
# kernel (3,3) matrix of ones used for morphological operations
kernel = np.ones((3,3), np.uint8)
sure_bg = cv.dilate(otsu_image, kernel, iterations=3)

# subtract sure foreground from sure background to get regions where its unknown if its bg or fg
unknown_regions = cv.subtract(sure_bg, sure_fg)

# label the sure foreground regions
_, markers = cv.connectedComponents(sure_fg)
# background is labeled as 1
markers = markers + 1
# unknown regions are labeled as 0
markers[unknown_regions==255] = 0

# apply the watershed algorithm using markers as starting points
# convert to BGR color space (required for opencv watershed)
# updates the markers array with segmentation labels
markers = cv.watershed(cv.cvtColor(blurred, cv.COLOR_GRAY2BGR), markers)

# highlights boundaries found by the watershed algorithm by setting them to white
boundaries_image = blurred.copy()
boundaries_image[markers==-1] = [255]


############ COUNT CELLS #######################

#????? How do I count the individual occurances of a certain region?
#??? I think I'm counting the number of distinct regions instead of the number of cells

unique_markers = np.unique(markers)

# exclude background label (1) and boundary label (-1)
cell_labels = unique_markers[(unique_markers != 1) & (unique_markers != -1)]
num_cells = len(cell_labels)


# Optionally, create a labeled image for visualization
labeled_image = np.zeros_like(markers, dtype=np.uint8)
for label in cell_labels:
    labeled_image[markers == label] = 255

print(f"number of cells found: {num_cells}")

images_dict = {
    'image_0': image_0,
    'blurred': blurred,
    'adaptive_image': adaptive_image,
    'otsu_image': otsu_image,
    'manual_image': manual_image,
    'distance_transform': distance_transform,
    'sure_fg': sure_fg,
    'sure_bg': sure_bg,
    'unknown_regions': unknown_regions,
    'markers': markers,
    'boundaries_image': boundaries_image,
    'labeled_image': labeled_image
}

showImages(images_dict)







