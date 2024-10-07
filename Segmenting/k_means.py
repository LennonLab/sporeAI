import cv2 as cv
from skimage import io
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import tifffile

def showImages(images):
    n = len(images)
    cols = math.ceil(np.sqrt(n))
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize = (5*rows, 5*cols))
    axes = axes.ravel() # flatten array of axes for easy indexing

    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].set_title(f"image_{i}")
        axes[i].axis('off')

    for j in range(n, rows*cols):
        axes[j].axis('off')

    plt.show()


img_path = "../../CellSegmentation/images/20240917_delta6_6.tif"


with tifffile.TiffFile(img_path) as tif:
    image = tif.asarray()
    if image.shape == (1608, 1608, 3):
        # print(f"image: {image.shape}")
        # print('transposing')
        image = np.transpose(image, (2,0,1))



image_reshaped = image.reshape(3, -1).T

print(image_reshaped.shape)

## K-means clustering
n_clusters = 2

kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=33)
labels = kmeans.fit_predict(image_reshaped)

clustered_image = labels.reshape(1608, 1608)


fig, ax = plt.subplots()




showImages([np.transpose(image, (1, 2, 0)),clustered_image])

