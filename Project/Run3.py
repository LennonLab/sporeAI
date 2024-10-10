import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_path = "20240917_delta6_5.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
output_folder = "Output3"
os.makedirs(output_folder, exist_ok=True)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply thresholding to segment the cells
_, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)


kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert the grayscale image to BGR (for color annotations)
image_annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


margin = 10


pixels_per_10um = 100  
pixel_to_micron_ratio = 10 / pixels_per_10um


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 0, 0)  
font_thickness = 1


for i, contour in enumerate(contours):
    # Compute the bounding rectangle for each contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add margin to the bounding box
    x_new = max(0, x - margin)
    y_new = max(0, y - margin)
    w_new = min(image.shape[1], x + w + margin) - x_new
    h_new = min(image.shape[0], y + h + margin) - y_new
    
    # Crop the detected cell from the original image with extra margin
    cell = image[y_new:y_new+h_new, x_new:x_new+w_new]
    
    
    cell_filename = f"{output_folder}/cell_{i+1}.png"
    cv2.imwrite(cell_filename, cell)
    
    
    width_in_microns = w * pixel_to_micron_ratio
    height_in_microns = h * pixel_to_micron_ratio
    
    # Draw the exact outline of the detected cell contour for visualization
    cv2.drawContours(image_annotated, [contour], -1, (0, 255, 0), 2)
    
    # Annotate the image with the size of the cell in microns
    size_text = f"{width_in_microns:.2f} um x {height_in_microns:.2f} um"
    text_position = (x, y - 10)  
    cv2.putText(image_annotated, size_text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# Display the result with exact contours and size annotations
plt.figure(figsize=(10, 10))
plt.imshow(image_annotated)
plt.title("Cells with Size Annotations (in Microns)")
plt.axis("off")
plt.show()

num_cells = len(contours)
print(f"Number of cells detected: {num_cells}")

