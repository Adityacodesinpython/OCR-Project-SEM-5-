import cv2 
import matplotlib.pyplot as plt
import numpy as np


# Read and resize the image
img_color = cv2.imread(r"mini-proj-assets\Device3\44.88.jpeg")  # Updated image path
img_color = cv2.resize(img_color, None, None, fx=0.7, fy=0.7)  # Resize image

# Convert to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Use adaptive thresholding to focus on the bright area
# Tweak the parameters for better region detection
thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

# Dilate the thresholded image to better capture the region
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)

# Find contours on the dilated image
cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out contours that are too small or not rectangular (we expect the region to be large and roughly rectangular)
filtered_contours = [c for c in cnts if cv2.contourArea(c) > 500]  # Adjusted threshold for smaller areas
filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

# Assuming the display screen is one of the larger areas, select the largest contour
largest_contour = filtered_contours[0]

# Create a mask for the selected contour (assuming it's the display area)
mask = np.zeros_like(img_gray)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Extract the region of interest (ROI) using the mask
roi = cv2.bitwise_and(img_color, img_color, mask=mask)

# Find the bounding box of the selected contour to crop the ROI
x, y, w, h = cv2.boundingRect(largest_contour)
cropped_green_screen = roi[y:y+h, x:x+w]

# Display the cropped region (ROI) without performing text detection
cv2.imshow("ROI Cropped ", cropped_green_screen)

# Display the final image
# plt.imshow(cv2.cvtColor(cropped_green_screen, cv2.COLOR_BGR2RGB))
# plt.show()
cv2.imwrite("cropped_green_screen.jpeg",cropped_green_screen)

