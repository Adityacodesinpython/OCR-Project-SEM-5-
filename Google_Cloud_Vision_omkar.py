"""
need to install google-cloud-vision (GCP SDK) from conda -c conda-forge
conda install -c conda-forge pillow=10.1.0 pandas=2.1.2 google-cloud-vision=3.4.5 scikit-learn=1.3.2 ipykernel jupyterlab notebook python=3.12.0
to set up in jupyterlabs:
python -m ipykernel install --user --name=gcp-cloud-vision
repo: https://github.com/donaldsrepo/gcp-solution
"""

import os
from os import listdir
from os.path import isfile, join
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='vision_omkar.json'
from google.cloud import vision
from my_timer import my_timer
import time
import cv2
from decimal import Decimal, InvalidOperation

# Load the image
image_path = 'mini-proj-assets\WhatsApp Image 2023-11-29 at 08.28.35 (2).jpeg'  # Change this to your actual image path
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to get a clean binary image for edge detection
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Use edge detection
edges = cv2.Canny(thresh, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Variable to store the largest rectangle
largest_area = 0
largest_rect = None

# Loop through contours to find the largest rectangle
for contour in contours:
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(contour, True)  # Smaller epsilon for a more accurate shape
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the contour has 4 corners, it might be a rectangle
    if len(approx) == 4:
        # Compute the bounding box of the contour and its area
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        
        # Filter based on area to avoid very small or large objects
        if area > 1000:  # Adjust this threshold based on your image size
            # Update if this is the largest rectangle found so far
            if area > largest_area:
                largest_area = area
                largest_rect = (x, y, w, h)

# If a largest rectangle was found, crop and save it
if largest_rect is not None:
    x, y, w, h = largest_rect
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped largest rectangle
    cropped_image_path = 'cropped_largest_rectangle.jpeg'
    cv2.imwrite(cropped_image_path, cropped_image)
    print("Largest rectangle saved at:", cropped_image_path)
else:
    print("No rectangle detected.")


def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    ocr_text = []
    for text in texts:
        ocr_text.append(f"\r\n{text.description}")
        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        # print("bounds: {}".format(",".join(vertices)))        
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    # return texts[0].description

    return ocr_text

# @my_timer
def main():
    texts = detect_text(r"D:\Aditya_Work\VS_Code\sem_mini_project\ocr\OCR-Project-SEM-5-\cropped_largest_rectangle.jpeg")
    # print(image_path)
    for text in texts:
        try :
            num = Decimal(text)
            # if num == float :
            print(text) 
            

        except InvalidOperation:
            continue

if __name__ == "__main__":
     main()
