import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply median filter to reduce noise while preserving edges
    median_filtered_image = cv2.medianBlur(image, 5)
    
    #norm = np.zeros(median_filtered_image.shape)

    # Normalize the image to enhance contrast
    norm_image = cv2.normalize(median_filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    return image, norm_image

original_image1, processed_image1 = process_image('img/test3.png')

plt.imshow(processed_image1, cmap='gray')
plt.show()