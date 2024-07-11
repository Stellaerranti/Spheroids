import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import skimage as ski


plt.close('all')

def process_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply median filter to reduce noise while preserving edges
    median_filtered_image = cv2.medianBlur(image, 5)

    # Normalize the image to enhance contrast
    norm_image = cv2.normalize(median_filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    return image, norm_image


image_path = 'img/test3.png'


image, processed_image = process_image(image_path)



fig, ax = plt.subplots(figsize=(5, 5))
qcs = ax.contour(processed_image, origin='image')
ax.set_title('Contour plot of the same raw image')
plt.show()

thresholds = ski.filters.threshold_multiotsu(image, classes=3)
regions = np.digitize(image, bins=thresholds)


cells = image > thresholds[0]
dividing = image > thresholds[1]
labeled_cells = ski.measure.label(cells)
labeled_dividing = ski.measure.label(dividing)
naive_mi = labeled_dividing.max() / labeled_cells.max()

print(labeled_cells.max())