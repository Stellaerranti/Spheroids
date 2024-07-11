import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

image= cv2.imread('img/test.png',cv2.IMREAD_GRAYSCALE)

#dst = cv2.fastNlMeansDenoising(img,10,10,7,21)


# Step 1: Noise Reduction using Non-Local Means Denoising
denoised = cv2.fastNlMeansDenoising(image, None, h=10)

# Step 2: Contrast Enhancement using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(denoised)

# Step 3: Gaussian Blur to smooth the image
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

# Step 4: Adaptive Thresholding to create a binary image
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)

# Step 5: Find coordinates of bright spots
coords = np.column_stack(np.where(binary > 200))

# Step 6: Apply DBSCAN clustering
db = DBSCAN(eps=5, min_samples=5).fit(coords)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Step 7: Visualize the clusters
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = coords[class_member_mask]
    for x, y in xy:
        cv2.circle(contour_image, (y, x), 1, tuple([int(c * 255) for c in col[:3]]), -1)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Denoised Image")
plt.imshow(denoised, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Contrast Enhanced Image")
plt.imshow(enhanced, cmap='gray')

plt.subplot(2, 2, 4)
plt.title(f"Detected Clusters: {n_clusters_}")
plt.imshow(contour_image)

plt.show()

print(f"Total number of clusters detected: {n_clusters_}")

#plt.subplot(121),plt.imshow(img)
#plt.subplot(122),plt.imshow(dst)
#plt.show()