import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from math import sqrt

plt.close('all')

def process_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply median filter to reduce noise while preserving edges
    median_filtered_image = cv2.medianBlur(image, 5)

    # Normalize the image to enhance contrast
    norm_image = cv2.normalize(median_filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    return image, norm_image


image_path = 'img/test_layers/1.png'

_, processed_image = process_image(image_path)


'''
#Plot image as 3d surface
lena = cv2.resize(processed_image, (100,100))

xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1,  linewidth=0)

# show it
plt.show()
'''

blobs_log = blob_log(processed_image, max_sigma=30, min_sigma = 1, num_sigma=30, threshold=0.01)

blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(processed_image, max_sigma=15, threshold=0.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(processed_image, max_sigma=15, threshold=0.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(processed_image,  cmap='gray')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()

print(blobs_log.shape[0])