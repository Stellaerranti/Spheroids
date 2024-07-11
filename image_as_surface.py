import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

plt.close('all')

def process_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply median filter to reduce noise while preserving edges
    median_filtered_image = cv2.medianBlur(image, 5)

    # Normalize the image to enhance contrast
    norm_image = cv2.normalize(median_filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    return image, norm_image


image_path = 'img/test.png'

_, processed_image = process_image(image_path)



#Plot image as 3d surface
lena = cv2.resize(processed_image, (100,100))

xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1,  linewidth=0)

# show it
plt.show()
