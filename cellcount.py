import cv2
import numpy as np
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.cluster import DBSCAN

plt.close('all')


def conv(x, kernel, pad):
    H = x.shape[0]
    W = x.shape[1]

    Kh = kernel.shape[0]
    Kw = kernel.shape[1]

    H1 = 1+(H+2*pad-Kh)
    W1 = 1+(W+2*pad-Kw)

    res = np.zeros(shape = (H1,W1))

    ph = np.zeros(shape=(H,pad))
    pw = np.zeros(shape = (pad,W+2*pad))

    x=np.hstack((ph,x,ph))
    x=np.vstack((pw,x,pw))

    for i in range(H1):
      for j in range(W1):
        sum = 0
        for k in range(Kh):
          for f in range(Kw):
            sum+=x[i+k][j+f]*kernel[k][f]
        res[i][j]=sum
    return res


def process_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply median filter to reduce noise while preserving edges
    median_filtered_image = cv2.medianBlur(image, 5)

    # Normalize the image to enhance contrast
    norm_image = cv2.normalize(median_filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    return image, norm_image




image_path = 'img/test.png'

image, processed_image = process_image(image_path)

kernelx = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
kernely = np.array([[-1, 0, 1],[-2,0,2],[-1,0,1]])

imx = conv(processed_image,kernelx,pad=3)
imy = conv(processed_image,kernely,pad=3)

image_sobel = np.sqrt(imx*imx+imy*imy)

max_value = image_sobel.max()

map_L = []
for i in range(image_sobel.shape[0]):
  for j in range(image_sobel.shape[1]):
    if(image_sobel[i][j]>= 0.5*max_value):
      map_L.append((j,i))
map_max = np.asarray(map_L)

db = DBSCAN(eps=0.3, min_samples=10).fit(map_max)
labels = db.labels_


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


