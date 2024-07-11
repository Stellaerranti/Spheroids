import cv2
from matplotlib import pyplot as plt
from skimage.feature import  blob_log
import os

plt.close('all')

def process_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply median filter to reduce noise while preserving edges
    median_filtered_image = cv2.medianBlur(image, 5)

    # Normalize the image to enhance contrast
    norm_image = cv2.normalize(median_filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    return image, norm_image

def getCellsCoords(image_path):
    _, processed_image = process_image(image_path)
    blobs= blob_log(processed_image, max_sigma=30, min_sigma = 1, num_sigma=30, threshold=0.01)
    return blobs

def plot3D(blobs):
    
    x = []
    y = []
    z = []
    
    s = []
    
    n = 1
    
    for blob in blobs:
        for i in range(blob.shape[0]):
            x.append(blob[i][0])
            y.append(blob[i][1])
            z.append(n*10)
            s.append(blob[i][2]*5)
        n = n+1
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    
    
    ax.scatter(x,y, z,s = s) 
    

    plt.show()
    
    

path = 'img/test_layers'

files = []
for f in os.listdir(path):
    if os.path.isfile(os.path.join(path, f)):
        files.append(path+'/'+f)

blobs = []        
for image_path in files:
    blobs.append(getCellsCoords(image_path))
    
plot3D(blobs)