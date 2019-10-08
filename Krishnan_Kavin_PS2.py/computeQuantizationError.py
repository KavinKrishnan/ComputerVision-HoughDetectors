import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn import cluster
from skimage import color

def computeQuantizationError(origImg,quantizedImg):
    img1 = imageio.imread(origImg)
    img2 = imageio.imread(quantizedImg)
    width, height, rgb3 = img1.shape

    flattenedImg1 = np.reshape(img1, (width * height * 3,1)).astype(float)
    flattenedImg2 = np.reshape(img2, (width * height * 3,1)).astype(float)

    difference = flattenedImg1 - flattenedImg2
    differenceSq = np.square(difference)
    # difference = np.linalg.norm(img1 - img2, axis=2)
    # differenceSq = np.square(difference)

    return np.sum(differenceSq).astype(np.uint64)

# print(computeQuantizationError('fish.jpg', 'fishhsv4.jpg'))
