import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn import cluster
from skimage import color

def quantizeHSV(origImg, k):
    img = imageio.imread(origImg)
    hsvImg = color.rgb2hsv(img)
    width, height, hsvS = hsvImg.shape

    imgHues = np.reshape(hsvImg[:,:,0], (width * height, 1))

    kCluster = cluster.KMeans(n_clusters=k)
    kCluster.fit(imgHues)
    meanHues = kCluster.cluster_centers_

    centerInds = kCluster.predict(imgHues)
    imgClusters = meanHues[centerInds]
    newHues = np.reshape(imgClusters, (width, height))
    hsvImg[:, :, 0] = newHues

    outputImg = color.hsv2rgb(hsvImg) * 255

    return outputImg, meanHues


# outputImg, meanColors = quantizeHSV('fish.jpg', 4)
# plt.imshow(outputImg.astype(np.uint8))
# imageio.imwrite ('fishhsv4.jpg', outputImg.astype(np.uint8))
# plt.show()