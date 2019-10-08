import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn import cluster
from skimage import color
from quantizeHSV import quantizeHSV
import cv2

def getHueHists(im, k):
    img = imageio.imread(im)
    hsvImg = color.rgb2hsv(img)
    width, height, hsvS = hsvImg.shape

    imgHues = np.reshape(hsvImg[:, :, 0], (width * height, 1)) * 179.0
    histEqual = plt.hist(imgHues, bins=k)
    plt.show()

    hsvImg = color.rgb2hsv(img)
    width, height, hsvS = hsvImg.shape

    imgHues = np.reshape(hsvImg[:, :, 0], (width * height, 1))

    kCluster = cluster.KMeans(n_clusters=k)
    kCluster.fit(imgHues)
    meanHues = kCluster.cluster_centers_

    centerInds = kCluster.predict(imgHues)
    imgClusters = meanHues[centerInds]
    imgClusters = imgClusters * 179.0

    meanHuesSorted = np.sort(meanHues.flatten()) * 179.0
    bins = []

    if k > 1:

        firstEdge = meanHuesSorted[0] - (((meanHuesSorted[0] + meanHuesSorted[1])/2.0) - meanHuesSorted[0])

        if firstEdge > 0:
            bins = [firstEdge]
        else:
            bins = [0]

        for c in range(len(meanHuesSorted)-1):
            bins.append((meanHuesSorted[c] + meanHuesSorted[c+1])/2.0)

        lastEdge = meanHuesSorted[len(meanHuesSorted)-1] + (((meanHuesSorted[len(meanHuesSorted)-1] + meanHuesSorted[len(meanHuesSorted)-2])/2.0) - meanHuesSorted[len(meanHuesSorted)-2])
        if lastEdge < 179.0:
            bins.append(lastEdge)
        else:
            bins.append(179.0)
    else:
        bins = 1

    histClustered = plt.hist(imgClusters, bins=bins)
    plt.show()

    return histEqual, histClustered
