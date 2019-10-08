import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn import cluster

def quantizeRGB(origImg, k):
    img = imageio.imread(origImg)
    width, height, rgb3 = img.shape

    flattenedImg = np.reshape(img, (width * height, 3))

    kCluster = cluster.KMeans(n_clusters=k)
    kCluster.fit(flattenedImg)

    meanColors = kCluster.cluster_centers_

    centerInds = kCluster.predict(flattenedImg)
    imgClusters = meanColors[centerInds]
    outputImg = np.reshape(imgClusters, (width, height, 3))

    return outputImg, meanColors

# outputImg, meanColors = quantizeRGB('fish.jpg', 4)
# plt.imshow(outputImg.astype(np.uint8))
# plt.show()
