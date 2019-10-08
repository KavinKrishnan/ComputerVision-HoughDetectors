import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn import cluster
from skimage import color
from skimage import feature
import cv2

def detectCircles(im, radius, useGradient):
    img = cv2.imread(im)
    greyImg = color.rgb2gray(img)
    edges = feature.canny(greyImg, sigma=3)
    bin_size = 9
    # plt.title("Canny Edge Detection")
    # plt.imshow(edges)
    # plt.show()

    dx, dy = np.gradient(greyImg)
    gradientThetas = np.arctan2(-dy, dx)

    h = np.zeros_like(greyImg)
    # bin size expirment
    # h = np.zeros((int(greyImg.shape[0]/bin_size),int(greyImg.shape[0]/bin_size)))

    for index, currEdge in np.ndenumerate(edges):
        if currEdge != 0:
            r = radius
            x = index[1]
            y = index[0]
            if useGradient is 0:
                for t in range(0,360,1):
                    theta = np.radians(t)
                    a = int(int(x - (r * np.cos(theta))))
                    b = int(int(y + (r * np.sin(theta))))
                    # a = int(int(x - (r * np.cos(theta))) / bin_size)
                    # b = int(int(y + (r * np.sin(theta))) / bin_size)
                    if (b in range(edges.shape[0])) and (a in range(edges.shape[1])):
                    # if (b in range(int(greyImg.shape[0]/bin_size))) and (a in range(int(greyImg.shape[1]/bin_size))):
                        h[b,a] += 1
            else:
                theta = gradientThetas[index[0]][index[1]]
                a = int(int(x - (r * np.cos(theta))))
                b = int(int(y + (r * np.sin(theta))))
                # a = int(int(x - (r * np.cos(theta))) / bin_size)
                # b = int(int(y + (r * np.sin(theta))) / bin_size)
                if (b in range(edges.shape[0])) and (a in range(edges.shape[1])):
                # if (b in range(int(greyImg.shape[0] / bin_size))) and (a in range(int(greyImg.shape[1] / bin_size))):
                    h[b, a] += 1

    maxAccum = np.max(h)
    plt.title("Hough Accumulator")
    plt.imshow(h)
    plt.show()

    centers = np.transpose(np.nonzero(h >= (maxAccum * .70)))
    if useGradient is 1:
        centers = np.transpose(np.nonzero(h >= (maxAccum * .78)))


    fig, ax = plt.subplots()
    plt.imshow(img)
    for r in range(centers.shape[0]):
        cx = centers[r][1]
        cy = centers[r][0]
        circle = plt.Circle((cx, cy), radius, color='r', fill=False)
        ax.add_artist(circle)

    plt.title("Predicted Circle")
    plt.show()




# detectCircles('egg.jpg', 5, 0)
# detectCircles('egg.jpg', 5, 1)
#
# detectCircles('jupiter.jpg', 105, 0)
# detectCircles('jupiter.jpg', 105, 1)
