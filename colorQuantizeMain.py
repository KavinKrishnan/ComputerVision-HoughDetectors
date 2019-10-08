from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn import cluster
from skimage import color
import cv2

origImg = 'fish.jpg'
k = 5

outputImg1, meanColors2 = quantizeRGB(origImg, k)
imageio.imwrite('quantizeRGBFish'+ str(k) +'.jpg', outputImg1.astype(np.uint8))
print("quantizeRGB Error with n = " + str(k) + ":")
print(computeQuantizationError(origImg, 'quantizeRGBFish'+ str(k) +'.jpg'))


outputImg2, meanColors2 = quantizeHSV(origImg, k)
imageio.imwrite('quantizeHSVFish'+ str(k) +'.jpg', outputImg2.astype(np.uint8))
print("quantizeHSV Error with n = " + str(k) + ":")
print(computeQuantizationError(origImg, 'quantizeHSVFish'+ str(k) +'.jpg'))

getHueHists(origImg, k)

k = 15

outputImg1, meanColors2 = quantizeRGB(origImg, k)
imageio.imwrite('quantizeRGBFish'+ str(k) +'.jpg', outputImg1.astype(np.uint8))
print("quantizeRGB Error with n = " + str(k) + ":")
print(computeQuantizationError(origImg, 'quantizeRGBFish'+ str(k) +'.jpg'))


outputImg2, meanColors2 = quantizeHSV(origImg, k)
imageio.imwrite('quantizeHSVFish'+ str(k) +'.jpg', outputImg2.astype(np.uint8))
print("quantizeHSV Error with n = " + str(k) + ":")
print(computeQuantizationError(origImg, 'quantizeHSVFish'+ str(k) +'.jpg'))

getHueHists(origImg, k)