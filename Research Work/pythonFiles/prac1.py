import cv2
import numpy as np
import preprocess

img = cv2.imread('us-2.jpg')
height, width, numChannels = img.shape

cv2.imshow('Image',img)
cv2.waitKey(0)

imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
imgThreshScene = np.zeros((height, width, 1), np.uint8)
imgContours = np.zeros((height, width, 3), np.uint8)

imgGrayscaleScene, imgThreshScene = preprocess.preprocess(img)

cv2.imshow("1a", imgGrayscaleScene)
cv2.imshow("1b", imgThreshScene)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_blue = np.array([100,50,50])
upper_blue = np.array([180,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(imgGray, lower_blue, upper_blue)
cv2.imshow('Gray',mask)
cv2.waitKey(0)
