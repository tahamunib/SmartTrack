import numpy as np
import cv2

def findContour(image):
    # loop over our contours
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(image, [approx], -1, (0,255,0), 3)
        # compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c)

image = cv2.imread('C:\Python27\us-1.jpg')
cv2.imshow('Image',image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

cv2.imwrite('detect.png', edged)
(_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)





#rect = findContour(image)
#box = cv2.boxPoints(rect)
#box = np.int0(box)
#cv2.drawContours(image,[box],0,(0,0,255),2)
cv2.drawContours(image, cnts[10], -1, (0,255,0), 3)
cv2.imshow('Image',image)
cv2.waitKey(0)
