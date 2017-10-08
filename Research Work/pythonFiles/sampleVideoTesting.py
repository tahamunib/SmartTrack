import numpy as np
import cv2

cap = cv2.VideoCapture('C:/Python27/d1.mp4')

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')

out = cv2.VideoWriter('C:/Python27/New2.mp4',fourcc,25.0,(640,480))

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

startSec = 5
endSec = 15
totalSec = int(length/cap.get(5))
cap.set(0,startSec*1000)

while(cap.isOpened()):
	ret,frame = cap.read()
	if ret == True:
		out.write(frame)
		cv2.imshow('Video',frame)
		k = cv2.waitKey(45) & 0xFF
		currPos = int(cap.get(0)/1000)
		if k == ord('q') or currPos == endSec:
			break
		elif k == ord('p'):
			cv2.waitKey(0)
	else:
		break




cap.release()
out.release()
cv2.destroyAllWindows()
