from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt

# -------- camShift iteration function start --------------

def func(roi_histogram,roiBox,term_criteria):
	global breakStatus,perA,mask
	while(cap.isOpened()):
		ret,frame = cap.read()
		if ret == True:
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			hsv = cv2.GaussianBlur(hsv,(1,1),0)
	
			backProj = cv2.calcBackProject([hsv],[0],roi_histogram,[0,180],1)
			retval, threshold = cv2.threshold(backProj, 76, 255, cv2.THRESH_BINARY)
			backProj = cv2.normalize(threshold,threshold,0,255,cv2.NORM_MINMAX)
			
			ret, roiBox = cv2.CamShift(backProj, roiBox, term_criteria)
					
			# Draw it on image
			pts = cv2.boxPoints(ret)
			pts = np.int0(pts)
			
			currWithoutLines = frame.copy()
			img2 = cv2.polylines(frame,[pts],True, 255,2)
			
			pts = np.array(pts)
			
			s = pts.sum(axis = 1)
			tl = pts[np.argmin(s)]
			br = pts[np.argmax(s)]
			
			diff = np.diff(pts,axis=1)
			tr = pts[np.argmin(diff)]
			bl = pts[np.argmax(diff)]
			
			
			if tl[0] < 0:
				tl[0] = 0
			if tl[1] < 0:
				tl[1] = 0
			
			camshiftROIMask = backProj[tl[1]:br[1],tl[0]:br[0]]
			count2 = cv2.countNonZero(camshiftROIMask)
			if camshiftROIMask.size == 0 :
				perB = 0
			else:
				perB = count2/camshiftROIMask.size
				lastFrameROI = frame[tl[1]:br[1],tl[0]:br[0]]
				lastFrameROIhsv = cv2.cvtColor(lastFrameROI,cv2.COLOR_BGR2HSV)
				lastFrameROIhsv = cv2.GaussianBlur(lastFrameROIhsv,(1,1),0)
				
				mask = cv2.inRange(lastFrameROIhsv,np.array((0.,60.,32.)),np.array((255.,255.,180.)))
				
				roi_hist = cv2.calcHist([lastFrameROIhsv],[0],mask,[180],[0,180])
				roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
				
				backProj = cv2.calcBackProject([lastFrameROIhsv],[0],roi_hist,[0,180],1)
				retval, threshold = cv2.threshold(backProj, 76, 255, cv2.THRESH_BINARY)
				backProj = cv2.normalize(threshold,threshold,0,255,cv2.NORM_MINMAX)
				
				
			perB = perB *100
			# print "Difference:"
			diff = perA-perB
			# print diff
			
			#for blurring out ROI
			curr = img2.copy()
			
			img2 = cv2.blur(img2[tl[1]:br[1],tl[0]:br[0]],(91,91),4)
			
			# img2 = cv2.GaussianBlur(img2[tl[1]:br[1],tl[0]:br[0]],(21,21),4)
			curr[tl[1]:br[1],tl[0]:br[0]] = img2
			
			
			if diff < 30 and  diff > -30:
				cv2.imshow('img2',curr)
				p = cv2.waitKey(60) & 0xFF
				if p == ord('q'):
					breakStatus = False
					break
			else:
				cv2.imshow('img2',currWithoutLines)
				p = cv2.waitKey(60) & 0xFF
				if p == ord('q'):
					breakStatus = True
					break
				
		else:
			cv2.destroyAllWindows()
			cap.release()
			
				# ------------- camShift Iteration function End ---------

# --------- objSelect function Start ---------

def objSelect(event,x,y,flags,param):

	global originalCap, ROI, roiPts, roiBox
	if event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 2:
		
		# draw point and display
		roiPts.append((x,y))
		cv2.circle(originalCap,(x,y),4,(0,255,0),2)
		cv2.imshow('Captured',originalCap)

	if event == cv2.EVENT_MOUSEMOVE and len(roiPts) == 2:
		
		#Crop Selected ROI and finding 1st, 2nd, 3rd & 4th Points of ROI
		cv2.destroyWindow('Captured')
		
		roiPts = np.array(roiPts)
		
		s = roiPts.sum(axis = 1)
		tl = roiPts[np.argmin(s)]
		br = roiPts[np.argmax(s)]

		diff = np.diff(roiPts,axis=1)
		tr = roiPts[np.argmin(diff)]
		bl = roiPts[np.argmax(diff)]
				
		first = (tl[0],tl[1])
		second = (tr[0],tr[1])
		third = (br[0],br[1])
		fourth = (bl[0],bl[1])

		cv2.rectangle(originalCap,first,third,(0,255,0),2)
		
		ROI = originalCap[tl[1]:br[1],tl[0]:br[0]]

		#cv2.imwrite('ROISaved.jpg',ROI)
		
		roiBox = (tl[0],tl[1],br[0],br[1])    

	# -------- objSelect Function End -----------------


# --------- Code Start (Entry Point) ---------

#cap = cv2.VideoCapture('E:/Taha/Videos/trackerTest.mp4')
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
	ret,frame = cap.read()
	cv2.imshow('Frame',frame)
	k = cv2.waitKey(60) & 0xFF
	if k == ord('q'):
		break
	elif k == ord('c'):
		originalCap = None
		originalCap = frame.copy()
		cv2.imshow('Captured',originalCap)
		roiPts=[]
		cv2.setMouseCallback('Captured',objSelect)

cv2.destroyWindow('Frame')

try:
	if len(roiPts) == 2:
		hsv_roi = cv2.cvtColor(ROI,cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((255.,255.,180.)))
		hsv_roi = cv2.GaussianBlur(hsv_roi,(1,1),0)

		roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
		roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

		backProj = cv2.calcBackProject([hsv_roi],[0],roi_hist,[0,180],1)
		retval, threshold = cv2.threshold(backProj, 76, 255, cv2.THRESH_BINARY)
		backProj = cv2.normalize(threshold,threshold,0,255,cv2.NORM_MINMAX)
		
		countA = cv2.countNonZero(backProj)
		perA = countA/backProj.size
		perA = perA * 100
		          
		term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT ,10,1)
		print roiPts
		func(roi_hist,roiBox,term_crit)
		
	
except NameError:
	print 'Error'
try:
	if breakStatus == True:
		func(roi_hist,roiBox,term_crit)
	else:
		cv2.destroyAllWindows()
		
	
except:
	print ''

cap.release()

