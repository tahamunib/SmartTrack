from __future__ import division
import cv2
import numpy as np
import json,sys

# -------- camShift iteration function start --------------

def func(roi_histogram,roiBox,term_criteria):
	global breakStatus,perA,mask
	fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
	out = cv2.VideoWriter(data['outputFile'],fourcc, 25, (640,480))

	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	startSec = data['startSec']
	endSec = data['endSec']
	print startSec
	print endSec
	
	totalSec = endSec
	print totalSec
	cap.set(0,startSec*1000)
	
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
			
			diffStartRow = 0
			diffEndRow = 0
			diffStartCol = 0
			diffEndCol = 0
			if tl[1]>tr[1]:
				diffStartRow = tl[1]-tr[1] 
			
			if bl[1] > br[1]:
				diffEndRow = bl[1]-br[1]
			
			
			#for blurring out ROI
			curr = img2.copy()
			img2 = cv2.GaussianBlur(img2[tl[1]-diffStartRow:br[1]+diffEndRow,tl[0]:br[0]],(21,21),4)
			curr[tl[1]-diffStartRow:br[1]+diffEndRow,tl[0]:br[0]] = img2
			
			currPos = int(cap.get(0)/1000)
			print currPos
			print totalSec
			if diff < 30 and  diff > -30:
				if currPos == totalSec:
					breakStatus = False
					out.release()
					break
				else:
					out.write(curr)
			else:
				if currPos == totalSec:
					breakStatus = False
					out.release()
					break
				else:
					out.write(currWithoutLines)
				
		else:
			cv2.destroyAllWindows()
			cap.release()
			out.release()
			
				# ------------- camShift Iteration function End ---------


# --------- Code Start (Entry Point) ---------

#cap = cv2.VideoCapture(0)


##fourcc = cv2.CV_FOURCC('m', 'p', '4', 'v')
# print sys.argv[1]
data = sys.argv[1]
# data = data.replace("\"",'\\"')
# data = data.replace("{",'\"{')
# data = data.replace("}",'}\"')


# data=json.dumps(data)

# print data
data = json.loads(data)
if len(data) == 5:

	cap = cv2.VideoCapture(data['inputFile'])
	ret,frame = cap.read()
	if ret == True:
		
		if len(data['box']['TL']) == 2 and len(data['box']['BR']) == 2 :
			topLeft = (data['box']['TL'][0],data['box']['TL'][1])
			bottomRight = (data['box']['BR'][0],data['box']['BR'][1])
		else:
			sys.exit('Invalid box points')

		originalCap = None
		originalCap = frame.copy()
				
		roiPts=[]
		
		roiPts.append(topLeft)
		roiPts.append(bottomRight)

		ROI = originalCap[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
		roiBox = (topLeft[0],topLeft[1],bottomRight[0],bottomRight[1])

		cv2.circle(originalCap,roiPts[0],4,(0,255,0),2)
		cv2.circle(originalCap,roiPts[1],4,(0,255,0),2)

		
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
			func(roi_hist,roiBox,term_crit)
		
		
	else:
		cap.release()
		sys.exit('Cannot open the given file!!')
	try:
		if breakStatus == True:
			func(roi_hist,roiBox,term_crit)
	except:
		print ''
	
cap.release()


