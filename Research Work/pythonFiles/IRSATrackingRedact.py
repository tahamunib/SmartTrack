from __future__ import division
import cv2
import numpy as np
import json,sys,os
import subprocess as sp

# -------- camShift iteration function start --------------

def validateInput(data):
	validate = None
	if os.path.exists(data['inputFile']):
		inputFile = data['inputFile']
		try:
			startSec = int(data['startSec'])
			try:
				endSec = int(data['endSec'])
				if os.path.exists(os.path.split(data['outputFile'])[0]):
					outputFile = data['outputFile']
					if len(data['box']) == 2 and len(data['box']['TL']) == 2 and len(data['box']['BR']) == 2:
						try:
							topLeft = (int(data['box']['TL'][0]),int(data['box']['TL'][1]))
							bottomRight = (int(data['box']['BR'][0]),int(data['box']['BR'][1]))
							try:
								trimFlag = data['trimVideo'].lower()
								if trimFlag == 'true' or trimFlag == 'false':
									try:
										blurValue = int(data['blurValue'])
										if blurValue > 0:
											if len(data)==10:
												try:
													translucency = int(data['translucency'])
													if translucency >= 1 and translucency <= 10:
														try:
															isImageOverlay = data['isImageOverlay'].lower()
															if isImageOverlay == 'true':
																if os.path.exists(data['overlay']):
																	validate = (True,'data')
																	return validate
																else:
																	validate = (False,'imagePath')
																	return validate
															elif isImageOverlay == 'false':
																if data['overlay']:
																	validate = (True,'data')
																	return validate
																else:
																	validate = (False,'overlayText')
																	return validate
															else:
																validate(False,'isImageOverlay')
																return validate
														except:
															validate(False,'isImageOverlay')
															return validate
													else:
														validate = (False,'translucency')
														return validate
												except:
													validate = (False,'translucency')
													return validate
											elif len(data)==7:
												validate = (True,'data')
												return validate
											elif len(data)==8 or len(data)==9 or len(data) > 10:
												validate = (False,'Missing Required Inputs')
												return validate
										else:
											validate = (False,'blurValue')
									except:
										validate = (False,'blurValue')
								else:
									validate= (False,'trimVideo')
									return validate
							except:
								validate= (False,'trimVideo')
								return validate
						except:
							validate = (False,'box')
							return validate
					else:
						validate = (False,'box')
						return validate
				else:
					validate = (False,'outputFile')
					return validate
			except ValueError:
				validate = (False,'endSec')
				return validate
		except ValueError:
			validate = (False,'startSec')
			return validate
	else:
		validate = (False,'inputFile')
		return validate



def resize(image,width,height):
	resized = cv2.resize(image,(width,int(height)))
	return resized

def func(roi_histogram,roiBox,term_criteria):
	global breakStatus,perA,mask,startSec,endSec,blurValue,trimVideo,outputFile,isImageOverlay,overlay,translucency 
	# fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
	

	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	totalSec = int(length/cap.get(5))
	
	if startSec < 0: 				# If Start or End is negative or End is beyond the total then full video should be redacted. 
		startSec = 0
	if endSec < 0 or endSec > totalSec:
		endSec = totalSec
	if endSec < startSec or startSec >= endSec:							# If End is less than Start or Start is greater than End then break
		cap.release()
		sys.exit('Invalid Start or End Limit, Exiting..')
	if blurValue < 0 or blurValue > 100:								# Blur value should be in 0-100 inclusive.
		cap.release()
		sys.exit('Invalid blurValue, must be between 0-100, Exiting..')
	
	dimension = '{}x{}'.format(width, height)
	f_format = 'bgr24'
	fps = str(cap.get(5))
	
	if trimVideo == "true":
		cap.set(0,startSec*1000)
	
	command = ['ffmpeg',
		'-hide_banner',
		'-loglevel','panic',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', dimension,
        '-pix_fmt', 'bgr24',
        '-r',fps,
        '-i', '-',
        '-an',
        '-vcodec', 'h264',
		'-pix_fmt','yuv420p',
		outputFile ]
	proc = sp.Popen(command, stdin=sp.PIPE)
	try:
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
				# img2 = cv2.polylines(frame,[pts],True, 255,2)
				img2 = frame
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
				img2 = cv2.blur(img2[tl[1]-diffStartRow:br[1]+diffEndRow,tl[0]:br[0]],(blurValue,blurValue),4)
				if img2 is not None:
					h,w,c = img2.shape
					y = int(h/2)
					x = int(w/2)
					if overlay:
						if isImageOverlay == 'true':
							logo = cv2.imread(overlay)
							hl,wl,cl = logo.shape
							overlayImg = cv2.resize(logo,(w,h))
							hl,wl,cl = overlayImg.shape
							if w >= wl:
								overlayObjCenterX = int(wl/2)
								overlayObjCenterY = int(hl/2)
								#overlay = img2[y-overlayObjCenterY:y+overlayObjCenterY+1,x-overlayObjCenterX:x+overlayObjCenterX+1]
								#overlay = img2.copy()
								#print translucency
								cv2.addWeighted(img2,translucency,overlayImg,1-translucency,0,img2)
								#img2[y-overlayObjCenterY:y+overlayObjCenterY+1,x-overlayObjCenterX:x+overlayObjCenterX+1] = overlay
								# img2 = overlay
						else:
							font = cv2.FONT_HERSHEY_PLAIN
							size = cv2.getTextSize(overlay,font,2,2)
							if w >= int(size[0][0]):
								overlayObjCenterX = int(size[0][0]/2)
								overlayObjCenterY = int(size[0][1]/2)
								baseline = int(size[1])
								overlayImg = img2.copy()
								text = overlayImg[(y-overlayObjCenterY)-(baseline+1):y+overlayObjCenterY,x-overlayObjCenterX:x+overlayObjCenterX]
								#print overlay
								#print translucency
								h1,w1,c1 = text.shape
								cv2.putText(text,overlay,(0,(h1-baseline)+5),font,2,(0,0,255),2)
								if(w > 0 and h1 > 0):
									text = cv2.resize(text,(w,h1))
									# cv2.imshow("overlayImg",overlayImg)
									# cv2.waitKey(0)
									overlayImg[(y-overlayObjCenterY)-(baseline+1):y+overlayObjCenterY,0:w] = text
									cv2.addWeighted(img2,translucency,overlayImg,1-translucency,0,img2)
								else:
									overlayImg[(y-overlayObjCenterY)-(baseline+1):y+overlayObjCenterY,x-overlayObjCenterX:x+overlayObjCenterX] = text
									cv2.addWeighted(img2,translucency,overlayImg,1-translucency,0,img2)
						
					
					
				curr[tl[1]-diffStartRow:br[1]+diffEndRow,tl[0]:br[0]] = img2			
				currPos = cap.get(0)/1000
				
				
				
				#print diff
				if diff < 30 and  diff > -30:
					if startSec < currPos and currPos < endSec+1:			
						try:
							#cv2.imshow("debugWithBlur",curr)
							#cv2.waitKey(25)
							proc.stdin.write(curr.tostring())										# Current position lies in between Start and End
						except:
							sys.exit('Error writing frame, Exiting..')
					else:
						if trimVideo == "false":								
							try:
								#cv2.imshow("debugWithoutBlur",currWithoutLines)
								#cv2.waitKey(25)
								proc.stdin.write(currWithoutLines.tostring())						# Full Video write with redaction only of given boundaries
							except:
								sys.exit('Error writing frame, Exiting..')
						else:												# Trim Video with redaction
							if startSec-1 >= currPos:						
								continue									# Current position is before the starting point of redaction
							else:											
								breakStatus = False
								proc.stdin.close()
								break										# Current Position exceeded ending point of redaction, hence exit.
				else:
					if startSec-1 < currPos and currPos < endSec+1:
						try:
							proc.stdin.write(currWithoutLines.tostring())
						except:
							sys.exit('Error writing frame, Exiting..')
					else:
						if trimVideo == "false":
							try:
								#cv2.imshow("debug",currWithoutLines)
								#cv2.waitKey(25)
								proc.stdin.write(currWithoutLines.tostring())
							except:
								sys.exit('Error writing frame, Exiting..')
						else:
							if startSec-1 >= currPos:
								continue
							else:
								breakStatus = False
								proc.stdin.close()
								break
					
			else:
				cap.release()
				proc.stdin.close()
				sys.exit()
	except:	
		cap.release()
		proc.stdin.close()
		sys.exit()
			
				# ------------- camShift Iteration function End ---------


# --------- Code Start (Entry Point) ---------

try:
	data = sys.argv[1]
	data = json.loads(data)
except:
	sys.exit('Invalid Input, Exiting..')
if len(data) >= 7:
	validate = validateInput(data)
	if validate[0]:
		inputFile = data['inputFile']
		outputFile = data['outputFile']
		startSec = data['startSec']
		endSec = data['endSec']
		trimVideo = data['trimVideo'].lower()
		blurValue = data['blurValue']
		overlay = False
		if len(data) == 10:
			translucency = data['translucency']/10
			isImageOverlay = data['isImageOverlay'].lower()
			overlay = data['overlay']
			overlay = overlay.replace('^',' ')
		
		cap = cv2.VideoCapture(inputFile)
		ret,frame = cap.read()
		if ret == True:
			topLeft = (int(data['box']['TL'][0]),int(data['box']['TL'][1]))
			bottomRight = (int(data['box']['BR'][0]),int(data['box']['BR'][1]))
			
			height,width,ch = frame.shape
			originalCap = None
			originalCap = frame.copy()
					
			roiPts=[]
			
			roiPts.append(topLeft)
			roiPts.append(bottomRight)

			ROI = originalCap[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
			roiBox = (topLeft[0],topLeft[1],bottomRight[0],bottomRight[1])

			# cv2.circle(originalCap,roiPts[0],4,(0,255,0),2)
			# cv2.circle(originalCap,roiPts[1],4,(0,255,0),2)

			
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
	else:
		sys.exit('Invalid Input: '+ validate[1])
else:
	cap.release()
	sys.exit('Required Inputs Missing, Exiting...')
	
cap.release()
sys.exit()


