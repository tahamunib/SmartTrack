import subprocess as sp
import cv2
import numpy

cap = cv2.VideoCapture('C:\Python27\d1.mp4')
ret,frame = cap.read()
height,width,ch = frame.shape

command = [ 'ffmpeg',
            '-i', 'C:\Python27\d1.mp4',
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

while True:
        
	# read 420*360*3 bytes (= 1 frame)
	raw_image = pipe.stdout.read(width*height*ch)
	# transform the byte read into a numpy array
	image =  numpy.fromstring(raw_image, dtype='uint8')
	image = image.reshape((480,640,3))
	cv2.imshow('Image',image)

	k=cv2.waitKey(25)
	if k==ord('q'):
		break

# throw away the data in the pipe's buffer.
cap.release()
cv2.destroyAllWindows()
pipe.stdout.flush()
pipe.terminate()
