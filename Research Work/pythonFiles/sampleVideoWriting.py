import numpy as np
import cv2
import subprocess as sp

cap = cv2.VideoCapture(0)

input_file = 'input_file_name.mp4'
output_file = 'C:\Python27\output_file_name.mp4'

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
ret,frame = cap.read()
height,width,ch = frame.shape

# out = cv2.VideoWriter('New.mp4',fourcc,20.0,(640,480))

ffmpeg = 'FFMPEG'
dimension = '{}x{}'.format(width, height)
f_format = 'bgr24' # remember OpenCV uses bgr format
fps = str(cap.get(5))

command = ['ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', dimension,
        '-pix_fmt', 'bgr24',
        
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '648k',
        output_file ]

proc = sp.Popen(command, stdin=sp.PIPE)
    
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==True:
        frame=cv2.flip(frame,1)
        proc.stdin.write(frame.tostring())
        
        
    else:
        break
# When everything done, release the capture
cap.release()
# out.release()
cv2.destroyAllWindows()
proc.stdin.close()

