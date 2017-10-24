import numpy as np
import cv2
from sys import argv
import matplotlib.pyplot as plt

script = argv

cap = cv2.VideoCapture('tracking_video.mp4')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]

hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert image color from BGR to HSV

mask = cv2.inRange(hsv_roi, np.array((9.,100.,100.)), np.array((29.,255.,255.)))

#plot histogram of the image based on the ditribution of the color orange and min max normalize the  
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1 )

#check for end of video if not continue
while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)    #caluclate the relative histogram of the image object color

        # apply meanshift to get the new location of centroids in each frame
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        
    
        
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()