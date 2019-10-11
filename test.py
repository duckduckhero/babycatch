# imports
from __future__ import print_function 
from imutils.video import VideoStream
from scipy.spatial import distance as dist 
from imutils.video import FPS
from imutils import paths 
from pygame import mixer 
from ar_markers import detect_markers
from imutils import perspective
import argparse
import imutils
import math 
import os,sys,time
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pandas as pd
import time 
from sklearn.cluster import KMeans


def lineLine(x1, y1, x2, y2, x3, y3, x4, y4): 
    uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    if (uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1) : return True 
    else: return False 

def lineRect(x1, y1, x2, y2, rx, ry, rw, rh):
    left = lineLine(x1,y1,x2,y2, rx,ry,rx, ry+rh)
    right = lineLine(x1,y1,x2,y2, rx+rw,ry, rx+rw,ry+rh)
    top = lineLine(x1,y1,x2,y2, rx,ry, rx+rw,ry)
    bottom = lineLine(x1,y1,x2,y2, rx,ry+rh, rx+rw,ry+rh)

    if (left or right or top or bottom) : return True
    else : return False


def length(x1, y1, x2, y2):
    return (x1-x2)**2+(y1-y2)**2

def abs(a, b):
    if a-b<0:
        return b-a 
    else : 
        return a-b 

 
def order_points(pts):
    # sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")



ap = argparse.ArgumentParser()
"""
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
"""
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('ssd.prototxt.txt', 'ssd.caffemodel')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
mixer.init()
mixer.music.load('instruction1.mp3') 
mixer.music.play()
# init camera
#camera = cv2.VideoCapture(0) ### <<<=== SET THE CORRECT CAMERA NUMBER
#camera.set(3,1280)             # set frame width
#camera.set(4,720)              # set frame height
#time.sleep(0.5)
#print(camera.get(3),camera.get(4))

#plt.axis([0, 1000, 0, 1000])

# master frame
master = None
centx=-1
centy=-1
intersectcnt=0
intersectfrm=0
startX=0
startY=0
endX=0
endY=0
modeset=0
points =[]
tempx=-1
tempy=-1
cnt3=0
flag=0 #0 means does not exist / 1 means exist

sideAcnt = 0 #number of contours that intersect with side A
sideBcnt = 0 
sideCcnt = 0 
sideDcnt = 0 
consetime = 0 

prevTime = 0 

intside = 'K'
formerside = 'X'

while 1: 
    frame_captured, frame = vs.read() 
    curTime = time.time() 
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    frame = imutils.resize(frame, width=800)
    markers = detect_markers(frame)
    for marker in markers:
        marker.highlite_marker(frame)
        print(marker.center)
        if tempx>=0: 
            if abs(marker.center[0],tempx)<=5 and abs(marker.center[1], tempy)<=5:
                if(cnt3==40):
                    flag=0
                    for point in points: 
                        if (abs(point[0], marker.center[0])<=5 and abs(point[1], marker.center[1])<=5): flag=1
                    if flag==0:
                        mixer.music.load('instruction2.mp3')
                        mixer.music.play()
                        points.append([tempx, tempy])
                        cnt3=0
                        tempx=-1
                        tempy=-1  
                cnt3+=1
        tempx = marker.center[0]
        tempy = marker.center[1]
        print(marker.center)
        print(points)
    for point in points:
        cv2.circle(frame,(point[0],point[1]),2,(0,0,255),2)
    cv2.imshow('Test Frame', frame)
    
    if len(points)==4: 
        mixer.music.load('instruction3.mp3')
        mixer.music.play()
        break 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
# process of filtering the dots 

points = [[20, 30], [40, 20], [15, 300], [80, 40]]
points = np.array(points, dtype="int32")
res = perspective.order_points(points)
res = res.reshape((-1, 1, 2))

#dots filtering ends above 

while 1:
     
    if intersectfrm!=0 : intersectfrm+=1
    df = pd.DataFrame(columns=['x','y'])
    
    midpoints=[]
    # grab a frame
    (grabbed,frame0) = vs.read()
    
    #markers = detect_markers(frame0)

    #for marker in markers:
    #    marker.highlite_marker(frame)


    frame0 = imutils.resize(frame0, width=1000)
    framek = frame0.copy() 
    framek = imutils.resize(framek, width=800)
 

    (h, w) = framek.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(framek, (300, 300)),0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()


    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            if CLASSES[idx]=='person':
                cv2.rectangle(framek, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(framek, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    # end of feed
    if not grabbed:
        break

    # gray frame
    frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
    
    # blur frame
    frame2 = cv2.GaussianBlur(frame1,(15,15),0)

    # initialize master
    if master is None:
        master = frame2
        continue

    # delta frame
    frame3 = cv2.absdiff(master,frame2)

    # threshold frame
    frame4 = cv2.threshold(frame3,15,255,cv2.THRESH_BINARY)[1]
    
    val1=frame1.shape[0]
    val2=frame1.shape[1]
    if val1>val2:large=val1
    else : large=val2
    #plt.axis([0, large, 0, large])

    # dilate the thresholded image to fill in holes
    kernel = np.ones((2,2),np.uint8)
    frame5 = cv2.erode(frame4,kernel,iterations=4)
    frame5 = cv2.dilate(frame5,kernel,iterations=8)

    # find contours on thresholded image
    nada,contours,nada = cv2.findContours(frame5.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # make coutour frame
    frame6 = frame0.copy()

    # target contours
    targets = []
    
    xmin=large+1
    xmax=-1
    ymin=large+1
    ymax=-1

    tmpx = centx
    tmpy = centy 

    
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            for k in range(len(contours[i][j])):
                if contours[i][j][k][0]>xmax:xmax=contours[i][j][k][0]
                if contours[i][j][k][0]<xmin:xmin=contours[i][j][k][0]
                if contours[i][j][k][1]>ymax:ymax=contours[i][j][k][1]
                if contours[i][j][k][1]<ymin:ymin=contours[i][j][k][1]
                #print(contours[i][j][k][0], contours[i][j][k][1])

    centx = (startX+endX)/2
    centy = (startY+endY)/2
    #plt.scatter(centx, centy)
    #plt.pause(0.05)
    #print("이번 centx : "+str(centx))
    #print("이번 centy : "+str(centy))
    #print("이전 centx : "+str(tmpx))
    #print("이전 centy : "+str(tmpy))
    
    #if centx > tmpx : print("오른쪽으로 이동중")
    #elif centx < tmpx : print("왼쪽으로 이동중")
    
    index=0
    frame10 = frame0.copy()
    frame11 = frame0.copy()
    # loop over the contours
    for c in contours:

        if cv2.contourArea(c) < 20: 
            continue 
        # contour data
        M = cv2.moments(c)#;print( M )
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(c) 

        # if the contour is too small, ignore i
        rx = x+int(w/2)
        ry = y+int(h/2)
        ca = cv2.contourArea(c)
        midpoints.append((cx, cy))
        if ((startX<=cx and cx<=endX) and (startY<=cy and cy<=endY)):
            df.loc[index] = [cx, cy]
            index+=1
        # plot contours
        cv2.drawContours(framek,[c],0,(0,0,255),2)
        cv2.rectangle(framek,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(framek,(cx,cy),2,(0,0,255),2)
        cv2.circle(framek,(rx,ry),2,(0,255,0),2)
        cv2.rectangle(frame10, (xmin, ymin), (xmax, ymax), (0, 255,0), 2)
        cv2.circle(frame10, ((xmin+xmax)//2, (ymin+ymax)//2), 2, (0, 255, 0), 2)
        ##epsilon2 = 0.1*cv2.arcLength(c, True)
        # save target contours
        targets.append((cx,cy,ca))
        if lineRect(points[0][0], points[0][1], points[1][0], points[1][1], x,y,w,h):
            sideAcnt+=1 
            print("A intersect")
        if lineRect(points[1][0], points[1][1], points[2][0], points[2][1], x, y, w,h):
            sideBcnt+=1 
            print("B intersect")
        if lineRect(points[2][0], points[2][1], points[3][0], points[3][1], x, y, w,h):
            sideCcnt+=1 
            print("C intersect")
        if lineRect(points[3][0], points[3][1], points[0][0], points[0][1], x, y, w,h):
            sideDcnt+=1 
            print("D intersect")
        
    ###########################################################################################################################################################       
    if sideAcnt >=5:
        intside = 'A'
    if sideBcnt >=5: 
        intside = 'B'
    if sideCcnt >=5: 
        intside = 'C'
    if sideDcnt >=5: 
        intside = 'D'
    elif sideAcnt+sideBcnt+sideCcnt+sideDcnt<20: #5개 넘는 모서리가 없어도 움직임 중단했다고 판단하고 정지시킨다 
        consetime=0
        sideAcnt=0
        sideBcnt=0
        sideCcnt=0
        sideDcnt=0

    if intside == formerside : consetime+=1 #연속되서 계속 그 모서리에서 접촉 발생 
    if intside != formerside : #연결성 끊겼음 - 계속 세던 것 정지시킴 
        consetime=0
        sideAcnt=0
        sideBcnt=0
        sideCcnt=0
        sideDcnt=0

    if consetime>=390 : #1초에 30프레임이기 때문에 13초면 390프레임 쓴다 
        print("Baby is escaping!") 
    if sideAcnt>=5 or sideBcnt>=5 or sideCcnt>=5 or sideDcnt>=5 : formerside = intside #연결성을 확인하기 위해서 formerside 에 이번 프레임에 움직인 모서리가 어딘지를 저장한다

    ########################################################################################################################################################## 

    # make target
    mx = 0
    my = 0
    data_points = df.values
    
    if len(data_points)>=10:
        kmeans = KMeans(n_clusters=5).fit(data_points)
        kmeans.cluster_centers_
        for i in range(len(kmeans.cluster_centers_)):
            cv2.circle(framek, (int(kmeans.cluster_centers_[i][0]), int(kmeans.cluster_centers_[i][1])), 2, (0,255,0),10)
            
        

    #print(midpoints)
    
    if targets:
        
        # average centroid adjusted for contour size
        #area = 0
        #for x,y,a in targets:
        #    mx += x*a
        #    my += y*a
        #    area += a
        #mx = int(round(mx/area,0))
        #my = int(round(my/area,0))

        # centroid of largest contour
        area = 0
        for x,y,a in targets:
            if a > area:
                mx = x
                my = y
                area = a

    # plot target
    tr = 50
    frame7 = frame0.copy()
    framek = cv2.polylines(framek, np.int32([res]), True, (255, 0, 255))
    if targets:
        cv2.circle(frame7,(mx,my),tr,(0,0,255,0),2)
        cv2.line(frame7,(mx-tr,my),(mx+tr,my),(0,0,255,0),2)
        cv2.line(frame7,(mx,my-tr),(mx,my+tr),(0,0,255,0),2)
    
    # update master
    master = frame2
    cv2.putText(framek, str, (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    # display
    #cv2.imshow("Frame0: Raw",frame0)
    #cv2.imshow("Frame1: Gray",frame1)
    #cv2.imshow("Frame2: Blur",frame2)
    #cv2.imshow("Frame3: Delta",frame3)
    #cv2.imshow("Frame4: Threshold",frame4)
    #cv2.imshow("Frame5: Dialated",frame5)
    #cv2.imshow("Frame6: Contours",frame6)
    #cv2.imshow("Frame7: Target",frame7)
    #cv2.imshow("Frame10: Movement", frame10)
    cv2.imshow("Frame11: Body Parts+Human", framek)
    #cv2.imshow("Frame12: Human Detection", framek)

    # key delay and action
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('key:',[chr(key)])

# release camera
vs.release()


# close all windows
cv2.destroyAllWindows()