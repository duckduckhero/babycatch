# imports
import os,sys,time
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.cluster import KMeans 


# init camera
camera = cv2.VideoCapture('Video2.mp4') ### <<<=== SET THE CORRECT CAMERA NUMBER
#camera.set(3,1280)             # set frame width
#camera.set(4,720)              # set frame height
time.sleep(0.5)
print(camera.get(3),camera.get(4))

plt.axis([0, 1000, 0, 1000])

# master frame
master = None
centx=-1
centy=-1


while 1:

    df = pd.DataFrame(columns=['x','y'])
    
    midpoints=[]
    # grab a frame
    (grabbed,frame0) = camera.read()
    
    
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
    plt.axis([0, large, 0, large])

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

    centx = (xmin+xmax)/2
    centy = (ymin+ymax)/2
    plt.scatter(centx, centy)
    plt.pause(0.05)
    print("이번 centx : "+str(centx))
    print("이번 centy : "+str(centy))
    print("이전 centx : "+str(tmpx))
    print("이전 centy : "+str(tmpy))
    
    if centx > tmpx : print("오른쪽으로 이동중")
    elif centx < tmpx : print("왼쪽으로 이동중")
    
    index=0
    frame10 = frame0.copy()
    frame11 = frame0.copy()
    # loop over the contours
    for c in contours:
        
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 25:
            continue

        # contour data
        M = cv2.moments(c)#;print( M )
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(c)
        rx = x+int(w/2)
        ry = y+int(h/2)
        ca = cv2.contourArea(c)
        midpoints.append((cx, cy))
        df.loc[index] = [cx, cy]
        index+=1
        # plot contours
        cv2.drawContours(frame6,[c],0,(0,0,255),2)
        cv2.rectangle(frame6,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(frame6,(cx,cy),2,(0,0,255),2)
        cv2.circle(frame6,(rx,ry),2,(0,255,0),2)
        cv2.rectangle(frame10, (xmin, ymin), (xmax, ymax), (0, 255,0), 2)
        cv2.circle(frame10, ((xmin+xmax)//2, (ymin+ymax)//2), 2, (0, 255, 0), 2)
        ##epsilon2 = 0.1*cv2.arcLength(c, True)
        # save target contours
        targets.append((cx,cy,ca))
    
    # make target
    mx = 0
    my = 0
    data_points = df.values
    
    if len(data_points)>=10:
        kmeans = KMeans(n_clusters=5).fit(data_points)
        kmeans.cluster_centers_
        for i in range(len(kmeans.cluster_centers_)):
            cv2.circle(frame11, (int(kmeans.cluster_centers_[i][0]), int(kmeans.cluster_centers_[i][1])), 2, (0,255,0),10)
            
        

    print(midpoints)
    
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
    if targets:
        cv2.circle(frame7,(mx,my),tr,(0,0,255,0),2)
        cv2.line(frame7,(mx-tr,my),(mx+tr,my),(0,0,255,0),2)
        cv2.line(frame7,(mx,my-tr),(mx,my+tr),(0,0,255,0),2)
    
    # update master
    master = frame2

    # display
    #cv2.imshow("Frame0: Raw",frame0)
    #cv2.imshow("Frame1: Gray",frame1)
    #cv2.imshow("Frame2: Blur",frame2)
    #cv2.imshow("Frame3: Delta",frame3)
    #cv2.imshow("Frame4: Threshold",frame4)
    #cv2.imshow("Frame5: Dialated",frame5)
    cv2.imshow("Frame6: Contours",frame6)
    #cv2.imshow("Frame7: Target",frame7)
    cv2.imshow("Frame10: Movement", frame10)
    cv2.imshow("Frame11: Body Parts", frame11)

    # key delay and action
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('key:',[chr(key)])

# release camera
camera.release()


# close all windows
cv2.destroyAllWindows()