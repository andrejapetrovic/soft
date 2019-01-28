import cv2
import numpy as np
import os

boxes = {}
sum = 0

def findLine(lowerColor, upperColor, frame):       
    mask = cv2.inRange(frame, lowerColor, upperColor)
    image = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1], 100, 250)

    blur = cv2.GaussianBlur(edges, (5, 5), 0)
    lines = cv2.HoughLinesP(blur, 1, np.pi/180, 100, 175, 200)
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            #cv2.line(frame, (x1,y1), (x2,y2), (255,255,0),2)
            return [x1, y1, x2, y2]
    
def playVideo(path):
    video = cv2.VideoCapture(path)
    
    sum = 0 
    boxes.clear()

    boxId = 0
    while True:
        ret, frame = video.read()

        #plava linija - koordinate [x1, y1, x2, y2]
        blue = findLine(np.array([120, 0, 0]), np.array([255, 100, 100]), frame)
        #zelena
        green = findLine(np.array([0, 130, 0]), np.array([59, 255, 50]), frame)

        #konture  
        mask = cv2.inRange(frame, np.array([160, 160, 160]), np.array([255, 255, 255]))
        image = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, contours, -1, (255,0,0), 2)
        for contour in contours:
            currentKey = -1;
            x, y, w, h = cv2.boundingRect(contour)
            if(w>9 or h>8) and (w<25 and h<25):
                #geometrijski centar
                cx = x+w*0.5
                cy = y+h*0.5
                dists = {}
                
                if not boxes:
                    boxId += 1
                    #[id] = [x, y, sirina, visina, centar-x, centar-y, presekao plavu, presekao zelenu]
                    boxes[boxId] = [x, y, w, h, cx, cy, 0, 0]
                    currentKey = boxId;
                else:
                    for key in boxes:
                        box = boxes[key]
                        dists[key] = (np.linalg.norm(np.array((cx, cy))-np.array((box[4], box[5]))))
                
                    minDist = min(dists.values())
                        
                    if (minDist < 20 and minDist>5):
                        k = (list(dists.keys())[list(dists.values()).index(minDist)])
                        boxes[k] = [x, y, w, h, x+w*0.5, y+h*0.5, boxes[k][6], boxes[k][7]]
                        currentKey = k;
                    if (minDist >= 20):
                        boxId += 1
                        boxes[boxId] = [x, y, w, h, cx, cy, 0, 0]

            if currentKey is not -1:
                cv2.rectangle(frame,(x, y),(x+w,y+h),(0,0,255),1)
                cv2.putText(frame,"id:" + str(currentKey), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))                        
                if blue is not None:
                    checkIntersection(blue, currentKey, frame, "blue")
                if green is not None:
                    checkIntersection(green, currentKey, frame, "green")      

        cv2.imshow(path, frame)
        #cv2.imshow("mask", mask)
        key = cv2.waitKey(1)
        if key == 27:
            break
            
    video.release()
    cv2.destroyAllWindows()

def checkIntersection(line, key, frame, lineColor):
    x, y, w, h, cx, cy, passedBlue, passedGreen = boxes[key]
    #proverava se presek svih stranica kvadrata s kojim je uokvirena kontura
    #sa linijom, pocevsi od gornje stranice u smeru kazaljke na satu
    if lineLineIntersection([x, y, x+w, y], line) or lineLineIntersection([x+w, y, x+w, y+h], line) or lineLineIntersection([x, y+h, x+w, y+h], line) or lineLineIntersection([x, y, x, y+h], line):
        if lineColor is "blue" and passedBlue == 0:
            boxes[key][6] = 1
            cv2.putText(frame,"PLAVA", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        if lineColor is "green" and passedGreen == 0:
            boxes[key][7] = 1
            cv2.putText(frame,"ZELENA", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))



def lineLineIntersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2 
    a = (x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)
    b = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    k = a/b
    if k>=0.0 and k<=1.0:
        xp = x1 + k*(x2 - x1)
        yp = y1 + k*(y2 - y1)
        if xp>=min([x3,x4]) and xp<=max([x3,x4]) and yp>=min([y3,y4]) and yp<=max([y3,y4]):
            return True 
    return False     

if __name__ == '__main__':
    playVideo("proj-lvl3-data/video-3.avi")
    #playVideo("proj-lvl3-data/video-0.avi")
            
        