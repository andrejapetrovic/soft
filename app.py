import cv2
import numpy as np
import os
from keras.preprocessing import image

boxes = {}
sum = [0]

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
    
def playVideo(path, model):
    video = cv2.VideoCapture(path)
    
    frame_count = 0
    num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    sum[0] = 0 
    boxes.clear()

    boxId = 0
    while True:
        ret, frame = video.read()

        frame_count += 1

        #plava linija - koordinate [x1, y1, x2, y2]
        blue = findLine(np.array([120, 0, 0]), np.array([255, 100, 100]), frame)
        #zelena
        green = findLine(np.array([0, 130, 0]), np.array([59, 255, 50]), frame)

        #konture  
        mask = cv2.inRange(frame, np.array([160, 160, 160]), np.array([255, 255, 255]))
        image = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                    currentKey = boxId
                else:
                    for key in boxes:
                        box = boxes[key]
                        dists[key] = (np.linalg.norm(np.array((cx, cy))-np.array((box[4], box[5]))))
                
                    minDist = min(dists.values())
                        
                    if (minDist < 20 and minDist>2):
                        k = (list(dists.keys())[list(dists.values()).index(minDist)])
                        boxes[k] = [x, y, w, h, x+w*0.5, y+h*0.5, boxes[k][6], boxes[k][7]]
                        currentKey = k
                    if (minDist >= 20):
                        boxId += 1
                        boxes[boxId] = [x, y, w, h, cx, cy, 0, 0]

            if currentKey is not -1:
                #cv2.rectangle(frame,(x, y),(x+w,y+h),(0,0,255),1)
                #cv2.putText(frame,"id:" + str(currentKey), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))                        
                if blue is not None:
                    checkIntersection(blue, currentKey, frame, "blue", mask, model)
                if green is not None:
                    checkIntersection(green, currentKey, frame, "green", mask, model)      

        cv2.imshow(path, frame)
        #cv2.imshow("mask", mask)
        
        key = cv2.waitKey(1)
        if frame_count == num_of_frames or key == 27:
            video.release()
            cv2.destroyAllWindows()
            return sum[0]
            

def checkIntersection(line, key, frame, lineColor, mask, model):
    x, y, w, h, cx, cy, passedBlue, passedGreen = boxes[key]
    #proverava se presek svih stranica kvadrata s kojim je uokvirena kontura
    #sa linijom, pocevsi od gornje stranice u smeru kazaljke na satu
    if lineLineIntersection([x, y, x+w, y], line) or lineLineIntersection([x+w, y, x+w, y+h], line) or lineLineIntersection([x, y+h, x+w, y+h], line) or lineLineIntersection([x, y, x, y+h], line):
        if lineColor is "blue" and passedBlue == 0:
            boxes[key][6] = 1
            n = getPrediction(key, mask, model)
            print("sum=",sum[0],"+",n)
            sum[0] += n
            print("sum=",sum[0])
        if lineColor is "green" and passedGreen == 0:
            boxes[key][7] = 1
            n = getPrediction(key, mask, model)
            print("sum=",sum[0],"-",n)
            sum[0] -= n
            print("sum=",sum[0])

def getPrediction(key, mask, model):
    x, y, w, h, _,_,_,_ = boxes[key]
    n = 5
    img = mask[y-n:y+h+n, x-n:x+w+n]
    cv2.imshow("cropped", img)
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_NEAREST)
    img = img/255
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)

    pred = model.predict(im2arr)
    prob = np.max(pred)
    num = np.argmax(pred)
    print("number: ",num,"probability: ",prob)
    return num


def lineLineIntersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2 
    a = (x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)
    b = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    k = a/b
    if k>=0.0 and k<=1.0:
        xp = x1 + k*(x2 - x1)
        yp = y1 + k*(y2 - y1)
        u = 2
        if xp>=min([x3,x4])+u and xp<=max([x3,x4])-u and yp>=min([y3,y4])+u and yp<=max([y3,y4])-u:
            return True 
    return False     

if __name__ == '__main__':
    playVideo("proj-lvl3-data/video-3.avi")
    #playVideo("proj-lvl3-data/video-0.avi")
            
        