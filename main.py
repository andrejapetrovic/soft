import cv2
import numpy as np

def findLine(lowerColor, upperColor, frame):       
    mask = cv2.inRange(frame, lowerColor, upperColor)
    image = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1], 100, 250)

    blur = cv2.GaussianBlur(edges, (5, 5), 0)
    lines = cv2.HoughLinesP(blur, 1, np.pi/180, 100, 175, 200)
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,0),2)
            return [x1, y1, x2, y2]
    

if __name__ == '__main__':
    video = cv2.VideoCapture("proj-lvl3-data/video-0.avi")

    boxes = {}
    intersectedBlue = {}
    intersectedGreen = {}
    boxId = 0
    while True:
        ret, frame = video.read()
        
        #plava koordinate [x1, y1, x2, y2]
        blue = findLine(np.array([120, 0, 0], dtype = "uint8"), np.array([255, 100, 100], dtype = "uint8"), frame)
        #zelena
        green = findLine(np.array([0, 130, 0], dtype = "uint8"), np.array([59, 255, 50], dtype = "uint8"), frame)
            
        #if blue is not None:
         #   print(blue)

        #if green is not None:
         #   print(green)

        #konture  
        mask = cv2.inRange(frame, np.array([160, 160, 160], dtype = "uint8"), np.array([255, 255, 255], dtype = "uint8"))
        image = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, contours, -1, (255,0,0), 2)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            #print(w, h)
            if(w>9 or h>8) and (w<25 or h<26):
                #geometrijski centar
                cx = x+w*0.5
                cy = y+h*0.5
                dists = {}
                if not boxes:
                    boxId += 1
                    #[id] = [x, y, sirina, visina, centar-x, cenar-y]
                    boxes[boxId] = [x, y, w, h, cx, cy]
                    cv2.rectangle(frame,(x, y),(x+w,y+h),(0,0,255),1)
                    cv2.putText(mask,str(boxId), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                else:
                    for key in boxes:
                        box = boxes[key]
                        dists[key] = (np.linalg.norm(np.array((cx, cy))-np.array((box[4], box[5]))))
                
                    minDist = min(dists.values())
                        
                    if (minDist < 20):
                        k = (list(dists.keys())[list(dists.values()).index(minDist)])
                        boxes[k] = [x,y,w,h,x+w*0.5,y+h*0.5,False]
                        cv2.rectangle(frame,(x, y),(x+w,y+h),(0,0,255),1)
                        cv2.putText(mask,"id:" + str(k), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                    else:
                        boxId += 1
                        boxes[boxId] = [x, y, w, h, cx, cy, False]
                        cv2.rectangle(frame,(x, y),(x+w,y+h),(0,0,255),1)
                        cv2.putText(mask,str(boxId), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        key = cv2.waitKey(25)
        if key == 27:
            break
        
    video.release()
    cv2.destroyAllWindows()


