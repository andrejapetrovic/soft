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
        
        mask = cv2.inRange(frame, np.array([225, 225, 225], dtype = "uint8"), np.array([255, 255, 255], dtype = "uint8"))
        image = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, contours, -1, (255,0,0), 2)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            point = cv2.boundingRect(contour);
            print(point);
            cv2.rectangle(frame,(x, y),(x+w,y+h),(0,0,255),2)


        cv2.imshow("frame", frame)

        key = cv2.waitKey(25)
        if key == 27:
            break
        
    video.release()
    cv2.destroyAllWindows()


