import cv2
from line import Line
import numpy as np

def findLine(lowerColor, upperColor):       
    mask = cv2.inRange(frame, lowerColor, upperColor)
    image = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 200, apertureSize = 3)

    blur = cv2.GaussianBlur(edges, (5, 5), 1)
    lines = cv2.HoughLinesP(blur, 1, np.pi/180, 100, 100, 50)
    if lines is not None:
        return lines[0][0]
    

if __name__ == '__main__':
    video = cv2.VideoCapture("proj-lvl3-data/video-0.avi")

    while True:
        ret, frame = video.read()

        greenBoundaries = ([0, 130, 0], [50, 255, 50])
        
        #plava koordinate [x1, y1, x2, y2]
        bluePoints = findLine(np.array([120, 0, 0], dtype = "uint8"), np.array([255, 100, 100], dtype = "uint8"))
        #zelena
        greenPoints = findLine(np.array([0, 130, 0], dtype = "uint8"), np.array([59, 255, 50], dtype = "uint8"))
        
        if bluePoints is not None: 
            blue = bluePoints

        if greenPoints is not None: 
            green = greenPoints
            
        cv2.imshow("frame", frame)

        key = cv2.waitKey(25)
        if key == 27:
            break
        
    video.release()
    cv2.destroyAllWindows()


