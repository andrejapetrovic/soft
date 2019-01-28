import cv2

def playVideo(path):
    video = cv2.VideoCapture(path)

    while True:
        ret, frame = video.read()

        cv2.imshow(path, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    playVideo("proj-lvl3-data/video-3.avi")
    playVideo("proj-lvl3-data/video-0.avi")