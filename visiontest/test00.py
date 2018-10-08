import os
import cv2



source = "rtsp://admin:password@192.168.0.119/Streaming/Channels/1"
cam = cv2.VideoCapture(source)
img_counter = 0
while(cam.isOpened()):
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    if not ret:
        break
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
