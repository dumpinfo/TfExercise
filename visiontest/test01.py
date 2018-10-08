import os
import cv2
import sys

savefolder = sys.argv[2]

def compare_image(image1, image2, index):
   global savefolder
   diff_cnt = 0
   for i in range(0, image1.shape[0],4):
      for j in range(0, image1.shape[1],4):
          if abs(int(image1[i][j][0])- int(image2[i][j][0])) > 20 :
            diff_cnt = diff_cnt + 1
   fileidx  = "%05d"%(index)
   filename = savefolder + '/save' + fileidx + '.jpg'
   if diff_cnt > (40 * 40):
      cv2.imwrite(filename, image2)



source = sys.argv[1]
cam = cv2.VideoCapture(source)
img_counter = 0
frame_cnt = 0
ret, frame = cam.read()
lastframe = frame
while(cam.isOpened()):
    #global frame
    #global lastframe
    #global frame_cnt
    ret, frame = cam.read()
    #cv2.imshow('frame', frame)
    if not ret:
        break
    frame_cnt = frame_cnt + 1
    if(frame_cnt%8 == 0):
      compare_image(frame, lastframe, frame_cnt)
      lastframe = frame
    print(frame_cnt)
    #cv2.waitKey(1)

cam.release()
#cv2.destroyAllWindows()
