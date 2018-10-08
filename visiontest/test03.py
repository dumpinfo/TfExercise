import os
import cv2
import numpy as np

def proc_image(image1):
   image2 = np.array(image1)
   for i in range(2, image1.shape[0]-2):
      for j in range(2, image1.shape[1]-2):
          image2[i][j][0] = image1[i][j][0]
          image2[i][j][1] = image1[i][j][0]
          image2[i][j][2] = image1[i][j][0]
          if image1[i][j][0] > 227 and image1[i-1][j][0] and image1[i+1][j][0]:
            image2[i][j][0] = 255
            image2[i][j][1] = 255
            image2[i][j][2] = 255
   return image2


rootdir = 'file2'
list = os.listdir(rootdir)
for i in range(0,len(list)):
       path = os.path.join(rootdir,list[i])
       if os.path.isfile(path):
          print(path)
          image = cv2.imread(path)
          image = proc_image(image)
          cv2.imwrite(path, image)