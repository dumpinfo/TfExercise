import numpy as np
import struct
import matplotlib.pyplot as plt

def bound_val(val):
   if val > 255:
      val = 255
   if val < 0:
      val = 0
   return val

def add_noise(img):
    for i in range(0, img.shape[0]):
      for j in range(0, img.shape[1]):
        val = (int)(img[i][j])
        noise_r =  np.random.randint(0,8)
        #print(noise_r)
        if( noise_r <2):
           img[i][j] = img[i][j] - 50 + np.random.randint(0,120)
        else :
           img[i][j] = img[i][j] - 40 + np.random.randint(0,40)
        if(img[i][j]<0):
           img[i][j] = np.random.randint(0,180) + img[i][j]
        img[i][j] = bound_val(img[i][j])
    return img
        

def loadImageSet(filename):
    print "load image set",filename
    binfile= open(filename, 'rb')
    buffers = binfile.read()
 
    head = struct.unpack_from('>IIII' , buffers ,0)
    print "head,",head
 
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    #[60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B' #like '>47040000B'
 
    imgs = struct.unpack_from(bitsString,buffers,offset)
 
    binfile.close()
    imgs = np.reshape(imgs,[imgNum,1,width*height])
    print "load imgs finished"
    return imgs
 
def loadLabelSet(filename):
 
    print "load label set",filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()
 
    head = struct.unpack_from('>II' , buffers ,0)
    print "head,",head
    imgNum=head[1]
 
    offset = struct.calcsize('>II')
    numString = '>'+str(imgNum)+"B"
    labels = struct.unpack_from(numString , buffers , offset)
    binfile.close()
    labels = np.reshape(labels,[imgNum,1])
 
    print 'load label finished'
    return labels
 
if __name__=="__main__":
 
    imgs = loadImageSet("./train-images-idx3-ubyte")
    labels = loadLabelSet("./train-labels-idx1-ubyte")
    
    
    print(imgs.shape)
    print(labels.shape)
    
    padsize = 120000
    add_imgs   = np.zeros((padsize, 1, 784))
    add_labels = np.zeros((padsize, 1))
    
    for i in range(0, padsize):
       if(i%1000 ==0 ):
           print(i)
       index = np.random.randint(0, 60000)
       im = imgs[index]
       lb = labels[index]
       im = add_noise(im)
       add_imgs[i]   = im
       add_labels[i] = lb
    
    imgs_new    = np.concatenate((imgs,   add_imgs ), axis=0)
    labels_new  = np.concatenate((labels, add_labels ), axis=0)
    
    print(imgs_new.shape)
    print(labels_new.shape)
    
    np.save('train_img.npy',imgs_new  )
    np.save('train_lb.npy',labels_new )
    
    index = np.random.randint(60005, (60000+padsize))
    im = imgs_new[index]
    im = im.reshape(28,28)
    print(labels_new[index])
    
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.show()
    #imgs = loadImageSet("t10k-images.idx3-ubyte")
    #labels = loadLabelSet("t10k-labels.idx1-ubyte")
