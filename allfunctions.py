import numpy as np
import cv2
img=cv2.imread('image/orignal_lion.jpg',0) #the 2-D matrix
print(img.shape[0])
a=np.array([[1,2],
            [2,3]])
s=np.linalg.svd(a,compute_uv=False)
q=np.array(
            [[16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]])

cv2.imshow('img',img)               #converting the matrix back to image

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('changed.png',img)    #saving the image generated from array


import random

s=random.getstate()
print(random.getrandbits(4))
random.setstate(s)
print(random.getrandbits(4))


import math

math.pow(a,b)