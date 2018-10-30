import numpy as np
import cv2
from extraction import P,y
import random
import math
import pywt


#program start
img=cv2.imread('image/orignal_lion.jpg')  #array
q=np.array(
            [[16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]]
            )
n=img.shape[0]
m=img.shape[1]
i=0
img_q=img
while i<n :
    j=0
    while j<m:
        for x in range(8):
            for y in range(8):
                img_q[i+x][j+y]=round(img[i+x][j+y]/q[x][y])
                img_q[i+x][j+y]=img_q[i+x][j+y]*q[x][y]
        j+=8
    i+=8
i=0
t1=np.zeros(2,2)
t2=np.zeros(2,2)
t3=np.zeros(2,2)
pblock=np.zeros(n/4,m/4)
cw=np.zeros(n/4,m/4)
while i<n:
    j=0
    while j<m:
        for x in range(2):
            for y in range(2):
                t1[x][y]=img_q[i+x+2][j+y]
        s1=np.linalg.svd(t1,compute_vh=False)
        for x in range(2):
            for y in range(2):
                t2[x][y]=img_q[i+x+2][j+y+2]
        s2=np.linalg.svd(t1,compute_vh=False)
        for x in range(2):
            for y in range(2):
                t3[x][y]=img_q[i+x][j+y+2]
        s3=np.linalg.svd(t1,compute_vh=False)
        if s1[1]>=s2[1]:
            b1=1
        else
            b1=0
        if s2[1]>=s3[1]:
            b2=1
        else
            b2=0
        if s1[1]>=s3[1]
            b3=1
        else
            b3=0
        cw[i/4][j/4]=b1^(b2^b3)
        pblock[i/4][j/4]=b1+b2+b3
        j+=4
    i+=4
#Deffie Hellmann Key Exchange Protocol
G=random.getrandbits(128)
a=random.getrandbits(128)
x=math.pow(G,a)%p
private=math.pow(y,a)%P
#end
iw=random.getrandbits(private)
sw=np.zeros(n/4,m/4)
for i in range(n/4):
    for j in range(m/4):
        sw[i][j]=iw^cw[i][j]
i=0
temp=np.zeros(4,4)
img_emb=np.zeros(n,m)
while i<n;
    j=0
    while j<m:
        for r in range(4):
            for c in range(4):
                temp[r][c]=imq[i+r][j+c]
        q_ada=11+2*pblock[i/4][j/4]
        coeffs=pywt.dw2(temp,'haar')
        ca,(ch,cv,cd)=coeffs
        ca[1][1]/=q_ada
        coeffs=ca,(ch,cv,cd)
        temp=pywt.idwt2(coeffs,'haar')
        for r in range(4):
            for c in range(4):
                img_emb[i+r][j+c]=temp[r][c]
        j+=4
    i+=4
cv2.imwrite('embedded.png',img_emb)
