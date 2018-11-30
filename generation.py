from __future__ import division
import numpy as np
import cv2
from getval import P,y,a,private
import random
import math
import pywt


#program start
#Deffie Hellmann Key Exchange Protocol
#G=random.getrandbits(128)
#a=random.getrandbits(128)
#x=math.pow(G,a)%P
#private=math.pow(y,a)%P
#end
img=cv2.imread('image/orignal_lion.jpg',0)  #array
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
print(img.shape)
i=0
img_q=img
while i<n :
    j=0
    while j<m:
        for r in range(8):
            for c in range(8):
                img_q[i+r][j+c]=img[i+r][j+c]/q[r][c]
#                img_q[i+r][j+c]=img_q[i+r][j+c]*q[r][c]
        j+=8
    i+=8
np.round(img_q)
while i<n :
    j=0
    while j<m:
        for r in range(8):
            for c in range(8):
#                img_q[i+r][j+c]=float(img[i+r][j+c]/q[r][c])
                img_q[i+r][j+c]=img_q[i+r][j+c]*q[r][c]
        j+=8
    i+=8




i=0
t1=np.zeros((2,2))
t2=np.zeros((2,2))
t3=np.zeros((2,2))
row=int(n/4)
col=int(m/4)
pblock=np.zeros((row,col))
cw=np.zeros((row,col),dtype=int)
print(img_q[0][0])
while i<n:
    j=0
    while j<m:
        for r in range(2):
            for c in range(2):
                t1[r][c]=img_q[i+r+2][j+c]
        s1=np.linalg.svd(t1,compute_uv=False)
        for r in range(2):
            for c in range(2):
                t2[r][c]=img_q[i+r+2][j+c+2]
        s2=np.linalg.svd(t1,compute_uv=False)
        for r in range(2):
            for c in range(2):
                t3[r][c]=img_q[i+r][j+c+2]
        s3=np.linalg.svd(t1,compute_uv=False)
        if s1[1]>=s2[1]:
            b1=1
        else:
            b1=0
        if s2[1]>=s3[1]:
            b2=1
        else:
            b2=0
        if s1[1]>=s3[1]:
            b3=1
        else:
            b3=0
        abc=int(i/4)
        defi=int(j/4)
        cw[abc][defi]=b1^(b2^b3)
        pblock[abc][defi]=b1+b2+b3
        j+=4
    i+=4

s=random.getstate()
private=int(private)
print(private)
iw=random.getrandbits(private)
sw=np.zeros((row,col))
for i in range(row):
    for j in range(col):
        sw[i][j]=iw^cw[i][j]
i=0
temp=np.zeros((4,4))
img_emb=np.zeros((n,m))
while i<n:
    j=0
    while j<m:
        for r in range(4):
            for c in range(4):
                temp[r][c]=img[i+r][j+c]
        abc=int(i/4)
        defi=int(j/4)
        q_ada=11+2*pblock[abc][defi]
        coeffs=pywt.dwt2(temp,'haar')
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
