import numpy as np
import cv2
from extraction import G,x,s
import random
import math
import pywt
from scipy.signal import signaltools
from scipy.ndimage.measurements import label


P=random.getrandbits(128)
b=random.getrandbits(128)
y=math.pow(G,b)%P
private=math.pow(x,b)%P
random.setstate(s)
iw=random.getrandbits(private)
img=cv2.imread('changed.png')      #name of the image that has to be checked
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
#t0=np.zeros(2,2)
t1=np.zeros(2,2)
t2=np.zeros(2,2)
t3=np.zeros(2,2)
pblock=np.zeros(n/4,m/4)
cw=np.zeros(n/4,m/4)
while i<n:
    j=0
    while j<m:
        for r in range(2):
            for c in range(2):
                t1[r][c]=img_q[i+r+2][j+c]
        s1=np.linalg.svd(t1,compute_vh=False)
        for r in range(2):
            for c in range(2):
                t2[r][c]=img_q[i+r+2][j+c+2]
        s2=np.linalg.svd(t1,compute_vh=False)
        for r in range(2):
            for c in range(2):
                t3[r][c]=img_q[i+r][j+c+2]
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
for i in range(n/4):
    for j in range(m/4):
        sw[i][j]=iw^cw[i][j]
i=0
ew=np.zeros(n/4,m/4)
temp=np.zeros(4,4)
while i<n:
    j=0
    while j<m:
        for r in range(4):
            for c in range(4):
                temp[r][c]=img[i+r][j+c]
        q_ada=11+2*pblock[i/4][j/4]
        coeffs=pywt.dw2(temp,'haar')
        ca,(ch,cv,cd)=coeffs
        ew[i/4][j/4]=round(ca[1][1]/q_ada)
        ew[i/4][j/4]=ew[i/4][j/4]%2
        j+=4
    i+=4
error_map=np.zeros(n/4,m/4)
no_ep=0
for i in range(n/4):
    for j in range(m/4):
        if ew[i][j]==sw[i][j]:
            error_map[i][j]=0
        else:
            error_map[i][j]=1
            no_ep+=1
m1=float(float(no_ep)/float((n/4)*(m/4)))
#applying median filter
error_map=np.array(error_map,dtype='f')
if m1<=0.15:
    error_map=signaltools.medfilt2d(error_map,3)
no_step=0
sterror_map=error_map
for i in range(1,n/4-1):
    count1=0;
    count2=0;
    if error_map[i-1][0]==1:
        count1+=1
    if error_map[i-1][1]==1:
        count1+=1
    if error_map[i][1]==1:
        count1+=1
    if error_map[i+1][1]==1:
        count1+=1
    if error_map[i+1][0]==1:
        count1+=1
    if count1>4:
        sterror_map[i][0]=1
        no_step+=1
    if error_map[i-1][m/4-1]==1:
        count2+=1
    if error_map[i-1][m/4-2]==1:
        count2+=1
    if error_map[i][m/4-2]==1:
        count2+=1
    if error_map[i+1][m/4-2]==1:
        count2+=1
    if error_map[i+1][m/4-1]==1:
        count2+=1
    if count2>4:
        sterror_map[i][m/4-1]=1
        no_step+=1
for i in range(1,m/4-1):
    count1=0
    count2=0
    if error_map[0][i-1]==1:
        count1+=1
    if error_map[0][i+1]==1:
        count1+=1
    if error_map[1][i-1]==1:
        count1+=1
    if error_map[1][i]==1:
        count1+=1
    if error_map[1][i+1]==1:
        count1+=1
    if count1>4:
        sterror_map[0][i]=1
        no_step+=1
    if error_map[n/4-1][i-1]==1:
        count2+=1
    if error_map[n/4-1][i+1]==1:
        count2+=1
    if error_map[n/4-2][i-1]==1:
        count2+=1
    if error_map[n/4-2][i]==1:
        count2+=1
    if error_map[n/4-2][i+1]==1:
        count2+=1
    if count2>4:
        sterror_map[n/4-1][i]=1
        no_step+=1
for i in range(1,n/4-1):
    for j in range(1,m/4-1):
        count=0
        if error_map[i-1][j-1]==1:
            count+=1
        if error_map[i-1][j]==1:
            count+=1
        if error_map[i-1][j+1]==1:
            count+=1
        if error_map[i][j-1]==1:
            count+=1
        if error_map[i][j+1]==1:
            count+=1
        if error_map[i+1][j-1]==1:
            count+=1
        if error_map[i+1][j]==1:
            count+=1
        if error_map[i+1][j+1]==1:
            count+=1
        if count>4:
            sterror_map[i][j]=1
            no_step+=1
m2=no_step/no_ep
structure=np.ones((3,3),dtype=np.int)
labelled,ncomponents=label(sterror_map,structure)
nostep=0
for i in range(n/4):
    for j in range(m/4):
        if sterror_map[i][k]==1:
            nostep+=1
m3=nostep/ncomponents
indices=np.indices(sterror_map.shape).T[:,:,[1,0]]
vertices=np.zeros((n/4,m/4))
no_v=0
largestcomponent=0
teri=np.zeros(ncomponents)
size_largest=0
for i in range(1,ncomponents+1):
    for j in range(len(indices[labelled==i])):
        if vertices[indices[labelled==i][j][0]]==0:
            vertices[indices[labelled==i][j][0]]=1
            no_v+=1
        if vertices[indices[labelled==i][j][1]]==0:
            vertices[indices[labelled==i][j][1]]=1
            no_v+=1
    if size_largest<no_v:
        largestcomponent=i
        teri=no_v
    teri[i-1]=no_v
    no_v=0
m4=np.std(teri)
