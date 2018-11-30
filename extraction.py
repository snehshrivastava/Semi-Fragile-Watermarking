import numpy as np
import cv2
from generation import s
from getval import G,x,private
import random
import math
import pywt
from scipy.signal import signaltools
from scipy.ndimage.measurements import label


#P=random.getrandbits(128)
#b=random.getrandbits(128)
#y=math.pow(G,b)%P
#private=math.pow(x,b)%P

random.setstate(s)
private=int(private)
iw=random.getrandbits(private)
img=cv2.imread('changed.png',0)      #name of the image that has to be checked
print(img.shape)
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
row=int(n/4)
col=int(m/4)
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
#t0=np.zeros(2,2)
t1=np.zeros((2,2))
t2=np.zeros((2,2))
t3=np.zeros((2,2))
pblock=np.zeros((row,col),dtype=int)

cw=np.zeros((row,col),dtype=int)
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
        s2=np.linalg.svd(t2,compute_uv=False)
        for r in range(2):
            for c in range(2):
                t3[r][c]=img_q[i+r][j+c+2]
        s3=np.linalg.svd(t3,compute_uv=False)
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
        cw[int(i/4)][int(j/4)]=b1^(b2^b3)
        pblock[int(i/4)][int(j/4)]=b1+b2+b3
        j+=4
    i+=4
sw=np.zeros((row,col))
for i in range(row):
    for j in range(col):
        sw[i][j]=iw^cw[i][j]
i=0
ew=np.zeros((row,col))
temp=np.zeros((4,4))
while i<n:
    j=0
    while j<m:
        for r in range(4):
            for c in range(4):
                temp[r][c]=img[i+r][j+c]
        q_ada=11+2*pblock[int(i/4)][int(j/4)]
        coeffs=pywt.dwt2(temp,'haar')
        ca,(ch,cv,cd)=coeffs
        ew[int(i/4)][int(j/4)]=round(ca[1][1]/q_ada)
        ew[int(i/4)][int(j/4)]=ew[int(i/4)][int(j/4)]%2
        j+=4
    i+=4
error_map=np.zeros((row,col))
no_ep=0
for i in range(row):
    for j in range(col):
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
mterror_map=error_map
for i in range(1,int(n/4)-1):
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
    elif count1>0:
        mterror_map[i][0]=1
    if error_map[i-1][int(m/4)-1]==1:
        count2+=1
    if error_map[i-1][int(m/4)-2]==1:
        count2+=1
    if error_map[i][int(m/4)-2]==1:
        count2+=1
    if error_map[i+1][int(m/4)-2]==1:
        count2+=1
    if error_map[i+1][int(m/4)-1]==1:
        count2+=1
    if count2>4:
        sterror_map[i][int(m/4)-1]=1
        no_step+=1
    elif count2>0:
        mterror_map[i][int(m/4)-1]=1
for i in range(1,int(m/4)-1):
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
    elif count1>0:
        mterror_map[0][i]=1
    if error_map[row-1][i-1]==1:
        count2+=1
    if error_map[row-1][i+1]==1:
        count2+=1
    if error_map[row-2][i-1]==1:
        count2+=1
    if error_map[row-2][i]==1:
        count2+=1
    if error_map[row-2][i+1]==1:
        count2+=1
    if count2>4:
        sterror_map[row-1][i]=1
        no_step+=1
    elif count2>0:
        mterror_map[row-1][i]=1
for i in range(1,row-1):
    for j in range(1,col-1):
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
        elif count>0:
            mterror_map[i][j]=1
m2=no_step/no_ep
structure=np.ones((3,3),dtype=np.int)
labelled,ncomponents=label(sterror_map,structure)
nostep=0
for i in range(row):
    for j in range(col):
        if sterror_map[i][j]==1:
            nostep+=1
m3=nostep/ncomponents
indices=np.indices(sterror_map.shape).T[:,:,[1,0]]
vertices=np.zeros((row,col))
no_v=0
largestcomponent=0
teri=np.zeros(ncomponents)
size_largest=0
#check the largest component
for i in range(1,ncomponents+1):
    no_v=len(indices[labelled==i])
    if size_largest<no_v:
        largestcomponent=i
        size_largest=no_v
    teri[i-1]=no_v
    no_v=0



m4=np.std(teri)
x_centroid=0
y_centroid=0
for j in range(len(indices[labelled==largestcomponent])):
    x_centroid+=indices[labelled==largestcomponent][j][0]-1
    y_centroid+=indices[labelled==largestcomponent][j][1]-1

x_centroid=float(x_centroid/len(indices[labelled==largestcomponent]))

y_centroid=float(y_centroid/len(indices[labelled==largestcomponent]))

sizeC=math.sqrt(size_largest)

modifiedsterror_map=sterror_map
modifiedmterror_map=mterror_map
mtep=0
step=0
ctep=0
for i in range(row):
    for j in range(col):
        dist=math.sqrt(math.pow(x_centroid-i,2)+math.pow(y_centroid-j,2))
        if dist>sizeC:
            #if modifiedmterror_map[i][j]==1:
            modifiedmterror_map[i][j]=0
            modifiedsterror_map[i][j]=0
        if modifiedmterror_map[i][j]==1:
            mtep+=1
        if modifiedsterror_map[i][j]==1:
            step+=1
        if modifiedmterror_map[i][j]==1 and modifiedsterror_map[i][j]==1:
            ctep+=1
m5=step/(mtep+step-ctep)
