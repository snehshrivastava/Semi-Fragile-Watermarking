import random
import math
P=random.getrandbits(4)
G=random.getrandbits(4)
b=random.getrandbits(4)
a=random.getrandbits(4)
x=math.pow(G,a)%P
y=math.pow(G,b)%P
private=math.pow(y,a)%P
private+=10
