import numpy as np
import scipy.linalg

h = 1000  #差分法の刻み幅
x = 0   #微分を求めたい点

#Aは1000行1000列の行列で2行目以降は1,-2,1を順にずらしていく
A = np.zeros((h, h))
for i in range(h):
    if i == 0:
        A[i][i] = 1
    elif i == 1:
        A[i][i] = 1
        A[i][i-1] = -2
        A[i][i-2] = 1
    #elif i == h-2:
        #A[i][i] = 1
    elif i == h:
        A[i][i] = 1
    elif i == h-2:
        A[i][i] = 1
        A[i][i-1] = -2
        A[i][i-2] = 1
    else:
        A[i][i-1] = 1
        A[i][i] = -2
        A[i][i+1] = 1

print(A)
#bは1000行1列の行列で2行目以降は0を入れていく
b = np.zeros((h, 1))
for i in range(h-1):
    if i == 0:
        b[i] = 10
    elif i == h-2:
        b[i+1] = 0
    else:
        b[i+1] = 0
print(b)

def u(x, A, b):
    for x in range(h):
        if x == 0:
            return 10
        elif x == h-1:
            return 0
        elif 0 < x < h-1:
            A_inv = scipy.linalg.inv(A)
            return np.dot(A_inv, b)
    

print(u(x, A, b))