import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axes3d import Axes3D

d = 1
node = 51
n = node - 1
h = d/n

f = 20

A = np.zeros((node**2, node**2))
b = np.zeros((node**2,1))
##################################################
for i in range(node):
    for j in range(node):
        # 格子点のインデックス
        idx = i + node * j
        
        # 境界条件 u = 0
        if i == 0 or i == n  or j == 0 or j == n:
            A[idx, idx] = 1
            b[idx] = 0
        
        # 内部格子点
        else:
            # 中心格子点(i, j)
            A[idx, idx] = 4
            
            # 隣接格子点
            # 左の点 (i-1, j)
            ##A[idx, (i-1) + node*j] = -1
            A[idx, (i-1) + node*j] = -1
            # 右の点 (i+1, j)
            A[idx, (i+1) + node*j] = -1
            
            # 上の点 (i, j-1)
            A[idx, idx - node] = -1
            
            # 下の点 (i, j+1)
            A[idx, idx + node] = -1
            
            # 右辺項（熱源項）
            b[idx] = f * (h**2)
##################################################
u = linalg.solve(A, b)

X = np.zeros((node**2, 1))
Y = np.zeros((node**2, 1))
for i in range(n+1):
    for j in range(n+1):
        X[ i+node*j ] = i*h
        Y[ i+node*j ] = j*h
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cm = plt.cm.get_cmap('coolwarm')
mappable = ax.scatter(X, Y, u, c=u[:,0], cmap=cm)
fig.colorbar(mappable, ax=ax)
plt.show()