import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

xm = 1 #長さ[m]
node = 11 #分割数
h = xm/(node-1) #一要素の長さ0.1[m] = dx

A = np.zeros((node, node))
f = np.zeros(node)

ﬀ = 20 #右辺の値

# 境界条件:
ul = 0 #左端の温度
ur = 10 #右端の温度

tmax = 0.5 #いつまで解きたいか
dt = 0.001 # dt

####################################
# ここを書く
for j in range(node):
    if j == 0:
        A[j, j] = 1
        f[j] = 0
    elif j == node-1:
        A[j, j] = 1
        f[j] = 0
    else:
        A[j, j-1] = -dt/(h**2)
        A[j, j] = 1 + 2 * dt/(h**2)
        A[j, j+1] = -dt/(h**2)

        f[j] = ff*dt
####################################
u = np.zeros(node)
# 初期値:
u[0] = ul
u[node-1] = ur
u[node//2] = 1

print(u)

t = 0 # 計算したい時刻
i = 0
x = [i/(node-1)*xm for i in range(node)] #グラフ用
# while 文の中身も改良する
while t < tmax:
    u = linalg.solve(A, f+u)
    u[0] = ul
    u[node-1] = ur
    t += dt
    if i % 20 == 0:
        plt.plot(x, u)
        plt.show()
    i += 1
    print("t = ", t)
plt.close('all')