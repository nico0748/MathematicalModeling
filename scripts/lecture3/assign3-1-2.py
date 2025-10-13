import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

h = 1000  #差分法の刻み幅
x = 0   #微分を求めたい点

A = np.zeros((h, h))
A = np.diag(-2*np.ones(h)) + np.diag(np.ones(h-1), 1) + np.diag(np.ones(h-1), -1)
A[0,1] = 0
A[h-1,h-2] = 0
A[0,0] = 1
A[h-1,h-1] = 1
print(A)

b = np.zeros((h, 1))
b[0] = 10      # 境界条件: u(0) = 10
b[h-1] = 0     # 境界条件: u(1) = 0
#print(b)

# 連立方程式 Au = b を解く
u_solution = scipy.linalg.solve(A, b.flatten())

# x座標の配列を作成
x_values = np.linspace(0, 1, h)

# 結果をプロット
#plt.figure(figsize=(10, 6))
plt.plot(x_values, u_solution, 'b-', linewidth=1)


plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True, alpha=0.3)
plt.show()
