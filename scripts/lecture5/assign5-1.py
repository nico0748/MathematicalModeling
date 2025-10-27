import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def forward_euler(x0, tau, tf, f):
    ti = 0
    xi = x0
    tlist = [ ti ]
    xlist = [ x0 ]
    while ti < tf:
        tip1 = ti + tau
        xip1 = xi + tau * f(ti, xi)
        tlist.append(tip1)
        xlist.append(xip1)
        ti, xi = tip1, xip1
    return (tlist, xlist)
def strogatz( t, x ):
    # ストロガッツの恋愛方程式のパラメータ
    # dx/dt = ax + by (ロミオの愛の変化)
    # dy/dt = cx + dy (ジュリエットの愛の変化)
    a = 1  # ロミオの自己強化係数
    b = 1  # ジュリエットがロミオに与える影響
    c = 1  # ロミオがジュリエットに与える影響
    d = 1  # ジュリエットの自己強化係数
    fx = np.zeros_like(x)
    fx[0] = a * x[0] + b * x[1]  # ロミオの愛の変化率
    fx[1] = c * x[0] + d * x[1]  # ジュリエットの愛の変化率
    return fx

# 初期条件の設定
x0 = np.zeros(2)
x0[0] = 1  # ロミオの初期愛情度
x0[1] = 0  # ジュリエットの初期愛情度
t, x = forward_euler(x0, 0.1, 10, strogatz)

# リストをnumpy配列に変換してプロット
x_array = np.array(x)
plt.plot(t, x)
plt.legend()
plt.grid(True)
plt.show()


