import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def heun_auto(x0, tn, f, acc):
    ti = 0
    xi = x0
    tlist = [ ti ]
    xlist = [ x0 ]
    tau = 1
    while ti < tn:
        while True:
            tip1 = ti + tau              # 次の時刻
            xip1_b = xi + tau * f(ti, xi)  # ← 予測子ステップ
            xip1 = xi + (tau/2) * (f(ti, xi) + f(tip1, xip1_b))  # ホイン法の更新式
            if np.max(np.abs((xip1 - xip1_b)/xip1_b)) < acc:
                break
            tau = tau / 2  # うまくいかなかったら、刻み幅を半分にする
        # うまくいったら刻み幅を倍にする
        tlist.append(tip1)
        xlist.append(xip1)
        ti, xi = tip1, xip1
        tau = min(2 * tau, tn - ti)

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
x0[0] = 1  
x0[1] = 0 
t, x = heun_auto(x0, 10, strogatz, 1e-5)

# リストをnumpy配列に変換してプロット
x_array = np.array(x)
plt.plot(t, x)
plt.legend()
plt.grid(True)
plt.show()