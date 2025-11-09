import numpy as np 
from scipy import linalg 
from scipy import integrate
import matplotlib.pyplot as plt 

def malthus( t, x ): 
    fx = np.zeros_like(x) 
    a = 0.03  #xが増える割合
    p_l = 12000 # 人口の上限値
    fx = np.zeros_like(x) 
    # x[0]: 方程式におけるx 
    fx[0] =  a * x[0] * (1 - x[0] / p_l)
    return fx

# 初期条件の設定
# 時間の開始ts、終了te、分割数n(グラフ描画のプロット数)
ts = 0
te =88
n = 3000

# 初期値の設定
x0 = np.zeros(1)
x0[0] = 5500
# strogatz:解きたい関数　[ts, te]:時間範囲　x0:初期値 method="RK45":ルンゲ・クッタ45 dense_output=True:補間を有効に t_eval:評価する時間点
sol = integrate.solve_ivp(malthus, [ts, te], x0, method='RK45', dense_output=True, t_eval=np.linspace(ts, te, n))

t = np.linspace(ts, te, n)
z = sol.sol(t)
plt.plot(t, z.T)
plt.legend(['x'])
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Population [10000 people]")
plt.show()