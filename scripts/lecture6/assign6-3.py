import numpy as np 
from scipy import linalg 
from scipy import integrate
import matplotlib.pyplot as plt 

def strogatz( t, x ): 
    fx = np.zeros_like(x) 
    a = 0.8  #xが増える割合
    b = 0.5   #xが食べられて減る割合
    c = 0.5   #xを食べてyが増える割合
    d = 0.3   #xを食べられずyが減る割合
    fx = np.zeros_like(x) 
    # x[0]: 方程式におけるx x[1]: 方程式におけるy
    fx[0] = a * x[0] - b * x[0] * x[1]  
    fx[1] = c * x[0] * x[1] - d * x[1]  
    return fx

# 初期条件の設定
# 時間の開始ts、終了te、分割数n(グラフ描画のプロット数)
ts = 0
te =100
n = 3000

# 初期値の設定
x0 = np.zeros(2)
x0[0] = 1000 #被食者の個体値
x0[1] = 1000 #捕食者の個体値
# strogatz:解きたい関数　[ts, te]:時間範囲　x0:初期値 method="RK45":ルンゲ・クッタ45 dense_output=True:補間を有効に t_eval:評価する時間点
sol = integrate.solve_ivp(strogatz, [ts, te], x0, method='RK45', dense_output=True, t_eval=np.linspace(ts, te, n))

t = np.linspace(ts, te, n)
z = sol.sol(t)
plt.plot(t, z.T)
plt.legend(['x', 'y', 'z'])
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Population")
plt.show()