import numpy as np 
from scipy import linalg 
from scipy import integrate
import matplotlib.pyplot as plt 

def malthus(t, x): 
    fx = np.zeros_like(x)

    r = 0.00504   # 増加率 (1/day)  ← 推定値

    # x[0]: フォロワー数
    fx[0] = r * x[0]

    # 進行状況確認（任意）
    # 1000人未満 → 出力
    if x[0] < 1000:
        print(f"Day: {int(t)}, Followers: {int(x[0])}")

    return fx

# 初期条件
ts = 0        # 開始(2024/11/19からの経過日)
te = 730     # 終了(2026/11/19までの経過日)
n  = 3000     # 分割数(描画点)

x0 = np.zeros(1)
x0[0] = 143   # 2024/11/19 フォロワー数　→ # 2025/11/09 フォロワー数　856人　→ 推定増加率から計算


sol = integrate.solve_ivp(
    malthus, [ts, te], x0,
    method='RK45',
    dense_output=True,
    t_eval=np.linspace(ts, te, n)
)



t = np.linspace(ts, te, n)
z = sol.sol(t)

plt.plot(t, z.T)
plt.legend(['Followers'])
plt.grid(True)
plt.xlabel("Days from 2024-11-19")
plt.ylabel("Followers")
plt.title("Malthus Model (Follower Growth)")
plt.show()