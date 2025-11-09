import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# ===== ロジスティック改造モデル =====
def logistic_mod(t, x):
    fx = np.zeros_like(x)
    
    # ---- パラメータ ----
    r = 0.03             # 成長率
    N_inf = 13000        # 上限人口（万人）
    alpha = 2.0          
    beta  = 1.4          
    # --------------------
    
    # x[0] : 人口
    N = x[0]
    s = (N / N_inf)

    # f(N/N_inf) = (N/N_inf)^α
    # g(x) = x^β 
    fx[0] = r * N * (1 - s**alpha)**beta
    
    return fx


# ====== 時間設定 ======
ts = 1920
te = 2025
n  = 3000

# ====== 初期値 ======
x0 = np.zeros(1)
x0[0] = 5500     

# ====== 数値積分 ======
sol = integrate.solve_ivp(
    logistic_mod, [ts, te], x0,
    method="RK45",
    dense_output=True,
    t_eval=np.linspace(ts, te, n)
)

# ====== 描画 ======
t = np.linspace(ts, te, n)
z = sol.sol(t)

plt.plot(t, z.T)
plt.legend(['Population'])
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Population[10^4 people]")
plt.title("Japan population (Modified Logistic)")
plt.show()