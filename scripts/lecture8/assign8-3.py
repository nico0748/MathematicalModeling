import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sird_model(t, x0, r, c, mu):
    """
    SIRDモデル（死亡者数を含む）
    S: 感受性人口, I: 感染者数, R: 回復者数, D: 死亡者数
    """
    S = x0[0]
    I = x0[1]
    R = x0[2]
    D = x0[3]

    l = 0.05  # 免疫損失率

    N = S + I + R  # 生存総人口（死亡者は除く）

    # SIRD微分方程式（死亡を含む）
    dS_dt = -r * S * I / N + l * R  # 感染による減少 + 免疫損失による増加
    dI_dt = r * S * I / N - c * I - mu * I  # 感染による増加 - 回復による減少 - 死亡による減少
    dR_dt = c * I - l * R           # 回復による増加 - 免疫損失による減少
    dD_dt = mu * I                  # 死亡率 × 感染者数
    
    return [dS_dt, dI_dt, dR_dt, dD_dt]

   
# 初期条件（SIRDモデル用に拡張）
x0 = np.zeros(4)  # S, I, R, D の4要素
x0[0] = 99000  # S: 感受性人口
x0[1] = 1000   # I: 感染者数
x0[2] = 0      # R: 回復者数
x0[3] = 0      # D: 死亡者数

r = 0.3    # 感染率
c = 0.05   # 回復率
mu = 0.01  # 死亡率
    
# 時間範囲（統一）
t_span = (0, 300)
t_eval = np.linspace(0, 300, 1000)  # t_spanと一致させる
    
# 微分方程式を解く
sol = solve_ivp(sird_model, t_span, x0, t_eval=t_eval, args=(r, c, mu), method='RK45', rtol=1e-8)
    
# 結果のプロット
plt.figure(figsize=(15, 10))

plt.plot(sol.t, sol.y[0], label='Susceptible (S)', color='blue')
plt.plot(sol.t, sol.y[1], label='Infected (I)', color='red')
plt.plot(sol.t, sol.y[2], label='Recovered (R)', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid()
plt.show()  