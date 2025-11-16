import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sir_model(t, x0, r, c):
    S = x0[0]
    I = x0[1]
    R = x0[2]

    l = 0.1 # 免疫損失率

    N = S + I + R  # 総人口

    # SIR微分方程式（免疫損失を含む）
    dS_dt = -r * S * I / N + l * R  # 感染による減少 + 免疫損失による回復
    dI_dt = r * S * I / N - c * I   # 感染による増加 - 回復による減少
    dR_dt = c * I - l * R           # 回復による増加 - 免疫損失による減少
    
    return [dS_dt, dI_dt, dR_dt]

   
# 初期条件
x0 = np.zeros(3)
x0[0] = 99000
x0[1] = 1000
x0[2] = 0

r = 0.3 # 感染率
c = 0.05 # 回復率
    
# 時間範囲（統一）
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)  # t_spanと一致させる
    
# 微分方程式を解く
sol = solve_ivp(sir_model, t_span, x0, t_eval=t_eval, args=(r, c), method='RK45', rtol=1e-8)
    
# 結果のプロット
plt.figure(figsize=(12, 8))

# メインプロット
plt.plot(sol.t, sol.y[0], label='Susceptible (S)', color='blue')
plt.plot(sol.t, sol.y[1], label='Infected (I)', color='red')
plt.plot(sol.t, sol.y[2], label='Recovered (R)', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid()
plt.show()  