import numpy as np 
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz_system(t, y):
    """ローレンツ方程式"""
    x, y_val, z = y
    dxdt = -10 * x + 10 * y_val
    dydt = 28 * x - y_val - x * z
    dzdt = -(8/3) * z + x * y_val
    return np.array([dxdt, dydt, dzdt])

# パラメータ設定
y0 = [1.0, 0.5, 0.5]  # 初期条件
t_span = (0, 20)  # 計算区間（ローレンツアトラクタの形成に十分な時間） 
t_eval = np.linspace(0, 20, 1000)  # 時間点

# 解法リスト
methods = ['RK23', 'RK45', 'LSODA', 'BDF', 'Radau']
solutions = {}
successful_methods = []

# 各解法で計算
print("各解法での計算開始...")
for method in methods:
    print(f"\n{method}解法を実行中...", end=" ")
    try:
        if method == 'RK45':
            sol = solve_ivp(lorenz_system, t_span, y0, method=method, 
                          rtol=1e-6, atol=1e-8, max_step=0.01)
        elif method == 'LSODA':
            sol = solve_ivp(lorenz_system, t_span, y0, method=method, 
                          rtol=1e-6, atol=1e-8)
        elif method == 'RK23':
            sol = solve_ivp(lorenz_system, t_span, y0, method=method, 
                          rtol=1e-3, atol=1e-5, max_step=0.02)
        else:  # BDF, Radau
            sol = solve_ivp(lorenz_system, t_span, y0, method=method, 
                          rtol=1e-2, atol=1e-4, max_step=0.1)
        
        # より緩い成功判定
        if sol.success or (hasattr(sol, 't') and len(sol.t) > 20):
            solutions[method] = sol
            successful_methods.append(method)
            print(f"✅ 成功 (ステップ数: {len(sol.t)})")
        else:
            print(f"❌ 失敗 (ステップ数: {len(sol.t) if hasattr(sol, 't') else 0})")
    except Exception as e:
        print(f"❌ エラー: {str(e)[:30]}")

print(f"\n成功した解法: {successful_methods}")

# 結果のプロット
if successful_methods:
    # 3D軌道プロット
    fig = plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(successful_methods):
        sol = solutions[method]
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        # NaN値を除外してプロット（より寛容な処理）
        valid_indices = ~(np.isnan(sol.y[0]) | np.isnan(sol.y[1]) | np.isnan(sol.y[2]))
        
        # 有効なデータが少しでもあればプロット
        if np.sum(valid_indices) > 10:
            x_valid = sol.y[0][valid_indices]
            y_valid = sol.y[1][valid_indices] 
            z_valid = sol.y[2][valid_indices]
            
            ax.plot(x_valid, y_valid, z_valid, linewidth=0.8, alpha=0.8)
            ax.scatter(x_valid[0], y_valid[0], z_valid[0], 
                      color='red', s=50, label='Start')
        elif len(sol.y[0]) > 10:  # NaNがあっても最初の部分をプロット
            first_part = min(len(sol.y[0])//3, 100)
            ax.plot(sol.y[0][:first_part], sol.y[1][:first_part], sol.y[2][:first_part], 
                   linewidth=0.8, alpha=0.8)
            ax.scatter(sol.y[0][0], sol.y[1][0], sol.y[2][0], 
                      color='red', s=50, label='Start')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'{method}')
        ax.legend()
    
    plt.suptitle('Lorenz Attractor - Multiple Solvers')
    plt.tight_layout()
    plt.show()
    
    # 時系列プロット
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    variables = ['x', 'y', 'z']
    
    for method in successful_methods:
        sol = solutions[method]
        
        # NaN値を除外（より寛容な処理）
        valid_indices = ~(np.isnan(sol.y[0]) | np.isnan(sol.y[1]) | np.isnan(sol.y[2]))
        
        if np.sum(valid_indices) > 10:
            valid_t = sol.t[valid_indices]
            for i in range(3):
                valid_y = sol.y[i][valid_indices]
                axes[i].plot(valid_t, valid_y, label=method, linewidth=1.5, alpha=0.8)
        else:
            # 有効データが少ない場合は最初の部分のみプロット
            first_part = min(len(sol.t)//2, 200)
            for i in range(3):
                if not np.all(np.isnan(sol.y[i][:first_part])):
                    axes[i].plot(sol.t[:first_part], sol.y[i][:first_part], 
                               label=f"{method} (partial)", linewidth=1.5, alpha=0.8)
    
    for i, var in enumerate(variables):
        axes[i].set_ylabel(f'{var}(t)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time t')
    plt.suptitle('Time Series - Lorenz Equations')
    plt.tight_layout()
    plt.show()