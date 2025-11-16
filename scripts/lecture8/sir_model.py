"""
SIRモデル（感染症数理モデル）の数値解析

S: 感染症に免疫がない人 (Susceptible)
I: 現在，感染症にかかっている人(健常者Sにうつす人) (Infected)  
R: 感染症が治ったor亡くなった人(うつさない&かからない人) (Recovered)

微分方程式系:
dS/dt = -β * S * I / N
dI/dt = β * S * I / N - γ * I  
dR/dt = γ * I

ここで：
β: 感染率 (接触率 × 感染確率)
γ: 回復率 (1/感染期間)
N: 総人口 (S + I + R = N = 一定)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sir_model(t, y, beta, gamma):
    """
    SIRモデルの微分方程式系
    
    Parameters:
    t: 時間
    y: [S, I, R] の状態ベクトル
    beta: 感染率
    gamma: 回復率
    
    Returns:
    [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = y
    N = S + I + R  # 総人口
    
    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I
    
    return [dS_dt, dI_dt, dR_dt]

def solve_sir_model(S0, I0, R0, beta, gamma, t_max=100, num_points=1000):
    """
    SIRモデルを数値的に解く
    
    Parameters:
    S0, I0, R0: 初期値
    beta: 感染率
    gamma: 回復率  
    t_max: 計算期間
    num_points: 時間点の数
    
    Returns:
    solution object
    """
    # 初期条件
    y0 = [S0, I0, R0]
    
    # 時間範囲
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, num_points)
    
    # 微分方程式を解く
    sol = solve_ivp(sir_model, t_span, y0, t_eval=t_eval, 
                    args=(beta, gamma), method='RK45', rtol=1e-8)
    
    return sol

def plot_sir_results(sol, title="SIR Model"):
    """
    SIRモデルの結果をプロット
    """
    plt.figure(figsize=(12, 8))
    
    # 時系列プロット
    plt.subplot(2, 2, 1)
    plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='S (Susceptible)')
    plt.plot(sol.t, sol.y[1], 'r-', linewidth=2, label='I (Infected)')  
    plt.plot(sol.t, sol.y[2], 'g-', linewidth=2, label='R (Recovered)')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of People')
    plt.title(f'{title} - Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 位相図 (S-I平面)
    plt.subplot(2, 2, 2)
    plt.plot(sol.y[0], sol.y[1], 'purple', linewidth=2)
    plt.scatter(sol.y[0][0], sol.y[1][0], color='red', s=100, label='Start', zorder=5)
    plt.scatter(sol.y[0][-1], sol.y[1][-1], color='blue', s=100, label='End', zorder=5)
    plt.xlabel('S (Susceptible)')
    plt.ylabel('I (Infected)')
    plt.title('Phase Portrait (S-I plane)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 感染者数の拡大図
    plt.subplot(2, 2, 3)
    plt.plot(sol.t, sol.y[1], 'r-', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('I (Infected)')
    plt.title('Infected Population Detail')
    plt.grid(True, alpha=0.3)
    
    # 総人口の確認
    plt.subplot(2, 2, 4)
    total_pop = sol.y[0] + sol.y[1] + sol.y[2]
    plt.plot(sol.t, total_pop, 'k-', linewidth=2, label='S+I+R')
    plt.xlabel('Time (days)')
    plt.ylabel('Total Population')
    plt.title('Population Conservation Check')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_basic_reproduction_number(beta, gamma):
    """
    基本再生産数 R0 を計算
    R0 = β/γ
    R0 > 1: 感染拡大
    R0 < 1: 感染収束
    """
    return beta / gamma

def analyze_sir_scenario(S0, I0, R0, beta, gamma, scenario_name):
    """
    SIRモデルのシナリオ分析
    """
    print(f"\n{'='*60}")
    print(f"SIRモデル分析: {scenario_name}")
    print(f"{'='*60}")
    
    # パラメータ表示
    print(f"初期条件:")
    print(f"  S(0) = {S0:,} (感受性人口)")
    print(f"  I(0) = {I0:,} (感染者)")
    print(f"  R(0) = {R0:,} (回復者)")
    print(f"  総人口 = {S0+I0+R0:,}")
    print(f"\nパラメータ:")
    print(f"  β (感染率) = {beta:.4f}")
    print(f"  γ (回復率) = {gamma:.4f}")
    print(f"  感染期間 = {1/gamma:.1f} 日")
    
    # 基本再生産数
    R0_value = calculate_basic_reproduction_number(beta, gamma)
    print(f"  基本再生産数 R₀ = {R0_value:.3f}")
    
    if R0_value > 1:
        print(f"  → R₀ > 1: 感染は拡大します")
    else:
        print(f"  → R₀ < 1: 感染は収束します")
    
    # 数値解を求める
    sol = solve_sir_model(S0, I0, R0, beta, gamma, t_max=200)
    
    # 結果の分析
    max_infected_idx = np.argmax(sol.y[1])
    max_infected = sol.y[1][max_infected_idx]
    peak_time = sol.t[max_infected_idx]
    
    final_susceptible = sol.y[0][-1]
    final_recovered = sol.y[2][-1]
    
    print(f"\n分析結果:")
    print(f"  感染者数のピーク: {max_infected:,.0f} 人 (第 {peak_time:.1f} 日)")
    print(f"  最終的な感受性人口: {final_susceptible:,.0f} 人")
    print(f"  最終的な回復者数: {final_recovered:,.0f} 人") 
    print(f"  総感染率: {final_recovered/(S0+I0+R0)*100:.1f}%")
    
    # グラフ表示
    plot_sir_results(sol, scenario_name)
    
    return sol

# メイン実行部分
if __name__ == "__main__":
    print("SIRモデル（感染症数理モデル）シミュレーション")
    print("=" * 60)
    
    # シナリオ1: 基本的な感染拡大
    print("\n🦠 シナリオ1: 一般的なインフルエンザ様疾患")
    S0, I0, R0 = 99000, 1000, 0  # 総人口10万人
    beta = 0.3  # 感染率
    gamma = 0.1  # 回復率 (感染期間10日)
    
    sol1 = analyze_sir_scenario(S0, I0, R0, beta, gamma, "一般的なインフルエンザ")
    
    # シナリオ2: より感染力の強い疾患
    print("\n🦠 シナリオ2: 高感染力疾患 (COVID-19様)")
    beta2 = 0.5  # より高い感染率
    gamma2 = 0.067  # より長い感染期間 (15日)
    
    sol2 = analyze_sir_scenario(S0, I0, R0, beta2, gamma2, "高感染力疾患")
    
    # シナリオ3: 対策後の効果
    print("\n🦠 シナリオ3: 感染対策実施後")
    beta3 = 0.15  # 対策により感染率が半減
    gamma3 = 0.1   # 回復率は変わらず
    
    sol3 = analyze_sir_scenario(S0, I0, R0, beta3, gamma3, "感染対策実施後")
    
    # 比較プロット
    plt.figure(figsize=(15, 10))
    
    # 感受性人口の比較
    plt.subplot(2, 2, 1)
    plt.plot(sol1.t, sol1.y[0], 'b-', linewidth=2, label='シナリオ1 (通常)')
    plt.plot(sol2.t, sol2.y[0], 'r-', linewidth=2, label='シナリオ2 (高感染力)')
    plt.plot(sol3.t, sol3.y[0], 'g-', linewidth=2, label='シナリオ3 (対策後)')
    plt.xlabel('Time (days)')
    plt.ylabel('S (Susceptible)')
    plt.title('感受性人口の比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 感染者数の比較
    plt.subplot(2, 2, 2)
    plt.plot(sol1.t, sol1.y[1], 'b-', linewidth=2, label='シナリオ1 (通常)')
    plt.plot(sol2.t, sol2.y[1], 'r-', linewidth=2, label='シナリオ2 (高感染力)')
    plt.plot(sol3.t, sol3.y[1], 'g-', linewidth=2, label='シナリオ3 (対策後)')
    plt.xlabel('Time (days)')
    plt.ylabel('I (Infected)')
    plt.title('感染者数の比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 回復者数の比較
    plt.subplot(2, 2, 3)
    plt.plot(sol1.t, sol1.y[2], 'b-', linewidth=2, label='シナリオ1 (通常)')
    plt.plot(sol2.t, sol2.y[2], 'r-', linewidth=2, label='シナリオ2 (高感染力)')
    plt.plot(sol3.t, sol3.y[2], 'g-', linewidth=2, label='シナリオ3 (対策後)')
    plt.xlabel('Time (days)')
    plt.ylabel('R (Recovered)')
    plt.title('回復者数の比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 基本再生産数の比較
    plt.subplot(2, 2, 4)
    R0_values = [beta/gamma for beta, gamma in [(0.3, 0.1), (0.5, 0.067), (0.15, 0.1)]]
    scenarios = ['シナリオ1\n(通常)', 'シナリオ2\n(高感染力)', 'シナリオ3\n(対策後)']
    colors = ['blue', 'red', 'green']
    
    bars = plt.bar(scenarios, R0_values, color=colors, alpha=0.7)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2, label='R₀ = 1 (閾値)')
    plt.ylabel('基本再生産数 R₀')
    plt.title('基本再生産数の比較')
    plt.legend()
    
    # 各バーに値を表示
    for bar, value in zip(bars, R0_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("シミュレーション完了")
    print("="*60)