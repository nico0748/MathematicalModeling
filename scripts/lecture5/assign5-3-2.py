# ===============================================
# 蕨市（埼玉県）の人口成長モデル（前進オイラー法）
# ===============================================
# 対象：
#   - 2020-10-01 国勢調査人口：74,283人
#   - 2024-02-01 推定人口：75,614人
#
# 目的：
#   - 指数成長モデルとロジスティックモデルを
#     前進オイラー法で数値的に解き、将来予測する。
#   - 20年間（～2044年）をシミュレーション。
#
# 出力：
#   - グラフ（指数成長 vs ロジスティック）
#   - 年ごとの人口表
# ===============================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

y_2020_10_01 = 74283  # 国勢調査 2020-10-01
t0 = 2020 + (10 - 1) / 12.0  # 年小数表現（10月 → +9/12）
y_2024_02_01 = 75614  # 推定人口 2024-02-01
t1 = 2024 + (2 - 1) / 12.0  # 年小数表現（2月 → +1/12）

elapsed_years = t1 - t0  # 約3.33年
r_est = math.log(y_2024_02_01 / y_2020_10_01) / elapsed_years
print(f"推定成長率 r = {r_est:.6f} /年 （約 {r_est*100:.3f}%/年）")
print(f"観測期間：{elapsed_years:.3f} 年（2020-10-01 → 2024-02-01）\n")

y0 = y_2024_02_01  # 初期人口（2024-02-01）
T_years = 20        # 20年間先まで計算
h = 0.02            # 刻み幅（年）≈ 7.3日
N = int(T_years / h)
t = np.linspace(0, T_years, N + 1)

def euler_exponential(y0, r, h, N):
    """ dy/dt = r*y"""
    y = np.zeros(N + 1)
    y[0] = y0
    for n in range(N):
        y[n + 1] = y[n] + h * (r * y[n])
    return y

def euler_logistic(y0, r, K, h, N):
    """ロジスティックモデル dy/dt = r*y*(1 - y/K)"""
    y = np.zeros(N + 1)
    y[0] = y0
    for n in range(N):
        y[n + 1] = y[n] + h * (r * y[n] * (1 - y[n] / K))
    return y

# シミュレーション実行 
y_exp = euler_exponential(y0, r_est, h, N)
Ks = [80000, 90000, 100000]  # 仮の環境収容力
ys_log = {K: euler_logistic(y0, r_est, K, h, N) for K in Ks}

# グラフ表示
plt.figure(figsize=(9, 5))
plt.plot(t, y_exp, label=f"exponential growth (r={r_est:.4%}/year)")
for K in Ks:
    plt.plot(t, ys_log[K], label=f"Logistic (K={K:,})")

plt.title("Warabi City population growth model")
plt.xlabel("Years since 2024")
plt.ylabel("Population")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 7. 年ごとの人口一覧 ---
years = np.arange(0, 21)  # 0〜20年
indices = (years / h).astype(int)

table = pd.DataFrame({
    "経過年数（年）": years,
    "指数成長": y_exp[indices].round().astype(int)
})
for K in Ks:
    table[f"ロジスティック(K={K:,})"] = ys_log[K][indices].round().astype(int)

print("=== 蕨市の人口予測（前進オイラー法による数値解） ===")
print(table.to_string(index=False))

