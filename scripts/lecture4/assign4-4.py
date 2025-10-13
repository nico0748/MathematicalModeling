
import numpy as np
from scipy import linalg

def newton(x, f, Df, cond):
    i = 0
    while True:
        # Newton method
        DFinv_Fx = linalg.inv(Df(x)).dot(f(x))
        if cond(i, x, DFinv_Fx):
            break
        x = x - DFinv_Fx  # Newton法の更新式を追加
        i += 1
    return x

def f(x):
    # Problem
    fx = np.zeros((len(x)))
    fx[0] = 2*(x[0]**2) - np.exp(x[1])
    fx[1] = (-3*x[0])*(x[1]**2)-x[1]**3-3  # x[0]でスカラー値を取得
    return fx

def Df(p):
    # Jacobi matrix
    Dfp = np.zeros((len(p), len(p)))
    Dfp[0][0] = 4*p[0]  # p[0]でスカラー値を取得
    Dfp[0][1] = -np.exp(p[1])
    Dfp[1][0] = -3*(p[1]**2)
    Dfp[1][1] = -6*p[0] - 3
    return Dfp

def f2(x):
    # Problem
    fx = np.zeros((len(x)))
    fx[0] = 2*(x[0]**2) - np.exp(x[1])
    fx[1] = (-3*x[0])*(x[1]**2)-x[1]**3-3  # x[0]でスカラー値を取得
    return fx

def Df2(p):
    # Jacobi matrix
    Dfp = np.zeros((len(p), len(p)))
    Dfp[0][0] = 4*p[0]  # p[0]でスカラー値を取得
    Dfp[0][1] = -np.exp(p[1])
    Dfp[1][0] = -3*(p[1]**2)
    Dfp[1][1] = -6*p[0]*p[1] - p[1]**2
    return Dfp

def cond(i, x, DFinv_Fx):
    # Callback function
    #if DFinv_Fx < 1e-14:
        #return True
    #return False
    return np.max(np.abs(DFinv_Fx)) <= 10**(-14)

# Main:
x = np.zeros((2))
x[0] = 4
x[1] = 8
print(f"x0= {x[0]}, y0 = {x[1]}")
x = newton(x, f, Df, cond)
print(f"x = {x[0]}, y = {x[1]}")  # スカラー値として表示

# 発見した別解
# x0= 4.0, y0 = 3.0のときx = 0.28816607300191066, y = -1.795289464785593
