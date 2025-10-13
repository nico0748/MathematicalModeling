
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
    fx[0] = x[0]**2 - 2  # x[0]でスカラー値を取得
    return fx

def Df(p):
    # Jacobi matrix
    Dfp = np.zeros((len(p), len(p)))
    Dfp[0][0] = 2*p[0]  # p[0]でスカラー値を取得
    return Dfp

def cond(i, x, DFinv_Fx):
    # Callback function
    #if DFinv_Fx < 1e-14:
        #return True
    #return False
    return np.max(np.abs(DFinv_Fx)) <= 10**(-14)

# Main:
x = np.zeros((1))
x[0] = 2
x = newton(x, f, Df, cond)
print(x[0])  # スカラー値として表示
