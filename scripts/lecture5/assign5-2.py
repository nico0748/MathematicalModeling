import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def newton(x, f, Df, cond):
    i = 0
    while True:
        # Newton method
        DFinv_Fx = linalg.solve(Df(x), f(x))
        x = x - DFinv_Fx
        if cond(i, x, DFinv_Fx):
            break
        i += 1
    return x
def cond(i, x, DFinv_Fx):
    # Callback function
    return np.max(np.abs(DFinv_Fx)) <= 10**(-14)
# Df を追加!
def backward_euler(x0, tau, tf, f, Df):
    ti = 0
    xi = x0
    tlist = [ ti ]
    xlist = [ x0 ]
    I = np.eye(len(x0)) #単位行列
    while ti < tf:
        tip1 = ti + tau
        xip1 = newton(xi, lambda y: y - xi - tau * f(tip1, y), lambda y: I - tau * Df(tip1, y), cond)
        tlist.append(tip1)
        xlist.append(xip1)
        ti, xi = tip1, xip1
    return (tlist, xlist)

def backward_euler(x0, tau, tf, f, Df):
    ti = 0
    xi = x0
    tlist = [ ti ]
    xlist = [ x0 ]
    I = np.eye(len(x0)) #単位行列
    while ti < tf:
        tip1 = ti + tau
        xip1 = linalg.solve(I - tau * Df(ti, xi), xi)
        tlist.append(tip1)
        xlist.append(xip1)
        ti, xi = tip1, xip1

    return (tlist, xlist)

a = 1
b = 1
c = 1
d = 1
def strogatz( t, x ):
    fx = np.zeros_like(x)
    fx[0] = a * x[0] + b * x[1]
    fx[1] = c * x[0] + d * x[1]
    return fx

def strogatzDf( t, x ):
    Df = np.zeros(( len(x), len(x) ))
    Df[0, 0] = a
    Df[0, 1] = b
    Df[1, 0] = c
    Df[1, 1] = d
    return Df
x0 = np.zeros(2)
x0[0] = 1 
x0[1] = 0
t, x = backward_euler(x0, 0.1, 10, strogatz, strogatzDf)
plt.plot(t, x)
plt.show()
