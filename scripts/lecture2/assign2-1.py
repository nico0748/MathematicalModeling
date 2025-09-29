#課題 1: 下記のプログラムを元に 2 パターン行列積の作成せよ
#1) Python のリストと for 文を用いて 2 次元配列の行列積を作成せよ
#2) NumPy を用いた行列積を作成せよ

import time
import numpy as np
from numpy import linalg as np_linalg
from scipy import linalg

n = 100
A = [[float(i + j) for j in range(n)] for i in range(n)]
B = [[float(i + j) for j in range(n)] for i in range(n)]
C = [[float(0) for j in range(n)] for i in range(n)]

C =np.dot(A,B)

start_time = time.perf_counter() #開始時間の取得
##############################################
######### Python の 2 次元リストによる行列積 ##########
##############################################
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i][j] += A[i][k] * B[k][j]

##############################################
end_time = time.perf_counter() #終了時間の取得
#print(C)
print("2次元リストによる行列積の実行時間 :",end_time - start_time, " [sec]")

A = np.array(A)
B = np.array(B)
start_time = time.perf_counter() #開始時間の再取得
##############################################
############## Numpy による行列積 ###############
##############################################
C = np.dot(A, B)
##############################################
end_time = time.perf_counter() #終了時間の再取得
#print(C)
print("Numpy による行列積の実行時間 :",end_time - start_time, " [sec]")
##############################################
