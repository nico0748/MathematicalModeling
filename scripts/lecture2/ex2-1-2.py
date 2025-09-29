import numpy as np
A = np.zeros((3, 4))
print(A)

B = np.copy(A) #BにAのコピーを渡し→異なるアドレスを参照しているので、Bを変更してもAは変更されない
B[0, 0] = 99

print(B)

print(A)
