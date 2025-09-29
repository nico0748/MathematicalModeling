import numpy as np
A = np.zeros((3, 4))
print(A)

B = A #BにAのポインタ渡し→同様のアドレスを参照しているので、Bを変更するとAも変更される
B[0, 0] = 99

print(B)

print(A)

np.copy


#[重要]配列とリストの違いについて