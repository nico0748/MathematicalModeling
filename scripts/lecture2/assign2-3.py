import numpy as np
from numpy import linalg as np_linalg
from scipy import linalg
print("課題3")

A = np.array([[2,1,3,4,5,-13],
              [1,3,4,5,6,7],
              [1,4,5,6,7,8],
              [4,5,6,7,9,11],
              [1,6,7,8,10,17],
              [1,7,8,9,11,21]])

b = np.array([1, 2, 3, 4, 5, 6])

Ainv = linalg.inv(A)
#x = np.dot(Ainv, b)
#x = np.dot(b, 1/A)
x = np.linalg.solve(A,b)

print(x)