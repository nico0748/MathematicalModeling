import matplotlib.pyplot as plt
import numpy as np

h = 0
h0 = 634

g = 9.8
t = 0

i = 0
checker = True
while checker == True:
  h = h0 - (1/2)*g*(t**2)
  if h < 0:
    h = 0
    checker = False
  plt.plot(t,h,marker='.')
  plt.xlabel('Time [s]')
  plt.ylabel('Height [m]')
  plt.title('Free Fall')
  plt.grid()
  t += 0.1

plt.show()