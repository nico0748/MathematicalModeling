import matplotlib.pyplot as plt
import numpy as np


x = 0
y = 0

h0 = 634

v0 = 30

g = 9.8
t = 0

i = 0
Checker  = True

while Checker == True:
  x = v0*t
  y = h0 - (1/2)*g*(t**2)
  if y < 0:
    y = 0
    Checker = False
    
  plt.plot(x,y,marker='.') 
  plt.xlabel('x [m]')
  plt.ylabel('y [m]') 
  plt.title('Projectile Motion')
  plt.grid()
  t += 0.1

plt.show()