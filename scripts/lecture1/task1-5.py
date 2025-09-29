x = 0
y = 0

h0 = 634

v0 = 30

g = 9.8
t = 0

i = 0
for i in range(115):
  y = h0 - (1/2)*g*(t**2)
  x = v0*t
  if y < 0:
    y = 0
  print('t = {t}  (x = {x}[m]  y = {y}[m])'.format(t = t, x = x, y = y))
  t += 0.1