h = 0
h0 = 634

g = 9.8
t = 0

i = 0
for i in range(115):
  h = h0 - (1/2)*g*(t**2)
  v= g*t
  if h < 0:
    h = 0
  print('t = {t}[s]  h = {h}[m]  v = {v}[m/s]'.format(t = t, h = h, v = v))
  t += 0.1