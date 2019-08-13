import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
fontsize = '14'
N=100
# Data for plotting
v=0.4
t_max = 5.0
gamma = (1-v**2)**(-0.5)
# beta = np.arctan(gamma)
t = np.linspace(0.0, t_max, N)
s0=2
s1 = 0*t
light = t
ss = s0 + v*t
# s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(s1, t, color='k', lw=2)
ax.plot(ss, t, color='k', lw=2)

# Luz 1
t_0 = 0.8
dt_0 = 0.8
tt_max = s0 + v*t_0
tt = np.linspace(0.0, tt_max, N)

dist_0 = s0 + v*t_0
t_prime = np.linspace(t_0, tt_max+t_0, N)
li_1 = dist_0 - tt
ax.plot(li_1, t_prime, color='gold', zorder=0)

# Legenda Luz 1
x2, y2 = [s0+v*t_0, t_0]
label = pylab.annotate(
    r"$ t_0$", color='darkblue',
    xy=(x2, y2), xytext=(+45,+15), size=fontsize,
    textcoords='offset points', ha='right', va='bottom',
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
)

# Luz 2
tt_max = tt_max+v*dt_0
tt = np.linspace(0.0, tt_max, N)
t_0 = t_0 +dt_0
dist_0 = s0 + v*t_0
t_prime = np.linspace(t_0, tt_max+t_0, N)
li_1 = dist_0 - tt
ax.plot(li_1, t_prime, color='gold', zorder=0)

step_arrow = 0.05
plt.arrow(0, t_max, 0, step_arrow, shape='full', length_includes_head=True, head_width=.15, color='k', zorder=10)
plt.arrow(s0+v*t_max, t_max, v*step_arrow, step_arrow, shape='full', length_includes_head=True, head_width=.15, color='k', zorder=10)


# Legenda Luz 2
x2, y2 = [s0+v*t_0, t_0]
label = pylab.annotate(
    r"$ t_0+dt$", color='darkblue',
    xy=(x2, y2), xytext=(+70,+20), size=fontsize,
    textcoords='offset points', ha='right', va='bottom',
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
)


# ax.set(xlabel='x', ylabel='ct ',
#     title='About as simple as it gets, folks')
# ax.grid()
plt.axis('off')
ax.set_aspect('equal')
fig.set_size_inches(4, 4)
plt.show()