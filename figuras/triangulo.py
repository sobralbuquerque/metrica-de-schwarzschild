import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
fontsize = '14'
N=100
# Data for plotting
v=0.5
t_max = 5.0
t = np.linspace(0.0, t_max, N)
s0 = v*t_max
s1 = 0*t + s0
s2 = v*t
t2 = np.linspace(0.0, v*t_max, N)
tt = 0*t
# s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(s1, t, color='darkblue', lw=2)
ax.plot(s2, t, color='darkblue', lw=2)
ax.plot(t2, tt, color='darkblue' , lw=2)



# # Legenda Luz 2
# x2, y2 = [s0+v*t_0, t_0]
# label = pylab.annotate(
#     r"$ t_0+dt$", color='darkblue',
#     xy=(x2, y2), xytext=(+70,+20), size=fontsize,
#     textcoords='offset points', ha='right', va='bottom',
#     arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
# )


# ax.set(xlabel='x', ylabel='ct ',
#     title='About as simple as it gets, folks')
# ax.grid()
plt.axis('off')
ax.set_aspect('equal')
fig.set_size_inches(4, 4)
plt.show()