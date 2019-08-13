import texfig
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pylab

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
fontsize = '14'
N=100
# Data for plotting
# O = (0.0, 0.0)
w = 20.
h = 6.
t = np.linspace(0.0, 1.0, N)
theta = np.linspace(0.0, 2*np.pi, N)


# fig, ax = plt.subplots()
fig, ax = texfig.subplots()
ax.plot((t-0.5)*w, (t-0.5)*h, color='k', lw=0) # extremos da figura
rect = patches.Rectangle((-0.5*w, -0.5*h), w, h, linewidth=1.5, edgecolor='darkblue', facecolor='none', alpha=0.7) #retangulo
ax.add_patch(rect)


r=0.3
dr = 0.5
# circulos
for i in range(0, 5):
    r = r + dr
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    ax.plot(x, y, color='gold', lw=1.5, alpha=(0.8-0.15*i))


# feixes de luz

light = (0.1+0.2*t)*w
ax.plot(light, 0*t, color='gold', lw=2)
ax.plot(-light, 0*t, color='gold', lw=2)

# setas
step_arrow = 0.05
plt.arrow(0.3*w, 0.0, step_arrow, 0.0, shape='full', length_includes_head=True, head_width=.45, color='gold', zorder=9, alpha=1.0)
plt.arrow(-0.3*w, 0.0, -step_arrow, 0.0, shape='full', length_includes_head=True, head_width=.45, color='gold', zorder=9, alpha=1.0)





# ax.set(xlabel='x', ylabel='ct ',
#     title='About as simple as it gets, folks')
# ax.grid()
plt.axis('off')
ax.set_aspect('equal')
plt.tight_layout()
fig.set_size_inches(4, 2)
texfig.savefig('trem', transparent=True)