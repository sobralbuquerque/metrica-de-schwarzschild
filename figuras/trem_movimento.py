import texfig
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pylab


fontsize = '14'
N=100
# Data for plotting
# O = (0.0, 0.0)
w = 20.
h = 6.
t = np.linspace(0.0, 1.0, N)
dt = 0.1
theta = np.linspace(0.0, 2*np.pi, N)

# fig, ax = plt.subplots()
fig, ax = texfig.subplots()
ax.plot((t-0.5)*w, (t-0.5)*h, color='k', lw=0) # extremos da figura
ax.plot((t-0.5+dt)*w, (t-0.5)*h, color='k', lw=0) # extremos da figura2
rect = patches.Rectangle((-0.5*w, -0.5*h), w, h, linewidth=1.5, edgecolor='c', facecolor='none', alpha=0.4, ls='--', zorder=5) #retangulo
rect2 = patches.Rectangle(((-0.5+dt)*w, -0.5*h), w, h, linewidth=1.5, edgecolor='c', facecolor='none', alpha=0.7, zorder=0) #retangulo
ax.add_patch(rect)
ax.add_patch(rect2)

r=0.3
dr = 0.55
v = w*dt/10.0
# circulos
for i in range(0, 5):
    r = r + dr
    x = r*np.cos(theta)+v*(4-i)
    y = r*np.sin(theta)
    ax.plot(x, y, color='yellow', lw=1.5, alpha=(0.8-0.15*i))


# feixes de luz

light = (0.1+0.2*t)*w
ax.plot(light, 0*t, color='yellow', lw=2)
ax.plot(-light, 0*t, color='yellow', lw=2)

# setas
step_arrow = 0.05
plt.arrow(0.3*w, 0.0, step_arrow, 0.0, shape='full', length_includes_head=True, head_width=.45, color='yellow', zorder=9, alpha=1.0)
plt.arrow(-0.3*w, 0.0, -step_arrow, 0.0, shape='full', length_includes_head=True, head_width=.45, color='yellow', zorder=9, alpha=1.0)


# Velocidade
x2, y2 = [(0.3+0.1*t)*w, 0.65*h + 0*t]
ax.plot(x2, y2, color='darkblue', lw=2)
plt.arrow(0.42*w, 0.65*h, step_arrow, 0.0, shape='full', length_includes_head=True, head_width=.35, color='darkblue', zorder=9, alpha=1.0)
label = pylab.annotate(
    r"$\mathbf{v}$", color='darkblue',
    xy=(0.3*w, 0.65*h), xytext=(0.25*w,0.63*h), size=fontsize
)


# ax.set(xlabel='x', ylabel='ct ',
#     title='About as simple as it gets, folks')
# ax.grid()
plt.axis('off')
ax.set_aspect('equal')
plt.tight_layout()
fig.set_size_inches(4, 2)
# plt.show()
texfig.savefig('TremMovimento', transparent=True)