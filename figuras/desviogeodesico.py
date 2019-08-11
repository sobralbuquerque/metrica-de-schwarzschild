import texfig
import matplotlib.pyplot as plt
import numpy as np
import pylab


fontsize = '14'
N=100
# Data for plotting
t_max = 1.0

def f1(x):
    return np.sin(0.9*x+1.0)+0.3*x
def f2(x):
    return np.sin(0.9*x+0.9)+0.3*x + 0.18

t = np.linspace(0.0, t_max, N)
x = t
delta = 0.03
y1 = f1(t)
y2 = f2(t+delta)


# fig, ax = plt.subplots()
fig, ax = texfig.subplots()

ax.plot(x, y1, color='darkblue', lw=1.5, zorder=1)  # x^a
ax.plot(x, y2, color='darkblue', lw=1.5, zorder=1)  # ~x^a

# Vetores
t0 = 0.03
dist = 0.06
step_arrow = 0.05
t_medio = 0.6*t_max
for i in range(0,9):
    t0 = t0 + dist
    vec_x = t0 - delta*t
    vec_y = f1(t0) + t*(f2(t0)-f1(t0))
    ax.plot(vec_x, vec_y, color='darkblue', lw=1.5, zorder=0) # Linhas
    plt.arrow(                                                # Setas
        t0-delta*t_medio, f1(t0) + t_medio*(f2(t0)-f1(t0)),   # x, y
        -delta*step_arrow, (f2(t0)-f1(t0))*step_arrow,        # dx, dy
    shape='full', length_includes_head=True, head_width=.017, color='darkblue', zorder=10
    )



# Legendas
t_leg = 0.8*t_max
ax.text(t_leg, f1(t_leg)-0.05, r"$\gamma$", color='darkblue', size=fontsize) # gamma
ax.text(t_leg, f2(t_leg)+0.05, r"$\tilde{\gamma}$", color='darkblue', size=fontsize) # gamma til
t_leg = 1.02*t_max
ax.text(t_leg, f1(t_leg)-0.01, r"$x^a(u)$", color='darkblue', size=fontsize) # x^a
ax.text(t_leg, f2(t_leg)-0.01, r"$\tilde{x}^a(u)$", color='darkblue', size=fontsize) # x^a til

# Xi
x2, y2 = [t0, f1(t0) + (t_medio-0.05)*(f2(t0)-f1(t0))]
label = pylab.annotate(
    r"$ \xi^a(u) $", color='darkblue',
    xy=(x2, y2), xytext=(60,5.0), size=fontsize,
    textcoords='offset points', ha='right', va='bottom',
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
)



plt.axis('off')
ax.set_aspect('equal')
fig.set_size_inches(4, 4)
plt.tight_layout()
texfig.savefig('DesvioGeodesico', transparent=True)
# plt.show()
