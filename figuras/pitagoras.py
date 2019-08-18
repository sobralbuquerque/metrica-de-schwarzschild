import texfig
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab

fontsize = '24'
N=100

# Data for plotting
v=0.55
t_max = 5.0
t_medio = t_max*0.5
t = np.linspace(0.0, t_max, N)
s0 = v*t_max
s1 = 0*t + s0
s2 = v*t
t2 = np.linspace(0.0, v*t_max, N)
tt = 0*t

fig, ax = texfig.subplots()
ax.plot(s1, t, color='darkblue', lw=2)
ax.plot(s2, t, color='darkblue', lw=2)
ax.plot(t2, tt, color='darkblue' , lw=2)
ax.plot(t2, tt-0.2, color='darkblue' , lw=0) # ajuste de margem


# cdt
label = pylab.annotate(
    r"$c\Delta t$", color='darkblue',
    xy=(v*t_medio-0.8, t_medio+0.0), size=fontsize
)

# vdt
label = pylab.annotate(
    r"$v\Delta t$", color='darkblue',
    xy=(v*t_medio-0.1, -0.45), size=fontsize
)

# cdt'
label = pylab.annotate(
    r"$c\Delta t^\prime$", color='darkblue',
    xy=(s0+0.1, t_medio*0.8), size=fontsize
)


# # Legenda Luz 2
# x2, y2 = [s0+v*t_0, t_0]
# label = pylab.annotate(
#     r"$ t_0+dt$", color='darkblue',
#     xy=(x2, y2), xytext=(+70,+20), size=fontsize,
#     textcoords='offset points', ha='right', va='bottom',
#     arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
# )



plt.axis('off')
ax.set_aspect('equal')
fig.set_size_inches(4, 4)
# plt.show()
texfig.savefig('Pitagoras', transparent=True)