import texfig
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab


fontsize = '14'
N=100
# Data for plotting
t_max = 2*np.pi
a = 4.0
b = 2.0
P = [-1, -0.5]
t0 = 0.2
# beta = np.arctan(gamma)
t = np.linspace(0.0, t_max, N)
x = a*np.cos(t)
y = b*np.sin(t)

t1 = 0.2*np.pi
x_1 = a*np.cos(t1)
y_1 = b*np.sin(t1)
u = np.linspace(0.0, 1.0, N)
xi_1 = P[0] + u*(x_1 - P[0])
xi_2 = P[1] + u*(y_1 - P[1])

O = [a*np.cos(t0), b*np.sin(t0)]

xx = [O[0], P[0]]
yy = [O[1], P[1]]


# fig, ax = plt.subplots()
fig, ax = texfig.subplots()

ax.plot(x, y, color='darkblue', lw=1.5, zorder=1)  # elipse
ax.scatter(xx, yy, color='darkblue', s=20, zorder=5)    # pontos
ax.plot(xi_1, xi_2, color='darkblue', lw=1.5, zorder=0) # vetor
step_arrow = 0.05

# seta vetor
plt.arrow(P[0] + (1-step_arrow)*(x_1 - P[0]), P[1] + (1-step_arrow)*(y_1 - P[1]), (x_1 - P[0])*step_arrow, (y_1 - P[1])*step_arrow,
shape='full', length_includes_head=True, head_width=.15, color='darkblue', zorder=10
)

# Legendas

ax.text(P[0]-0.5, P[1]-0.5, r"$P$", color='darkblue', size=fontsize) # P
ax.text(O[0]+0.1, O[1]-0.1, r"$O$", color='darkblue', size=fontsize) # O 
G = [a*np.cos(0.7*np.pi), b*np.sin(0.7*np.pi)]
ax.text(G[0]-0.5, G[1]+0.3, r"$\gamma$", color='darkblue', size=fontsize) # gamma
XI = [(0.5)*(O[0]+P[0]), (0.5)*(O[1]+P[1])]
ax.text(XI[0]-0.3, XI[1]+0.7, r"$\xi^a$", color='darkblue', size=fontsize) # xi




plt.axis('off')
ax.set_aspect('equal')
fig.set_size_inches(4, 4)
# plt.show()
texfig.savefig('SmallLoop', transparent=True)
