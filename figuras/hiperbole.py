import texfig
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab


fontsize = '14'
N=100

# Data for plotting

t_max = 1.2
rho = 1
beta = 0.3*t_max
t0 = 0.2
t = np.linspace(-t_max, t_max, N)
x = rho*np.cosh(t)
y = rho*np.sinh(t)
nulo = 0*t
A = [rho*np.sinh(beta), rho*np.cosh(beta)]
B = [rho*np.cosh(beta), rho*np.sinh(beta)]

xx = [A[0], B[0]]
yy = [A[1], B[1]]


# fig, ax = plt.subplots()
fig, ax = texfig.subplots()

ax.plot(x, y, color='darkblue', lw=1.5, zorder=1)  # hiperbole 1
ax.plot(y, x, color='darkblue', lw=1.5, zorder=1)  # hiperbole 2
ax.plot(y, y, color='darkblue', lw=1, ls='--', zorder=1, alpha=0.5)  # assintota 1
ax.plot(-y, y, color='darkblue', lw=1, ls='--', zorder=1, alpha=0.5)  # assintota 2
ax.plot(y, y*np.tanh(beta), color='k', lw=1, ls='--',zorder=1, alpha=0.5)  # eixo x'
ax.plot(y*np.tanh(beta), y, color='k', lw=1, ls='--',zorder=1, alpha=0.5)  # eixo y'
ax.plot(y, nulo, color='k', lw=1, zorder=1)  # eixo x
ax.plot(nulo, y, color='k', lw=1, zorder=1)  # eixo y
ax.scatter(xx, yy, color='darkblue', s=20, zorder=5)    # pontos

# Ângulos

r = 0.75
tt = np.linspace(0.0, beta, N)
xx = r*rho*np.cosh(tt)
yy = r*rho*np.sinh(tt)
ax.plot(yy, xx, color='darkblue', lw=1, zorder=1) #BETA A
ax.plot(xx, yy, color='darkblue', lw=1, zorder=1) #BETA B





# Legendas

ax.text(A[0]+0.1, A[1]-0.15, r"$A$", color='darkblue', size=fontsize) # A
ax.text(B[0]-0.15, B[1]+0.1, r"$B$", color='darkblue', size=fontsize) # B
ax.text(0.5*np.exp(t_max), 0.0, r"$x$", color='k', size=12) # x
ax.text(0.0, 0.5*np.exp(t_max), r"$y$", color='k', size=12) # y
ax.text(0.5*np.exp(t_max), 0.5*np.exp(t_max)*np.tanh(beta), r"$x^\prime$", color='k', size=12) # x'
ax.text(0.5*np.exp(t_max)*np.tanh(beta), 0.5*np.exp(t_max), r"$y^\prime$", color='k', size=12) # y'
ax.text(0.5*A[0]-0.13, 0.5*A[1], r"$\beta$", color='darkblue', size=fontsize) # Beta A
ax.text(0.5*B[0], 0.5*B[1]-0.13, r"$\beta$", color='darkblue', size=fontsize) # Beta B


plt.axis('off')
ax.set_aspect('equal')
fig.set_size_inches(4, 4)
# plt.show()
texfig.savefig('Hiperbole', transparent=True)
