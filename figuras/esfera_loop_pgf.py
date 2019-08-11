import texfig
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pylab

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def annotate3d(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)


scalefactor = 1  #1 para qualidade normal, 5 para compilação rápida
azim_init = -50  # ângulo azimutal inicial
elev_init = 20  # angulo elev inicial
fontsize = '14'

# Create a sphere
r = 2
pi = np.pi
cos = np.cos
sin = np.sin
epsilon = 0.04*pi

#Esfera
phi, theta = np.mgrid[0.0:pi:(scalefactor**(-1))*50j, 0.0:2.0*pi:(scalefactor**(-1))*50j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

#Pontos
phi_0 = 0.4*pi
theta_0 = 0.4*pi
P_0 = (r*sin(phi_0)*cos(theta_0), r*sin(phi_0)*sin(theta_0), r*cos(phi_0))
phii = [0, phi_0, pi]
O_0 = (r*sin(phi_0-epsilon)*cos(theta_0-epsilon), r*sin(phi_0-epsilon)*sin(theta_0-epsilon), r*cos(phi_0-epsilon))
Q_0 = (r*sin(phi_0+epsilon)*cos(theta_0-epsilon), r*sin(phi_0+epsilon)*sin(theta_0-epsilon), r*cos(phi_0+epsilon))
R_0 = (r*sin(phi_0+epsilon)*cos(theta_0+epsilon), r*sin(phi_0+epsilon)*sin(theta_0+epsilon), r*cos(phi_0+epsilon))
S_0 = (r*sin(phi_0-epsilon)*cos(theta_0+epsilon), r*sin(phi_0-epsilon)*sin(theta_0+epsilon), r*cos(phi_0-epsilon))

xx_0 = [O_0[0], Q_0[0], R_0[0], S_0[0]]
yy_0 = [O_0[1], Q_0[1], R_0[1], S_0[1]]
zz_0 = [O_0[2], Q_0[2], R_0[2], S_0[2]]



xx = r*sin(phii)*cos(theta_0)
yy = r*sin(phii)*sin(theta_0)
zz = r*cos(phii)

#Curvas
phi_1 = np.linspace(0, pi, int(50/scalefactor))
theta_1 = [theta_0+epsilon, theta_0-epsilon]
phi_2 = [phi_0 + epsilon, phi_0 - epsilon]
theta_2 = np.linspace(-pi/2 - azim_init/180*pi, pi/2 - azim_init/180*pi, int(50/scalefactor))
theta_22 = np.linspace(-pi, pi, int(50/scalefactor))


x_11 = r*sin(phi_1)*cos(theta_1[0])
y_11 = r*sin(phi_1)*sin(theta_1[0])
z_11 = r*cos(phi_1)
x_21 = r*sin(phi_2[0])*cos(theta_2)
y_21 = r*sin(phi_2[0])*sin(theta_2)
z_21 = r*cos(phi_2[0])
x_12 = r*sin(phi_1)*cos(theta_1[1])
y_12 = r*sin(phi_1)*sin(theta_1[1])
z_12 = r*cos(phi_1)
x_22 = r*sin(phi_2[1])*cos(theta_2)
y_22 = r*sin(phi_2[1])*sin(theta_2)
z_22 = r*cos(phi_2[1])
xx_21 = r*sin(phi_2[0])*cos(theta_22)
yy_21 = r*sin(phi_2[0])*sin(theta_22)
zz_21 = r*cos(phi_2[0])
xx_22 = r*sin(phi_2[1])*cos(theta_22)
yy_22 = r*sin(phi_2[1])*sin(theta_22)
zz_22 = r*cos(phi_2[1])



#Ajustes da Imagem
fig = texfig.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-azim_init, elev=elev_init)  # vista da imagem
fig.set_size_inches(4, 4, 4)
cut = 0.65
ax.set_xlim([-r*cut,r*cut])
ax.set_ylim([-r*cut,r*cut])
ax.set_zlim([-r*cut,r*cut])

#Renderizar
ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.15, linewidth=0, zorder=0) #esfera

ax.scatter(xx, yy, zz, color="darkblue", s=20, zorder=1)  # pontos P + polos
ax.scatter(xx_0, yy_0, zz_0, color="darkblue", s=20, zorder=1)  # pontos O-Q-R-S

plt.plot(x_11, y_11, z_11, color="darkblue", alpha=0.8, zorder=0)
plt.plot(x_21, y_21, z_21, color="darkblue", alpha=0.8, zorder=0)
plt.plot(x_12, y_12, z_12, color="darkblue", alpha=0.8, zorder=0)
plt.plot(x_22, y_22, z_22, color="darkblue", alpha=0.8, zorder=0)
plt.plot(xx_21, yy_21, zz_21, color="darkblue", alpha=0.2, zorder=0, ls='--')
plt.plot(xx_22, yy_22, zz_22, color="darkblue", alpha=0.2, zorder=0, ls='--')

# #Setas
step_arrow=0.007
    # O Q
x_seta, y_seta, z_seta = [0.5*(Q_0[0]+O_0[0]), 0.5*(Q_0[1]+O_0[1]), 0.5*(Q_0[2]+O_0[2])]
x2, y2, _ = proj3d.proj_transform(x_seta, y_seta, z_seta, ax.get_proj())
plt.arrow(x2, y2, 0.05*step_arrow, -0.5*step_arrow,
shape='full', length_includes_head=True, head_width=.0035, color='darkblue', zorder=0, alpha=0.8
)
    # Q R
x_seta, y_seta, z_seta = [0.5*(Q_0[0]+R_0[0]), 0.5*(Q_0[1]+R_0[1]), 0.5*(Q_0[2]+R_0[2])]
x2, y2, _ = proj3d.proj_transform(x_seta, y_seta, z_seta, ax.get_proj())
plt.arrow(x2, y2, +0.45*step_arrow, +0.02*step_arrow,
shape='full', length_includes_head=True, head_width=.0035, color='darkblue', zorder=0, alpha=0.8
)
    # R S
x_seta, y_seta, z_seta = [0.5*(S_0[0]+R_0[0]), 0.5*(S_0[1]+R_0[1]), 0.5*(S_0[2]+R_0[2])]
x2, y2, _ = proj3d.proj_transform(x_seta, y_seta, z_seta, ax.get_proj())
plt.arrow(x2, y2, -0.05*step_arrow, +0.5*step_arrow,
shape='full', length_includes_head=True, head_width=.0035, color='darkblue', zorder=0, alpha=0.8
)
    # S O
x_seta, y_seta, z_seta = [0.5*(S_0[0]+O_0[0]), 0.5*(S_0[1]+O_0[1]), 0.5*(S_0[2]+O_0[2])]
x2, y2, _ = proj3d.proj_transform(x_seta, y_seta, z_seta, ax.get_proj())
plt.arrow(x2, y2, -0.45*step_arrow, -0.02*step_arrow,
shape='full', length_includes_head=True, head_width=.0035, color='darkblue', zorder=0, alpha=0.8
)


# #Legendas

ax.text(P_0[0]+0.1, P_0[1]-0.1, P_0[2]+0.02, r"$P$", color='darkblue', size=fontsize) # P
ax.text(S_0[0]+0.1, S_0[1]+0.2, S_0[2]+0.15, r"$S$", color='darkblue', size=fontsize) # S
ax.text(O_0[0]+0.2, O_0[1]-0.15, O_0[2]+0.05, r"$O$", color='darkblue', size=fontsize) # O
ax.text(Q_0[0]+0.2, Q_0[1]-0.2, Q_0[2]-0.32, r"$Q$", color='darkblue', size=fontsize) # Q
ax.text(R_0[0]+0.12, R_0[1]+0.2, R_0[2]-0.2, r"$R$", color='darkblue', size=fontsize) # R


# phi-menos
phi_leg = 0.2*pi
theta_leg = theta_0 - epsilon
x_leg = r*sin(phi_leg)*cos(theta_leg)
y_leg = r*sin(phi_leg)*sin(theta_leg)
z_leg = r*cos(phi_leg)
x2, y2, _ = proj3d.proj_transform(x_leg, y_leg, z_leg, ax.get_proj())
label = pylab.annotate(
    r"$ \phi=\phi_0-\varepsilon$", color='darkblue',
    xy=(x2, y2), xytext=(-15,10), size=fontsize,
    textcoords='offset points', ha='right', va='bottom',
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
)
# phi-mais
phi_leg = 0.7*pi
theta_leg = theta_0 + epsilon
x_leg = r*sin(phi_leg)*cos(theta_leg)
y_leg = r*sin(phi_leg)*sin(theta_leg)
z_leg = r*cos(phi_leg)
x2, y2, _ = proj3d.proj_transform(x_leg, y_leg, z_leg, ax.get_proj())
label = pylab.annotate(
    r"$ \phi=\phi_0+\varepsilon$", color='darkblue',
    xy=(x2, y2), xytext=(+75,-30), size=fontsize,
    textcoords='offset points', ha='right', va='bottom',
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
)
# theta-mais
phi_leg = phi_0 + epsilon
theta_leg = 0.1*pi
x_leg = r*sin(phi_leg)*cos(theta_leg)
y_leg = r*sin(phi_leg)*sin(theta_leg)
z_leg = r*cos(phi_leg)
x2, y2, _ = proj3d.proj_transform(x_leg, y_leg, z_leg, ax.get_proj())
label = pylab.annotate(
    r"$ \theta=\theta_0+\varepsilon$", color='darkblue',
    xy=(x2, y2), xytext=(-5,-45), size=fontsize,
    textcoords='offset points', ha='right', va='bottom',
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
)

# theta-menos
phi_leg = phi_0 - epsilon
theta_leg = 0.1*pi
x_leg = r*sin(phi_leg)*cos(theta_leg)
y_leg = r*sin(phi_leg)*sin(theta_leg)
z_leg = r*cos(phi_leg)  
x2, y2, _ = proj3d.proj_transform(x_leg, y_leg, z_leg, ax.get_proj())
label = pylab.annotate(
    r"$ \theta=\theta_0-\varepsilon$", color='darkblue',
    xy=(x2, y2), xytext=(-5,+35), size=fontsize,
    textcoords='offset points', ha='right', va='bottom',
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
)




# Plotar
plt.tight_layout()
fig.set_size_inches(4, 4, 4)
plt.axis('off')
texfig.savefig('EsferaLoop', transparent=True)