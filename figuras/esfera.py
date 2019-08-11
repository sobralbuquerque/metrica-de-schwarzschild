import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pylab

scalefactor = 0.8  #1 para qualidade normal, 5 para compilação rápida
azim_init = 20  # ângulo azimutal inicial

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


# Create a sphere
r = 2
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:(scalefactor**(-1))*50j, 0.0:2.0*pi:(scalefactor**(-1))*50j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

#Pontos
phi_0 = 0.4*pi
P_0 = (r*sin(phi_0), 0.0, r*cos(phi_0))
phii = [0, phi_0, pi]

xx = r*sin(phii)*cos(0)
yy = r*sin(phii)*sin(0)
zz = r*cos(phii)

#Curvas
phi_1 = np.linspace(0, pi, int(50/scalefactor))
theta_1 = 0.0
phi_2 = phi_0
theta_2 = np.linspace(-pi/2-azim_init/180*pi, pi/2-azim_init/180*pi, int(50/scalefactor))
phi_3 = phi_0
theta_3 = np.linspace(-pi, pi, int(50/scalefactor))

x_1 = r*sin(phi_1)*cos(theta_1)
y_1 = r*sin(phi_1)*sin(theta_1)
z_1 = r*cos(phi_1)

x_2 = r*sin(phi_2)*cos(theta_2)
y_2 = r*sin(phi_2)*sin(theta_2)
z_2 = r*cos(phi_2)

x_3 = r*sin(phi_3)*cos(theta_3)
y_3 = r*sin(phi_3)*sin(theta_3)
z_3 = r*cos(phi_3)


#Ajustes da Imagem
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-azim_init, elev=7.5)  # vista da imagem
fig.set_size_inches(4, 4, 4)
cut = 0.65
ax.set_xlim([-r*cut,r*cut])
ax.set_ylim([-r*cut,r*cut])
ax.set_zlim([-r*cut,r*cut])


#Renderizar
ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0) #esfera

ax.scatter(xx, yy, zz, color="darkblue", s=20)  # pontos

plt.plot(x_1, y_1, z_1, color="darkblue", alpha=0.8, zorder=0)
plt.plot(x_2, y_2, z_2, color="darkblue", alpha=0.8, zorder=0)
plt.plot(x_3, y_3, z_3, color="darkblue", alpha=0.2, zorder=0)

print(x_1)

#Legendas

ax.text(P_0[0], P_0[1]-0.15*r, P_0[2]+0.035*r, r"$P_0$", color='darkblue', size=14) # P_0

ax.text(r*sin(phi_0+0.13)*cos(-0.2*pi), r*sin(phi_0+0.13)*sin(-0.2*pi), r*cos(phi_0+0.13),
        r"$ \theta=\theta_0$", color='darkblue', size=14) # theta

x2, y2, _ = proj3d.proj_transform(0, r*sin(pi-0.2), r*cos(pi-0.2), ax.get_proj())
label = pylab.annotate(
    r"$ \phi=0$", color='darkblue',
    xy=(x2, y2), xytext=(-15,10), size=14,
    textcoords='offset points', ha='right', va='bottom',
    # bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', color='darkblue')
)


#Vetores

beta = pi/6.
base = [[cos(phi_0), 0.0, -sin(phi_0)],                             # phi
        [0.0, 1.0, 0.0],                                            # theta
        [cos(beta)*cos(phi_0), sin(beta), -cos(beta)*sin(phi_0)]]   # lambda
base_label = [r'$\hat{e}_{\phi}$', r'$\hat{e}_{\theta}$', r'${\lambda}$']

for v in base:
    # plt.quiver(
    #     P_0[0], P_0[1], P_0[2],  # <-- starting point of vector
    #     v[0], v[1], v[2],  # <-- directions of vector
    #     color='k', alpha=1.0, lw=1.5,
    # )
    a = Arrow3D([P_0[0], P_0[0]+v[0]], [P_0[1], P_0[1]+v[1]],
                [P_0[2], P_0[2]+v[2]], mutation_scale=10,
                lw=1.5, arrowstyle="-|>", color="k",zorder=9)
    ax.add_artist(a)

#Ângulo Entre Vetores
rho = 1.0/3
t = np.linspace(0.0,np.inner(base[0],base[2]), 50)
angulo = vetor_ortogonal = [0, 0, 0]

for i in range(0,3):
    vetor_ortogonal[i] = base[2][i] - np.inner(base[2], base[0]) * base[0][i]
    angulo[i] = P_0[i] + rho*cos(t)*base[0][i] + rho*sin(t)*vetor_ortogonal[i]

plt.plot(angulo[0], angulo[1], angulo[2], color="red", alpha=0.9, lw=1.2, zorder=5)

#Legenda Vetores
for i in range(0,3):
    ax.text(P_0[0]+base[i][0]+0.05,P_0[1]+base[i][1]-0.04,P_0[2]+base[i][2]-0.15, base_label[i], color='k', zorder=10, size=14)
ax.text(P_0[0]+0.45*(base[0][0]+0.23*base[2][0]),
        P_0[1]+0.45*(base[0][1]+0.23*base[2][1]),
        P_0[2]+0.45*(base[0][2]+0.23*base[2][2]),
        r'$\alpha$', color='r', zorder=10, alpha=0.7, size=14)


# Plotar
plt.tight_layout()
fig.set_size_inches(4, 4, 4)
plt.axis('off')
plt.show()