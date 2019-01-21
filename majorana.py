import numpy as np
from cmath import sqrt 
import qutip as qt
from operators import *

tol = 1e-16

def solvePoly(vec):
    roots = np.empty(2, dtype=np.complex128)
    vec[1]=2*vec[1]
    if abs(vec[0]) <= tol:
        roots[0] = np.inf
        if abs(vec[1]) <= tol:
            roots[1] = np.inf
        else:
            roots[1] = -vec[2]/vec[1]
    else:
        roots[0] = -0.5*(vec[1]+sqrt(vec[1]**2-4*vec[0]*vec[2]))/vec[0]
        roots[1] = -vec[1]/vec[0] - roots[0]
    return roots


def root_to_xyz(root):
    if root == np.inf:
        return [0,0,1]
    x = root.real
    y = root.imag
    den = 1/(1.+(x**2)+(y**2))
    return [2*x*den,2*y*den, (1.-(x**2)+(y**2))*den]


def getStars(vec):
    #converts 3-spinor into two stars
    roots = np.empty(2, dtype=np.complex128)
    stars = [[],[],[]] #stores x, y and z coordinates
    vec[1] *= -np.sqrt(2)
    if abs(vec[0]) <= tol:
        roots[0] = np.inf
        if abs(vec[1]) <= tol:
            roots[1] = np.inf
        else:
            roots[1] = -vec[2]/vec[1]
    else:
        roots[0] = -0.5*(vec[1] + sqrt(vec[1]**2-4*vec[0]*vec[2]))/vec[0]
        roots[1] = -vec[1]/vec[0] - roots[0]

    for r in roots:
        if r == np.inf:
            stars[0].append(0)
            stars[1].append(0)
            stars[2].append(-1)            
        else:
            x = r.real
            y = r.imag
            den = 1/(1.+(x**2)+(y**2))
            stars[0].append(2*x*den)    
            stars[1].append(2*y*den)
            stars[2].append((1.-(x**2)-(y**2))*den)    
    return stars

print(getStars([1,0,1]))


b = qt.Bloch()
b.point_color = ['b','b','r','r','g','g','#CC6600','#CC6600'] #ensures point and line are same colour
b.add_points(getStars([1,sqrt(2),1]))
#b.add_points(getStars([1/sqrt(2),0,1/sqrt(2)]),meth='l')
b.xlabel = ['$<F_x>$','']
b.ylabel = ['$<F_y>$','']
b.zlabel = ['$<F_z>$','']
#b.add_points([[0,0],[-1,1],[0,0]], meth='l')
#b.add_points([[-1,1],[0,0],[0,0]], meth='l')

#b.add_points([0,0])
#b.add_points([0,0,-1])
b.show()
