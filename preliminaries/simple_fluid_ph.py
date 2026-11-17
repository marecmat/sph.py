from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys

sys.path.append("../src/")
import kernels
from ghost_particles_1Dlattice import make_ghosts
from time_integration import symplectic_step, integrate
from matplotlib import rcParams
rcParams['text.usetex'] = True




# core code format 
# class sph:
# 
#     def __init__(self, )
# 
# def make_boundaries()
# 
# def sph(rho, v, p, x):
# 
# def single_time_step():
# 
# def run_case()    


@dataclass
class parameters():
    H: float = 1            # fluid height 
    N: int = 100            # number of particles
    rho0: float = 1000      # fluid density
    g: float = 9.81         # gravity acceleration
    drho_max: float = .01   # max admissible density variation  (WCSPH)
    gamma: float = 7        # isentropic expansion factor TODO: not sure why
    alpha: float = .01      # artificial viscosity parameter TODO: verify why

    @property       # particle spacing
    def dy(self):   return self.H/self.N
    @property       # numerical speed of sound 
    def c0(self):   return np.sqrt(self.g*self.H/self.drho_max)
    @property       # bulk modulus
    def B(self):    return self.rho0/self.gamma*self.c0**2
    @property       # smoothing length
    def h(self):    return 2*self.dy 
    @property
    def mass(self): return self.rho0*self.dy


def W(r, h):
    # Wendland C2
    q = np.abs(r)/h
    return np.where(q < 2.0, 3./(4.*h)*(1 - 0.5*q)**4*(2*q+1), 0.0)

def dWdr(r, h):
    q = np.abs(r)/h
    return np.where(q < 2.0, 3./(4.*h)*(-5.*q)/h*(1 - 0.5*q)**3, 0.0)


def core_equations(pos, vel, dens, params)
    # TODO implement boundary conditions here ?
    i, j = pairs of stuff using cKDTree
    r = xvals[i] - xvals[j]
    dW = dWdr(np.abs(r), params.h)

    pos_a = 
    PI_ij = ???
    pres_a = params.B*((dens_a/params.rho0)**params.gamma - 1)
    drho_a = params.mass*(vel_a[i] - vel_a[j])*dW*np.sign(r)
    dvel_a = -1*params.mass*((pres_a[i]/dens_a[i]**2) - (pres_a[j]/dens_a[j]**2) + PI_ij)*dW*np.sign(r) - params.g
    return ???

def symplectic_step():
    return None 


if __name__ == "__main__":
    p = parameters()