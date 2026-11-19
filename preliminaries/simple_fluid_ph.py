from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import sys
sys.path.append("../src/")

from kernels import wendland, wendland_gradient
from ghost_particles_1Dlattice import make_ghosts
from time_integration import symplectic_step

from tqdm import tqdm

@dataclass
class parameters():
    dim: int                  = 1             # physical dimension
    H: float                  = 1             # fluid height 
    N: int                    = 100           # number of particles
    T: float                  = 5.0           # end time of simulation
    rho0: float               = 1000          # fluid density
    g: float                  = 9.81          # gravity acceleration
    drho_max: float           = .01           # max admissible density variation  (WCSPH)
    gm: float                 = 7             # isentropic expansion factor TODO: not sure why
    alpha: float              = .01           # artificial viscosity parameter
    delta: float              = .1            # density diffusion coefficient
    n_wall: int               = 4             # wall layers (>= 2h/dy)
    cfl: float                = .25           # CFL Number
    t_relax: float            = 1.0           # relaxation time
    damp: float               = .01           # damping
    start: float              = 'hydrostatic' # particle profile ['uniform', 'hydrostatic']
    kernel_gradient: callable = wendland_gradient
    kernel: callable          = wendland

    @property       # particle spacing
    def dy(self):   return self.H/self.N
    @property       # numerical speed of sound 
    def c0(self):   return np.sqrt(self.g*self.H/self.drho_max)
    @property       # bulk modulus
    def B(self):    return self.rho0/self.gm*self.c0**2
    @property       # smoothing length
    def h(self):    return 2*self.dy 
    @property
    def mass(self): return self.rho0*self.dy
    @property       # effective viscosity
    def nu(self):   return self.alpha*self.h*self.c0/6
    @property       # time step
    def dt(self):   return self.cfl*min([self.h/self.c0, self.h**2 / self.nu])
    @property       # number of time steps
    def n(self):    return int(np.ceil(self.T/self.dt))    
    @property
    def kp(self):   return (self.gm - 1)*self.rho0*self.g/(self.B*self.gm)
    @property
    def ub(self):   return (1 + self.H*self.kp*self.gm/\
                                (self.gm - 1))**((self.gm - 1)/self.gm)
    @property       # surface height
    def ys(self):   return (self.ub - 1)/self.kp 
    
    def pressure(self, y):
        return self.B*((1 + self.kp*(self.ys - y))**(self.gm/(self.gm - 1)) - 1)
    
    def rho_hydro(self, y): # equilibrium density at height y
        return self.rho0*(1 + self.pressure(y)/self.B)**(1/self.gm)


def make_particles(p, start="hydrostatic"):
    k = np.arange(p.N)
    yw = -(np.arange(p.n_wall) + 0.5) * p.dy
    if start == "uniform":
        yf = (k + 0.5) * p.dy
        rf = np.full(p.N, p.rho0)
    else:
        u = (1 + p.kp*p.gm*(p.N - 1 - k + 0.5)*p.dy/(p.gm - 1))**((p.gm - 1)/p.gm)
    x = np.concatenate([yw, p.ys - (u - 1)/p.kp])
    v = np.zeros_like(x)

    rho = np.concatenate([np.full(p.n_wall, p.rho0), p.rho0*u**(1/(p.gm - 1))])
    wall = np.concatenate([np.ones(p.n_wall, bool), np.zeros(p.N, bool)])
    return x, v, rho, wall


# ------------------------------------------------------------ SPH right-hand side

def func(p, wall, x, v, rho, t):
    i, j = cKDTree(x[:, None]).query_pairs(2 * p.h, output_type="ndarray").T
    r = x[i] - x[j]
    n = len(wall)
    Wij = p.kernel(r, p.h, dim=p.dim)
    dW = p.kernel_gradient(r, p.h, dim=p.dim) # dW/dx_i

    pres = p.B * ((rho / p.rho0) ** p.gm - 1)
    dens = rho.copy()

    # wall pressure from fluid (Adami):  p_w = [sum p_f W + g.sum rho_f r_wf W] / sum W
    sel = wall[i] != wall[j]
    wi = np.where(wall[i], i, j)[sel]
    fi = np.where(wall[i], j, i)[sel]
    Ws = Wij[sel]
    r_wf = x[wi] - x[fi]
    sw = np.bincount(wi, Ws, n)
    num = np.bincount(wi, pres[fi] * Ws, n) \
        + np.bincount(wi, -p.g * rho[fi] * r_wf * Ws, n)     # gravity = -g (y)
    ok = wall & (sw > 0)
    pw = np.zeros(n)
    pw[ok] = num[ok] / sw[ok]
    pres[wall] = pw[wall]
    dens[wall] = p.rho0 * np.maximum(1 + pw[wall] / p.B, 1e-3) ** (1 / p.gm)

    vij = v[i] - v[j]
    c = p.mass*vij*dW # same contribution to i and j
    drho = np.bincount(i, c, n) + np.bincount(j, c, n)

    # delta-SPH density diffusion, hydrostatic-corrected (Fourtakas et al. 2019 idea):
    # diffuse only the deviation from the equilibrium density profile
    if p.delta > 0:
        ff = ~wall[i] & ~wall[j]
        psi = (dens[i] - dens[j]) - (p.rho_hydro(x[i]) - p.rho_hydro(x[j]))
        D = 2 * p.delta * p.h * p.c0 * psi * dW / r * ff
        drho += np.bincount(i, p.mass / dens[j] * D, n) \
                + np.bincount(j, p.mass / dens[i] * D, n)

    vr = vij * r
    mu = p.h * vr / (r**2 + 0.01 * p.h**2)
    Pi = np.where(vr < 0, -p.alpha * p.c0 * mu / (0.5 * (dens[i] + dens[j])), 0.0)
    f = -p.mass * (pres[i] / dens[i]**2 + pres[j] / dens[j]**2 + Pi) * dW
    acc = np.bincount(i, f, n) - np.bincount(j, f, n) - p.g
    if t < p.t_relax:
        acc -= p.damp * v

    acc[wall] = 0.0
    drho[wall] = 0.0
    return acc, drho


def run_case(p, n_out=200):
    x, v, rho, wall = make_particles(p, p.start)

    data = {"t": [], "vmax": [], "KE": []}

    for s in tqdm(range(p.n)):
        x, v, rho = symplectic_step(p, func, x, v, rho, wall, p.dt, s*p.dt)

        if s % max(1, p.n//n_out) == 0:
            data["t"].append((s + 1) * p.dt)
            data["vmax"].append(np.abs(v).max())
            data["KE"].append(0.5 * p.mass * np.sum(v[~wall]**2))
    data = {k: np.array(a) for k, a in data.items()}
    return x, v, rho, wall, data



if __name__ == "__main__":
    params = parameters(alpha=0.1, delta=0.1, T=3., start="hydrostatic")
    x, v, rho, wall, hist = run_case(params)

    yf = x[~wall] 
    pn = params.B * ((rho[~wall] / params.rho0) ** params.gm - 1)
    p_exact = params.pressure
    err = (pn - p_exact(x[~wall])) / (params.rho0 * params.g * params.H)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
    ax[0].plot(yf, pn/p_exact(x[~wall]), ".", ms=3, label='')
    ax[1].plot(yf, err, label='')
    ax[2].semilogy(hist["t"], hist["KE"] + 1e-30, label='')
    ax[0].set(xlabel="y", ylabel="p"); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="y", ylabel="(p - p_exact)/(rho g H)")
    ax[2].set(xlabel="t", ylabel="kinetic energy")

    plt.show()