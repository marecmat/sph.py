from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import sys
sys.path.append("../src/")

from kernels import wendland, wendland_gradient
from ghost_particles_1Dlattice import make_ghosts
from time_integration import symplectic, rk4, integrate

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
    # t_relax: float            = 1.0           # relaxation time
    # damp: float               = .01           # damping
    start: float              = 'hydrostatic' # particle profile ['uniform', 'hydrostatic']
    kernel_gradient: callable = wendland_gradient
    kernel: callable          = wendland
    time_stepper: callable    = symplectic
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
    @property
    def wall(self): return np.concatenate(
        [np.ones(self.n_wall, bool), np.zeros(self.N, bool)])

    def analytical_pressure(self, y):
        return self.B*((1 + self.kp*(self.ys - y))**(self.gm/(self.gm - 1)) - 1)
    
    def rho_hydro(self, y): # equilibrium density at height y
        return self.rho0*(1 + self.analytical_pressure(y)/self.B)**(1/self.gm)

    def compute_pressure(self, rho):
        return tait_eq(self.B, rho[-1, ~self.wall], self.rho0, self.gm)

    def relative_error(self, coord, rho):
        return np.abs((self.compute_pressure(rho) - \
            self.analytical_pressure(coord)))/self.analytical_pressure(coord) #/(self.rho0*self.g*params.H))


def tait_eq(B, rho, rho0, gamma):
    return B*((rho/rho0)**gamma - 1)


def make_particles(p, start="hydrostatic"):
    k = np.arange(p.N)
    yw = -(np.arange(p.n_wall) + 0.5) * p.dy
    if start == "uniform":
        yf = (k + 0.5) * p.dy
        rf = np.full(p.N, p.rho0)
    else:
        u = (1 + p.kp*p.gm*(p.N - 1 - k + 0.5)*p.dy/(p.gm - 1))**((p.gm - 1)/p.gm)
        yf = p.ys - (u - 1)/p.kp
        rf = p.rho0*u**(1/(p.gm - 1))
    x = np.concatenate([yw, yf])
    v = np.zeros_like(x)

    rho = np.concatenate([np.full(p.n_wall, p.rho0), rf])
    return x, v, rho


def build_equations(p, x, v, rho, t):
    # pair searching 
    i, j = cKDTree(x[:, None]).query_pairs(2*p.h, output_type="ndarray").T
    r = x[i] - x[j]
    n = len(p.wall)
    
    Wij  = p.kernel(r, p.h, dim=p.dim)
    dW   = p.kernel_gradient(r, p.h, dim=p.dim) # dW/dx_i
    pres = tait_eq(p.B, rho, p.rho0, p.gm)
    dens = rho.copy()

    # wall pressure from fluid (Adami):  p_w = [sum p_f W + g.sum rho_f r_wf W] / sum W
    wall_fluid_pairs = p.wall[i] != p.wall[j]
    where_wall_i     = np.where(p.wall[i], i, j)[wall_fluid_pairs]
    where_wall_j     = np.where(p.wall[i], j, i)[wall_fluid_pairs]
    r_wf = x[where_wall_i] - x[where_wall_j]
    sw   = np.bincount(where_wall_i, Wij[wall_fluid_pairs], n)
    num  = np.bincount(where_wall_i, pres[where_wall_j]*Wij[wall_fluid_pairs], n) \
         + np.bincount(where_wall_i, -p.g*rho[where_wall_j]*r_wf*Wij[wall_fluid_pairs], n)
    
    ok           = p.wall & (sw > 0)
    pw           = np.zeros(n)
    pw[ok]       = num[ok]/sw[ok]
    pres[p.wall] = pw[p.wall]
    dens[p.wall] = p.rho0*np.maximum(1 + pw[p.wall]/p.B, 1e-3)**(1/p.gm)

    vij  = v[i] - v[j]
    c    = p.mass*vij*dW # same contribution to i and j
    drho = np.bincount(i, c, n) + np.bincount(j, c, n)

    # delta-SPH density diffusion, hydrostatic-corrected (Fourtakas et al. 2019 idea):
    # diffuse only the deviation from the equilibrium density profile
    if p.delta > 0:
        ff = ~p.wall[i] & ~p.wall[j]
        psi = (dens[i] - dens[j]) - (p.rho_hydro(x[i]) - p.rho_hydro(x[j]))
        D = 2 * p.delta * p.h * p.c0 * psi * dW / r * ff
        drho += np.bincount(i, p.mass / dens[j] * D, n) \
                + np.bincount(j, p.mass / dens[i] * D, n)

    vr = vij * r
    mu = p.h * vr / (r**2 + 0.01 * p.h**2)
    Pi = np.where(vr < 0, -p.alpha * p.c0 * mu / (0.5 * (dens[i] + dens[j])), 0.0)
    f = -p.mass * (pres[i] / dens[i]**2 + pres[j] / dens[j]**2 + Pi) * dW
    acc = np.bincount(i, f, n) - np.bincount(j, f, n) - p.g

    acc[p.wall] = 0.0
    drho[p.wall] = 0.0
    return acc, drho


def load_case(path):
    """Load a case saved by run_case. Returns exactly what run_case returns."""
    with np.load(path) as f:
        data = {key: f[key] for key in f.files if key not in ("xs", "vs", "rs", "wall")}
        return f["xs"], f["vs"], f["rs"], data


def run_case(p, n_out:int = 200, save_to:str="none"):
    x, v, rho = make_particles(p, p.start)
    ts, xs, vs, rs = integrate(p, p.time_stepper, build_equations, x, v, rho, p.dt, p.n)

    every = max(1, p.n // n_out)
    k = np.unique(np.append(np.arange(1, p.n + 1, every), p.n))
    data = {
        'ts':   ts,
        'xs':   xs,
        'vs':   vs,
        'rs':   rs,    
        "t":    ts[k],
        "vmax": np.abs(vs[k]).max(axis=1),
        "KE":   0.5*np.sum(p.mass*vs[k][:, ~p.wall]**2, axis=1)
    }

    if save_to != 'none':              # e.g. save_to="test.npz"
        np.savez(save_to, **data)

    return xs, vs, rs, data   


def run_or_load(path, rerun=False):
    if os.path.exists(path) and not rerun:
        out = load_case(path)
    else:
        out = run_case(params, save_to=path)
    return out


if __name__ == "__main__":
    import os
    from dataclasses import replace
    from plotting import fancy_legend, matplot_header
    from misc import save_to_logs

    matplot_header(fontsize=11)

    if False:
        params = parameters(N=40)
        wall = params.n_wall
        fig, ax = plt.subplots(2, 1, figsize=(6, 2), 
                    gridspec_kw={'height_ratios': [.6, 1]}, layout='constrained')
        xu, _, _ = make_particles(params, start='uniform')
        ax[0].plot(xu[wall:], np.zeros_like(xu[wall:]), 'o', ms=6, mfc='none', label='uniform')
        xh, _, _ = make_particles(params, start='hydrostatic')
        ax[0].plot(xh[wall:], np.zeros_like(xh[wall:]), 'o', ms=6, mfc='none', label='hydrostatic')
        ax[1].semilogy(xh[wall:], np.abs(xu[wall:] - xh[wall:]),  )
        
        ax[0].set(xlabel='particle position')
        ax[1].set(xlabel='particle position', ylabel='distance')
        fancy_legend(ax[0], outside='top', ncol=2)
        save_to_logs(fig, '202611/particle_distribution_hydrostatic.pdf')
        plt.show()

    base_params = parameters(
        time_stepper=symplectic,
        alpha=0.1, 
        delta=0.1, 
        N=100,
        T=.5,
    )

    fig, ax = plt.subplots(1, 3, figsize=(18/2.54, 2.5), 
                gridspec_kw={'width_ratios': [1, 1, 1.5]}, layout='constrained')
    yy = np.linspace(0, base_params.H, base_params.N)
    ax[0].plot(yy, base_params.analytical_pressure(yy)/1e3, 
                                    "k--", lw=3, alpha=.6, label='analyt.')


    for case in ['uniform', 'hydrostatic']:
        output_file = f'../simulations/{case}.npz'
        params = replace(base_params, start=case)
        x, v, rho, hist = run_or_load(output_file, rerun=1)
        yf = x[-1, ~params.wall] 
        v_r = v[-1, ~params.wall]
        ax[0].plot(yf, params.compute_pressure(rho)/1e3, label=case)
        ax[1].semilogy(yf, params.relative_error(yf, rho), label=case)
        ax[2].semilogy(hist["t"], hist["KE"] + 1e-30, '-', ms=2, label=case)

    ax[0].set(xlabel=r"position $y$", ylabel="pressure $p$ (kPa)"); ax[0].legend(fontsize=11)
    ax[1].set(xlabel=r"position $y$", ylabel=r"relative error")# : $p_{\text{comp}}$, $p_{\text{analyt.}}$")
    ax[2].set(xlabel=r"time $t$", ylabel="total kinetic energy")
    fancy_legend(ax[0], match_width=True, square_box=True, ncol=1, draggable=True)

    plt.show()