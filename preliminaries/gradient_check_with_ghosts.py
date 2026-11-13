from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys

sys.path.append("../src/")
import kernels
from ghost_particles_1Dlattice import make_ghosts

from matplotlib import rcParams
rcParams['text.usetex'] = True


@dataclass
class Params: 
    L: float = 1.
    N: int = 250
    bc: str = 'none'
    bc_f_min: float = 0.0
    bc_f_max: float = 0.0                  

    @property
    def dx(self): 
        return self.L/self.N

    @property
    def V(self): 
        return self.dx

    @property
    def x(self): 
        return (np.arange(self.N) + .5) * self.dx


def sph_gradient(f, params, smoothing_param, kernel_gradient,
                 neighbour_search: bool = False):

    h = smoothing_param * params.dx
    grad = np.zeros(params.N)
    
    if params.bc == 'none':
        x_all, f_all = params.x, f
    else:
        xg, fg = make_ghosts(params, f, h)
        x_all = np.concatenate([params.x, xg])
        f_all = np.concatenate([f, fg])

    N_all = x_all.size
    if neighbour_search:
        pairs = cKDTree(x_all.reshape(-1, 1)).query_pairs(
            r=2.0*h, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        # keep only the non ghost pairs  
        i, j = i[i < params.N], j[i < params.N]
        dW = kernel_gradient(x_all[i] - x_all[j], h, dim=1)
        interp = params.V * (f_all[j] - f_all[i]) * dW

        grad_all = (np.bincount(i, weights=interp, minlength=N_all)
                    + np.bincount(j, weights=interp, minlength=N_all))
    else:
        r = x_all[:params.N, None] - x_all[None, :]          # real rows only
        dW = kernel_gradient(r, h, dim=1)
        return np.sum(params.V * (f_all[None, :] - f_all[:N, None]) * dW,
                      axis=1)

    return grad_all[:params.N]


if __name__ == "__main__":
    p = Params()
    smoothing_param = 5

    F  = lambda x: x**3 + 0.2*x**2 + 4*x
    f  = F(p.x)
    fp = 3*p.x**2 + .4*p.x + 4

    p_ghost = Params(bc='dirichlet', 
        bc_f_min=F(p.x[0] - p.dx/2), 
        bc_f_max=F(p.x[-1] + p.dx/2))

    xg, fg = make_ghosts(p_ghost, f, smoothing_param*p.dx)
    fig, ax = plt.subplots(num='ghost particles on a regular function', figsize=(3,2), tight_layout=True)
    ax.plot(xg, fg, 'ro', mfc='None', ms=3, label='ghost particles')
    ax.plot(p.x, f, 'ko', label='lattice')

    grad = sph_gradient(f, p, smoothing_param, 
        kernels.cubic_spline_gradient, neighbour_search=True)
    grad_ghost = sph_gradient(f, p_ghost, smoothing_param, 
        kernels.cubic_spline_gradient, neighbour_search=True)

    fig1, ax = plt.subplots(2, 1, figsize=(4, 5), tight_layout=True)
    ax[0].plot(p.x, fp, 'k--', lw=2, label='exact')
    ax[0].plot(p.x[::2], grad[::2], lw=1, label='no ghosts')
    ax[0].plot(p.x[::2], grad_ghost[::2], lw=1, label='ghosts')
    ax[0].legend()
    ax[1].semilogy(p.x, np.abs(fp - grad), label='error, no ghosts')
    ax[1].semilogy(p.x, np.abs(fp - grad_ghost), label='error, ghosts')
    ax[1].legend()


    # Test with a periodic function
    params = dict(N=150)
    p = Params(**params)
    smoothing_param = 5

    F  = lambda x: np.sin(10*np.pi*x/p.L)
    f  = F(p.x)
    # fp = 3.2*np.pi/p.L*np.cos(3.2*np.pi*p.x/p.L)
    fp = 10*np.pi/p.L*np.cos(10*np.pi*p.x/p.L)

    p_ghost = Params(bc='periodic', **params)
    xg, fg = make_ghosts(p_ghost, f, smoothing_param*p.dx)
    fig, ax = plt.subplots(num='ghost particles', figsize=(3,2), tight_layout=True)
    ax.plot(xg, fg, 'ro', mfc='None', ms=3, label='ghost particles')
    ax.plot(p.x, f, 'ko', label='lattice')

    grad = sph_gradient(f, p, smoothing_param, 
        kernels.cubic_spline_gradient, neighbour_search=True)
    grad_ghost = sph_gradient(f, p_ghost, smoothing_param, 
        kernels.cubic_spline_gradient, neighbour_search=True)

    fig2, ax = plt.subplots(2, 1, num='periodic function with ghosts',  figsize=(4, 5), tight_layout=True)
    ax[0].plot(p.x, fp, 'k--', lw=2, label='exact')
    ax[0].plot(p.x[::2], grad[::2], lw=1, label='no ghosts')
    ax[0].plot(p.x[::2], grad_ghost[::2], lw=1, label='ghosts')
    ax[0].legend()
    ax[1].semilogy(p.x, np.abs(fp - grad), label='error, no ghosts')
    ax[1].semilogy(p.x, np.abs(fp - grad_ghost), label='error, ghosts')
    ax[1].legend()
    plt.show()