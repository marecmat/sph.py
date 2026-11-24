from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys

sys.path.append("../src/")
import kernels

import misc
from matplotlib import rcParams
rcParams['text.usetex'] = True

@dataclass
class Params: 
    L: float = 1.
    N: int = 100

    @property
    def dx(self): 
        return self.L/self.N

    @property
    def V(self): 
        return self.dx

    @property
    def x(self): 
        return (np.arange(self.N) + 0.5) * self.dx


def sph_gradient(f, params, smoothing_param, kernel_gradient,
                 neighbour_search: bool = False):

    h = smoothing_param * params.dx
    grad = np.zeros(params.N)

    if neighbour_search:

        pairs = cKDTree(params.x.reshape(-1, 1)).query_pairs(r=2.*h,output_type="ndarray")

        i, j = pairs[:, 0], pairs[:, 1]

        dW = kernel_gradient(params.x[i] - params.x[j], h, dim=1)
        interp = params.V*(f[j] - f[i])*dW
        grad += np.bincount(i, weights=interp, minlength=params.N) \
              + np.bincount(j, weights=interp, minlength=params.N)

    else:
        r = params.x[:, None] - params.x[None, :]
        dW = kernel_gradient(r, h, dim=1)
        grad = np.sum(params.V*(f[None, :] - f[:, None])*dW, axis=1)

    return grad


if __name__ == "__main__":
    p = Params()
    smoothing_param = 3
    f = p.x**3 + 0.2*p.x**2 + 4*p.x
    fp = 3*p.x**2 + .4*p.x + 4
    grad = sph_gradient(f, p, smoothing_param, 
        kernels.cubic_spline_gradient, neighbour_search=True)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(p.x, fp, 'r')
    ax[0].plot(p.x[::2], grad[::2], 'ko--')
    ax[1].semilogy(p.x, np.abs(fp - grad), 'g-')
    # misc.save_to_logs(fig, "202611/gradient_check_wo_ghosts.pdf", dpi=100)
    plt.show()