import numpy as np

def make_ghosts(params, f, h):
    x = params.x
    x_min = x.min() - 0.5*params.dx
    x_max = x.max() + 0.5*params.dx
    L = x_max - x_min
    r = 2.0*h

    left = x < x_min + r
    right = x > x_max - r

    if params.bc == "periodic":
        xg = np.concatenate([x[right] - L, x[left] + L])
        fg = np.concatenate([f[right], f[left]])
    elif params.bc == "neumann":
        xg = np.concatenate([2*x_min - x[left], 2*x_max - x[right]])
        fg = np.concatenate([f[left], f[right]])
    elif params.bc == "dirichlet":
        xg = np.concatenate([2*x_min - x[left], 2*x_max - x[right]])
        fg = np.concatenate([2*params.bc_f_min - f[left],
                             2*params.bc_f_max - f[right]])
    else:
        raise ValueError(params.bc)
    return xg, fg