import numpy as np

def scatter(i, j, a, b, n):
    """Sum of pair contributions a to i and b to j per particle."""
    if a.ndim == 1:
        return np.bincount(i, a, n) + np.bincount(j, b, n)

    return np.stack(
        [np.bincount(i, a[:, k], n) + np.bincount(j, b[:, k], n)
         for k in range(a.shape[1])],
        axis=1,
    )

def tait(B, rho0, rho, gm):
    """Tait equation of state."""
    return B*((rho/rho0)**gm - 1)

def hydrostatic_constants(H, rho0, gm, c0, g):
    B = rho0*c0**2/gm
    kp = (gm - 1)*rho0*g/(B*gm)
    ys = ((1 + H*kp*gm/(gm - 1))**((gm - 1)/gm) - 1)/kp
    return B, kp, ys

def hydrostatic_pressure(y, H, rho0, gm, c0, g):
    B, kp, ys = hydrostatic_constants(H, rho0, gm, c0, g)
    return B*((1 + kp*(ys - y))**(gm/(gm - 1)) - 1)

def hydrostatic_density(y, H, rho0, gm, c0, g):
    B, kp, ys = hydrostatic_constants(H, rho0, gm, c0, g)
    p = hydrostatic_pressure(y, H, rho0, gm, c0, g)
    return rho0*(1 + p/B)**(1/gm)