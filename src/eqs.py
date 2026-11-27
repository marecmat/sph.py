import numpy as np

def accumulate_contribs(i, j, a, b, n):
    """
    s_p = sum_{q: i_q = p} a_q + sum_{q: j_q = p} b_q 

    for each interacting pair, 
        i first particle index
        j 2nd particle idx
        a contribution of 1st particle
        b contribution of 2nd particle
        n total nb of partcl
    
    bincount accumulates the pairwise contributions to each particle, 
    it counts the number of occurrences of each value in array given weights a/b
    """
    
    if a.ndim == 1:
        return np.bincount(i, a, n) + np.bincount(j, b, n)
    else:   
        return np.stack([   
                        np.bincount(i, a[:, k], n) + \
                        np.bincount(j, b[:, k], n) 
                    for k in range(a.shape[1]) #iteration over each spatial component
                ], axis=1,)

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