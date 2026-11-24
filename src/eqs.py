import numpy as np

def scatter(i, j, a, b, n):
    """Sum pair contributions ``a`` (to i) and ``b`` (to j) per particle."""
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
