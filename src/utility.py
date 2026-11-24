import numpy as np

def scatter(i, j, a, b, n):
    if a.ndim == 1:
        return np.bincount(i, a, n)+np.bincount(j, b, n)
    return np.stack([np.bincount(i, a[:, k], n)+np.bincount(j, b[:, k], n) for k in range(2)], axis=1)
