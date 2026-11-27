import numpy as np

cubic_factors = [ 2./3., 10./(7.*np.pi), 1./np.pi ]
wendland_factors = [ 3./4., 7./(4.*np.pi), 21./(16.*np.pi) ]

def _factor(h, dim, factors):
    if dim not in (1, 2, 3):
        raise ValueError("dim must be 1, 2 or 3.")
    return factors[dim - 1] / h**dim

def _q(r, h, dim):
    if dim == 1:
        return np.abs(r) / h
    return np.linalg.norm(r, axis=-1) / h

def _gradient(r, h, dim, dw, factor):
    if dim == 1: return (factor / h) * dw * np.sign(r)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    direction = np.divide( r, rn, out=np.zeros_like(r, dtype=float), where=rn != 0)
    return (factor / h) * dw[..., None] * direction

def cubic_spline(r, h, dim):
    factor = _factor(h, dim, cubic_factors)
    q = _q(r, h, dim)
    f = np.where( q < 1., 1. - 1.5*q**2 + 0.75*q**3,
        np.where( q < 2., 0.25*(2. - q)**3, 0.))
    return factor * f

def cubic_spline_gradient(r, h, dim):
    factor = _factor(h, dim, cubic_factors)
    q = _q(r, h, dim)
    dw = np.where( q < 1., -3.*q + 2.25*q**2,
         np.where( q < 2., -0.75*(2. - q)**2, 0.))
    return _gradient(r, h, dim, dw, factor)

def wendland(r, h, dim):
    factor = _factor(h, dim, wendland_factors)
    q = _q(r, h, dim)
    f = np.where( q < 2., (1. - 0.5*q)**4 * (2.*q + 1.), 0.)
    return factor * f

def wendland_gradient(r, h, dim):
    factor = _factor(h, dim, wendland_factors)
    q = _q(r, h, dim)
    dw = np.where( q < 2., -5.*q*(1. - 0.5*q)**3, 0.)
    return _gradient(r, h, dim, dw, factor)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    h = 2.
    r = np.linspace(-6., 6., 200)
    fig, ax = plt.subplots( figsize=(5, 4), constrained_layout=True)
    ax.plot(r/h, cubic_spline(r, h, dim=1), label="spline")
    ax.plot(r/h, wendland(r, h, dim=1), label="wendland")
    ax.set(xlabel="q", ylabel="W(r, h)"); ax.legend()
    plt.show()
