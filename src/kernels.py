import numpy as np 

def cubic_spline(r, h, dim):
    q = np.abs(r)/h

    if dim == 1:
        dim_fac = (2./(3.*h))
    elif dim == 2:
        dim_fac = 10/(7*np.pi*h**2)
    elif dim == 3:
        dim_fac = 1/(np.pi*h**3)
    else: 
        raise ValueError('dim error')

    return  np.where(q < 1.0, dim_fac*(1.0 - 1.5*q**2 + 0.75*q**3),
            np.where(q < 2.0, dim_fac*0.25*(2.0 - q)**3,
            0.0))



def cubic_spline_gradient(r, h, dim):

    r = np.asarray(r)
    q = np.abs(r) / h

    if dim == 1:
        dim_fac = 2. / (3. * h)
    elif dim == 2:
        dim_fac = 10. / (7. * np.pi * h**2)
    elif dim == 3:
        dim_fac = 1. / (np.pi * h**3)
    else:
        raise ValueError('dim error')

    dw = np.where(q < 1.0, -3.0*q + 2.25*q**2,
         np.where(q < 2.0, -0.75*(2.0-q)**2,
         0.0))

    if dim == 1:
        return (dim_fac / h) * dw * np.sign(r)

    else:
        norm_r = np.linalg.norm(r, axis=-1, keepdims=True)
        direction = np.divide(r, norm_r, 
                out=np.zeros_like(r, dtype=float), where=norm_r != 0)

        return (dim_fac / h) * dw[..., None] * direction



def wendland(r, h, dim):
    q = np.abs(r)/h

    if dim == 1:
        dim_fac = 3./(4.*h)
    elif dim == 2:
        dim_fac = 7./(4.*np.pi*h**2)
    elif dim == 3:
        dim_fac = 21./(16.*np.pi*h**3)
    else: 
        raise ValueError('dim error')

    return np.where(q < 2.0, dim_fac*(1 - 0.5*q)**4*(2*q+1), 0.0)



def wendland_gradient(r, h, dim):

    r = np.asarray(r)
    q = np.abs(r) / h

    if dim == 1:
        dim_fac = 3. / (4. * h)
    elif dim == 2:
        dim_fac = 7. / (4. * np.pi * h**2)
    elif dim == 3:
        dim_fac = 21. / (16. * np.pi * h**3)
    else:
        raise ValueError('dim error')

    dw = np.where(q < 2.0, -5.0*q*(1.0 - 0.5*q)**3, 0.0)

    if dim == 1:
        return (dim_fac / h) * dw * np.sign(r)

    else:
        norm_r = np.linalg.norm(r, axis=-1, keepdims=True)

        direction = np.divide(r, norm_r, 
                out=np.zeros_like(r, dtype=float), where=norm_r != 0)
        return (dim_fac / h) * dw[..., None] * direction


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    h = 2
    r = np.linspace(-6, 6, 100)

    fig, ax = plt.subplots(1, 1, figsize=(4,4), constrained_layout=True)
    ax.plot(r/h, cubic_spline(r,h, dim=2), label='cubic')
    ax.plot(r/h, wendland(r,h, dim=2), label='wendland')
    ax.set(xlabel="q", ylabel="W")
    ax.legend()
    plt.show()