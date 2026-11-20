import numpy as np
from tqdm import tqdm

def symplectic(f, x, v, rho, dt, t=0.0):
    a_n, d_n = f(x, v, rho, t)
    v_h   = v + 0.5*dt*a_n
    x_h   = x + 0.5*dt*v
    rho_h = rho + 0.5*dt*d_n
    
    a_h, d_h = f(x_h, v_h, rho_h, t + 0.5*dt)
    eps = -d_h/rho_h
    rho_new = rho*(2.0 - eps*dt) / (2.0 + eps*dt)
    v_new   = v + dt*a_h
    x_new   = x_h + 0.5*dt*v_new
    return x_new, v_new, rho_new


def rk4(f, x, v, rho, dt, t=0.0):
    k1v, k1r = f(x, v, rho, t)
    k1x = v
 
    x2, v2, r2 = x + 0.5*dt*k1x, v + 0.5*dt*k1v, rho + 0.5*dt*k1r
    k2v, k2r = f(x2, v2, r2, t + 0.5*dt)
    k2x = v2
 
    x3, v3, r3 = x + 0.5*dt*k2x, v + 0.5*dt*k2v, rho + 0.5*dt*k2r
    k3v, k3r = f(x3, v3, r3, t + 0.5*dt)
    k3x = v3
 
    x4, v4, r4 = x + dt*k3x, v + dt*k3v, rho + dt*k3r
    k4v, k4r = f(x4, v4, r4, t + dt)
    k4x = v4
 
    x_new   = x   + dt/6.0*(k1x + 2.0*k2x + 2.0*k3x + k4x)
    v_new   = v   + dt/6.0*(k1v + 2.0*k2v + 2.0*k3v + k4v)
    rho_new = rho + dt/6.0*(k1r + 2.0*k2r + 2.0*k3r + k4r)
    return x_new, v_new, rho_new


def integrate(p, time_stepper, func, x0, v0, rho0, dt, n_steps, t0=0.0):
    # func reduced to (x, v, rho, t)
    x, v, rho = (np.array(a, float) for a in (x0, v0, rho0))   # copies, inputs are left untouched
    ts = t0 + dt*np.arange(n_steps + 1)
    xs = np.empty((n_steps + 1,) + x.shape)
    vs = np.empty_like(xs)
    rs = np.empty_like(xs)

    xs[0], vs[0], rs[0] = x, v, rho
    f = lambda x, v, rho, t: func(p, x, v, rho, t)     
    
    for n in tqdm(range(n_steps)):
        x, v, rho = time_stepper(f, x, v, rho, dt, ts[n])
        xs[n + 1], vs[n + 1], rs[n + 1] = x, v, rho
    return ts, xs, vs, rs




if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    
    def damped_harmonic_oscillator_test():
        om = 2
        zeta = .2 
        T = 20
        om_d = om*np.sqrt(1 - zeta**2)
        dt_var = [.1, .05, .001, 1e-4, 1e-5]
        # Harmonic oscillator test
        acc = lambda x, v, t: -om**2 * x - 2*zeta*om*v
        
        fig, ax = plt.subplots(2, 1, figsize=(6, 5), tight_layout=True)
        for ii, dt in enumerate(dt_var):
            t, x, v, _ = integrate(acc, 1.0, 0.0, dt, int(round(T / dt)))
            xe = np.exp(-zeta*om*t)*(np.cos(om_d*t) + zeta*om/om_d*np.sin(om_d*t))
            if ii == 0:
                ax[0].plot(t, xe, "k-", label="analytical")
                ax[0].plot(t[::2], x[::2], "ro", mfc="none", ms=5, label="numerical")
                ax[0].set(xlabel="t", ylabel="x")
                ax[0].legend()
            ax[1].semilogy(t, np.abs(x - xe), 
                    color=plt.get_cmap('viridis', len(dt_var) + 1)(ii), label=dt)
            ax[1].set(xlabel="t", ylabel="|error|")
            ax[1].legend(title='time step')
        return None 


    coupled_oscillator_test()

    plt.show()
