"""
  Predictor (half step)
    v_{n+1/2}   = v_n   + dt/2 * a_n
    x_{n+1/2}   = x_n   + dt/2 * v_n
    rho_{n+1/2} = rho_n + dt/2 * drho_n

  Evaluate a_{n+1/2}, drho_{n+1/2} at the half-step state.

  Corrector (full step)
    eps         = -drho_{n+1/2} / rho_{n+1/2}
    rho_{n+1}   = rho_n * (2 - eps*dt) / (2 + eps*dt)
    v_{n+1}     = v_n + dt * a_{n+1/2}
    x_{n+1}     = x_{n+1/2} + dt/2 * v_{n+1}
"""
import numpy as np


def symplectic_step(acc, x, v, rho, dt, t=0.0, drho=None):
    a_n = acc(x, v, t)
    d_n = drho(x, v, rho, t) if drho else 0.0
    # predictor
    v_h = v + 0.5 * dt * a_n
    x_h = x + 0.5 * dt * v          # uses v_n (not v_h)
    rho_h = rho + 0.5 * dt * d_n
    # evaluate at half step
    t_h = t + 0.5 * dt
    a_h = acc(x_h, v_h, t_h)
    d_h = drho(x_h, v_h, rho_h, t_h) if drho else 0.0
    # corrector
    eps = -d_h / rho_h
    rho_new = rho * (2.0 - eps * dt) / (2.0 + eps * dt)
    v_new = v + dt * a_h
    x_new = x_h + 0.5 * dt * v_new
    return x_new, v_new, rho_new


def integrate(acc, x0, v0, dt, n_steps, rho0=1.0, drho=None, t0=0.0):
    x = np.asarray(x0, float).copy()
    v = np.asarray(v0, float).copy()
    rho = np.asarray(rho0, float).copy()
    ts = t0 + dt * np.arange(n_steps + 1)
    xs = np.empty((n_steps + 1,) + x.shape)
    vs = np.empty_like(xs)
    rs = np.empty((n_steps + 1,) + rho.shape)
    xs[0], vs[0], rs[0] = x, v, rho
    for n in range(n_steps):
        x, v, rho = symplectic_step(acc, x, v, rho, dt, ts[n], drho)
        xs[n + 1], vs[n + 1], rs[n + 1] = x, v, rho
    return ts, xs, vs, rs


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

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
    plt.show()