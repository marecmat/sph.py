
def symplectic(f, x, v, rho, dt, t=0.0):
    a_n, d_n = f(x, v, rho, t)
    v_h = v + 0.5*dt*a_n
    x_h = x + 0.5*dt*v
    rho_h = rho + 0.5*dt*d_n

    a_h, d_h = f(x_h, v_h, rho_h, t + 0.5*dt)
    eps = -d_h/rho_h
    rho_new = rho*(2.0 - eps*dt)/(2.0 + eps*dt)
    v_new = v + dt*a_h
    x_new = x_h + 0.5*dt*v_new
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

    x_new = x + dt/6.0*(k1x + 2.0*k2x + 2.0*k3x + k4x)
    v_new = v + dt/6.0*(k1v + 2.0*k2v + 2.0*k3v + k4v)
    rho_new = rho + dt/6.0*(k1r + 2.0*k2r + 2.0*k3r + k4r)
    return x_new, v_new, rho_new
