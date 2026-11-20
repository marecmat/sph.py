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


def make_particles_1D(p, start="hydrostatic"):
    k = np.arange(p.N)
    yw = -(np.arange(p.n_wall) + 0.5) * p.dy
    if start == "uniform":
        yf = (k + 0.5) * p.dy
        rf = np.full(p.N, p.rho0)
    else:
        u = (1 + p.kp*p.gm*(p.N - 1 - k + 0.5)*p.dy/(p.gm - 1))**((p.gm - 1)/p.gm)
        yf = p.ys - (u - 1)/p.kp
        rf = p.rho0*u**(1/(p.gm - 1))
    x = np.concatenate([yw, yf])
    v = np.zeros_like(x)

    rho = np.concatenate([np.full(p.n_wall, p.rho0), rf])
    return x, v, rho

def make_particles_2D(p, start="hydrostatic"):
    nx = np.sqrt(p.N*(p.H[0]/p.H[1]))
    ny = np.sqrt(p.N*(p.H[1]/p.H[0]))
    kx = np.arange(nx)
    ky = np.arange(ny)

    xf = (kx + 0.5) * p.dy[0]
    yf = (ky + 0.5) * p.dy[1]

    xx, yy = np.meshgrid()

    return x, v, rho


if __name__ == "__main__":

    import numpy as np

    def make_particles_2D(p, start="hydrostatic"):

        # Number of particles in each direction
        Nx = p.Nx
        Ny = p.Ny
        N = Nx * Ny

        # Domain
        x_min, x_max = p.xmin, p.xmax
        y_min, y_max = 0.0, p.ys

        Lx = x_max - x_min
        H = y_max - y_min

        # Horizontal particle positions
        dx = Lx / Nx
        xf = x_min + (np.arange(Nx) + 0.5) * dx

        if start == "uniform":

            # Uniform distribution
            yf = y_min + (np.arange(Ny) + 0.5) * H / Ny
            rho_y = np.full(Ny, p.rho0)

        else:

            # Analytical hydrostatic density profile
            yg = np.linspace(y_min, y_max, 10000)

            rho_g = p.rho0 * (
                1 + p.kp * p.gm * p.g * (H - yg)
                / (p.gm - 1)
            )**(1 / (p.gm - 1))

            # Cumulative mass per unit horizontal length
            dy = yg[1] - yg[0]

            M = np.zeros_like(yg)
            M[1:] = np.cumsum(
                0.5 * (rho_g[1:] + rho_g[:-1]) * dy
            )

            # Equal-mass vertical positions
            target_mass = (
                (np.arange(Ny) + 0.5) / Ny
            ) * M[-1]

            yf = np.interp(target_mass, M, yg)

            # Density at particle positions
            rho_y = p.rho0 * (
                1 + p.kp * p.gm * p.g * (H - yf)
                / (p.gm - 1)
            )**(1 / (p.gm - 1))

        # Cartesian particle distribution
        X, Y = np.meshgrid(xf, yf)

        xp = X.ravel()
        yp = Y.ravel()

        # Density: same for every particle in a horizontal row
        rho = np.repeat(rho_y, Nx)

        # Velocities
        vx = np.zeros(N)
        vy = np.zeros(N)

        # Equal particle masses
        mass = Lx * np.sum(
            0.5 * (rho_y[:-1] + rho_y[1:])
            * np.diff(yf)
        ) / N if Ny > 1 else Lx * rho_y[0] * H / N

        # More accurate mass from the hydrostatic profile
        if start == "hydrostatic":
            mass = Lx * M[-1] / N
        else:
            mass = Lx * p.rho0 * H / N

        m = np.full(N, mass)

        return xp, yp, vx, vy, rho, m


    import matplotlib.pyplot as plt

    p = params()
    xp, yp, vx, vy, rho, m = make_particles_2D(p)

    plt.figure(figsize=(7, 6))

    sc = plt.scatter(
        xp, yp,
        c=rho,
        cmap="viridis",
        s=8
    )

    plt.colorbar(sc, label="Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D hydrostatic particle distribution")
    plt.axis("equal")
    plt.show()
