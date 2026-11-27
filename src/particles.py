from dataclasses import dataclass
import numpy as np
from .eqs import hydrostatic_constants, hydrostatic_density

@dataclass
class Layout:
    # Initialisation data for all runs
    x: np.ndarray    # (n, dim)
    v: np.ndarray    # (n, dim)
    rho: np.ndarray  # (n,)
    wall: np.ndarray # (n,) bool
    dx: float        # spacing
    L: tuple = ()    # domain length / axis


def counts(dx, H, Lx=None):
    N = max(1, int(round(H/dx)))
    dy = H/N
    if Lx is None:
        return N, 1, dy, None
    Nx = max(1, int(round(Lx/dy)))
    return N, Nx, dy, Nx*dy


def fluid_column(N, H, rho0, gm, c0, g, start, n_wall):
    B, kp, ys = hydrostatic_constants(H, rho0, gm, c0, g)
    k = np.arange(N)

    if start == "uniform":
        dx = H/N
        yf = (k + 0.5)*dx
        rf = np.full(N, rho0)
        yw = -(np.arange(n_wall, 0, -1) - 0.5)*dx
        rw = np.full(n_wall, rho0)

    elif start == "hydrostatic":
        dx = H/N
        u = (1 + kp*gm*(N - 1 - k + 0.5)*dx/(gm - 1))**((gm - 1)/gm)
        yf = ys - (u - 1)/kp
        rf = rho0*u**(1/(gm - 1))
        yw = -(np.arange(n_wall, 0, -1) - 0.5)*dx
        rw = np.full(n_wall, rho0)

    elif start == "hydrostatic_tait":
        s0 = (1 + kp*ys)**(1/(gm - 1))
        sH = (1 + kp*(ys - H))**(1/(gm - 1))
        mass = B/g*(s0**gm - sH**gm) # exact fluid mass
        dx = mass/(rho0*N)
        m = rho0*dx

        sf = (s0**gm - (g/B)*(k + 0.5)*m)**(1/gm)
        yf = ys + (1 - sf**(gm - 1))/kp
        rf = rho0*sf

        kw = np.arange(1, n_wall + 1)
        sw = (s0**gm + (g/B)*(kw - 0.5)*m)**(1/gm)
        yw = ys + (1 - sw**(gm - 1))/kp
        rw = rho0*sw
        yw, rw = yw[::-1], rw[::-1]

    else:
        raise ValueError(f"unknown start: {start!r}")

    return yf, rf, yw, rw, dx


def lattice_rows(row_specs, N, Nx, dy, Lx, periodic, n_wall_x, rho0, diamond):
    xs, rhos, walls, rows, cols = [], [], [], [], []

    for r, y, rho_row, is_wall_row in row_specs:
        shift = 0.5*dy*(r % 2) if diamond else 0.0
        x0 = 0.5*dy + shift

        if periodic:
            col = np.arange(Nx)
            xrow = np.mod(x0 + dy*col, Lx)
            wall_col = np.zeros(Nx, dtype=bool)
        else:
            left = -np.arange(n_wall_x, 0, -1)
            fluid = np.arange(Nx)
            right = Nx + np.arange(0, n_wall_x)
            col = np.concatenate([left, fluid, right])
            xrow = x0 + dy*col
            wall_col = (col < 0) | (col >= Nx)

        rho = np.full(len(col), rho0 if rho_row is None else rho_row)
        if not is_wall_row:
            rho[~wall_col] = rho_row

        xs.append(xrow)
        rhos.append(rho)
        walls.append(wall_col | is_wall_row)
        rows.append(np.full(len(col), r))
        cols.append(col)

    y_all = np.concatenate([np.full(len(xrow), spec[1])
                            for xrow, spec in zip(xs, row_specs)])
    x_all = np.concatenate(xs)
    return (
        np.column_stack([x_all, y_all]),
        np.concatenate(rhos),
        np.concatenate(walls),
        np.concatenate(rows),
        np.concatenate(cols),
    )


def hydrostatic_column_1d(dx, H, rho0=1000., gm=7., c0=None, g=9.81, start="hydrostatic", 
                                                    n_wall=4, n_ghost=2):
    if c0 is None: raise ValueError("pass c0 for the 1D column")
    N, _, _, _ = counts(dx, H)
    yf, rf, yw, rw, dx_eff = fluid_column(N, H, rho0, gm, c0, g, start, n_wall)
    yg = H + (np.arange(n_ghost) + 1.0)*dx_eff
    rg = hydrostatic_density(yg, H, rho0, gm, c0, g)
    
    y = np.concatenate([yw, yf, yg])
    rho = np.concatenate([rw, rf, rg])
    wall = np.zeros(len(y), dtype=bool)
    wall[:n_wall] = True
    if n_ghost:
        wall[-n_ghost:] = True
    x = y[:, None]
    return Layout(x, np.zeros_like(x), rho, wall, dx_eff, ())


def hydrostatic_column_2d(dx, H, Lx, rho0=1000., gm=7., c0=None, g=9.81, start="hydrostatic", 
                                        n_wall=4, n_ghost=2, walls="bottom", periodic=True, diamond=True):
    """2D hydrostatic column with a periodic strip or rigid side walls."""
    if c0 is None:
        raise ValueError("pass c0 for the 2D column")
    if walls not in ("bottom", "both"):
        raise ValueError(f"unknown walls: {walls!r}")

    N, _, _, _ = counts(dx, H)
    yf, rf, yw, rw, dx_eff = fluid_column(
        N, H, rho0, gm, c0, g, start, n_wall,
    )
    Nx = max(1, int(round(Lx/dx_eff)))
    Lx_eff = Nx*dx_eff
    n_wall_x = 0 if periodic else n_wall

    specs = []
    for k in range(n_wall):
        specs.append((-n_wall + k, yw[k], None, True))
    for j in range(N):
        specs.append((j, yf[j], rf[j], False))
    if walls == "both":
        for k in range(n_wall):
            specs.append((N + k, H + (k + 0.5)*dx_eff, None, True))
    else:
        for k in range(n_ghost):
            yg = H + (k + 1.0)*dx_eff
            rg = hydrostatic_density(yg, H, rho0, gm, c0, g)
            specs.append((N + k, yg, rg, True))

    x, rho, wall, _, _ = lattice_rows(
        specs, N, Nx, dx_eff, Lx_eff, periodic, n_wall_x, rho0, diamond,
    )
    L = (Lx_eff, None) if periodic else (Lx_eff + 2*n_wall_x*dx_eff, None)
    return Layout(x, np.zeros_like(x), rho, wall, dx_eff, L)


def channel_2d(dx, H, Lx, rho0=1000., n_wall=3, diamond=True):
    N, Nx, dy, Lx_eff = counts(dx, H, Lx)
    specs = []
    for k in range(n_wall - 1, -1, -1):
        specs.append((-n_wall + k, -(k + 0.5)*dy, None, True))
    for j in range(N):
        specs.append((j, (j + 0.5)*dy, rho0, False))
    for k in range(n_wall):
        specs.append((N + k, H + (k + 0.5)*dy, None, True))

    x, rho, wall, _, _ = lattice_rows(
        specs, N, Nx, dy, Lx_eff, True, 0, rho0, diamond,
    )
    return Layout(x, np.zeros_like(x), rho, wall, dy, (Lx_eff, None))


def dam_break_2d(dx, H, Lx, rho0=1000., n_wall=3, fluid_width=0.4, fluid_height=0.5, diamond=True):
    N, Nx, dy, Lx_eff = counts(dx, H, Lx)
    specs = []
    for k in range(n_wall - 1, -1, -1):
        specs.append((-n_wall + k, -(k + 0.5)*dy, None, True))
    for j in range(N):
        specs.append((j, (j + 0.5)*dy, rho0, False))
    for k in range(n_wall):
        specs.append((N + k, H + (k + 0.5)*dy, None, True))

    x, rho, wall, row, col = lattice_rows(specs, N, Nx, dy, Lx_eff, False, n_wall, rho0, diamond)
    nfx = max(1, int(Nx*fluid_width))
    nfy = max(1, int(N*fluid_height))
    fluid_cell = ((row >= 0) & (row < nfy) & (col >= 0) & (col < nfx))
    keep = wall | fluid_cell
    L = (Lx_eff + 2*n_wall*dy, H + 2*n_wall*dy)

    return Layout(x[keep], np.zeros((keep.sum(), 2)), rho[keep], wall[keep], dy, L)