"""Particle layouts for the supported test cases.

The discretisation input is the particle spacing ``dx``.  Integer particle
counts and the effective domain length are derived from ``dx`` and the physical
domain, so callers never have to guess ``N`` / ``Nx``.

2D fluid layouts use a staggered (diamond) lattice: odd rows are shifted by
``dx/2`` relative to even rows, and wall layers follow the same parity.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class Layout:
    x: np.ndarray                    # (n, dim)
    v: np.ndarray                    # (n, dim)
    rho: np.ndarray                  # (n,)
    wall: np.ndarray                 # (n,) bool
    dx: float                        # effective spacing used by the solver
    L: Sequence[Optional[float]]     # domain length per axis, None if unused
    H: Optional[float] = None
    B: Optional[float] = None
    kp: Optional[float] = None
    ys: Optional[float] = None


def hydrostatic_constants(H, rho0, gm, c0, g):
    """Return ``B``, ``kp`` and free-surface height ``ys`` for the Tait EOS."""
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


def _counts(dx, H, Lx=None):
    """Derive integer counts and the effective spacing from a target ``dx``."""
    N = max(1, int(round(H/dx)))
    dy = H/N
    if Lx is None:
        return N, 1, dy, None
    Nx = max(1, int(round(Lx/dy)))
    return N, Nx, dy, Nx*dy


def _fluid_column(N, dy, H, rho0, gm, c0, g, start):
    B, kp, ys = hydrostatic_constants(H, rho0, gm, c0, g)
    k = np.arange(N)

    if start == "uniform":
        yf = (k + 0.5)*dy
        rf = np.full(N, rho0)
    elif start == "hydrostatic":
        u = (1 + kp*gm*(N - 1 - k + 0.5)*dy/(gm - 1))**((gm - 1)/gm)
        yf = ys - (u - 1)/kp
        rf = rho0*u**(1/(gm - 1))
    else:
        raise ValueError(f"unknown start: {start!r}")
    return yf, rf, B, kp, ys


def _lattice_rows(row_specs, N, Nx, dy, Lx, periodic, n_wall_x, rho0, diamond):
    """Build a staggered lattice row by row.

    ``row_specs`` contains ``(row_index, y, rho_row, is_wall_row)``.  Column
    indices are ``-n_wall_x..-1`` (left wall), ``0..Nx-1`` (fluid) and
    ``Nx..Nx+n_wall_x-1`` (right wall).
    """
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

        rho = np.full(len(col), rho0)
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


def hydrostatic_column_1d(
    dx, H, rho0=1000., gm=7., c0=None, g=9.81, start="hydrostatic", n_wall=4,
):
    if c0 is None:
        raise ValueError("pass c0 for the 1D column")
    N, _, dy, _ = _counts(dx, H)
    yf, rf, B, kp, ys = _fluid_column(N, dy, H, rho0, gm, c0, g, start)
    yw = -(np.arange(n_wall, 0, -1) - 0.5)*dy

    y = np.concatenate([yw, yf])
    rho = np.concatenate([np.full(n_wall, rho0), rf])
    wall = np.zeros(len(y), dtype=bool)
    wall[:n_wall] = True
    x = y[:, None]
    return Layout(x, np.zeros_like(x), rho, wall, dy, (), H, B, kp, ys)


def hydrostatic_column_2d(
    dx, H, Lx, rho0=1000., gm=7., c0=None, g=9.81, start="hydrostatic",
    n_wall=4, walls="bottom", periodic=True, diamond=True,
):
    """2D hydrostatic column with a periodic strip or rigid side walls."""
    if c0 is None:
        raise ValueError("pass c0 for the 2D column")
    if walls not in ("bottom", "both"):
        raise ValueError(f"unknown walls: {walls!r}")

    N, Nx, dy, Lx_eff = _counts(dx, H, Lx)
    yf, rf, B, kp, ys = _fluid_column(N, dy, H, rho0, gm, c0, g, start)
    n_wall_x = 0 if periodic else n_wall

    specs = []
    for k in range(n_wall - 1, -1, -1):
        specs.append((-n_wall + k, -(k + 0.5)*dy, None, True))
    for j in range(N):
        specs.append((j, yf[j], rf[j], False))
    if walls == "both":
        for k in range(n_wall):
            specs.append((N + k, H + (k + 0.5)*dy, None, True))

    x, rho, wall, _, _ = _lattice_rows(
        specs, N, Nx, dy, Lx_eff, periodic, n_wall_x, rho0, diamond,
    )
    L = (Lx_eff, None) if periodic else (Lx_eff + 2*n_wall_x*dy, None)
    return Layout(x, np.zeros_like(x), rho, wall, dy, L, H, B, kp, ys)


def channel_2d(dx, H, Lx, rho0=1000., gm=7., c0=None, n_wall=3, diamond=True):
    """Uniform fluid between two solid walls, periodic in x (Poiseuille)."""
    if c0 is None:
        raise ValueError("pass c0 for the channel")

    N, Nx, dy, Lx_eff = _counts(dx, H, Lx)
    specs = []
    for k in range(n_wall - 1, -1, -1):
        specs.append((-n_wall + k, -(k + 0.5)*dy, None, True))
    for j in range(N):
        specs.append((j, (j + 0.5)*dy, rho0, False))
    for k in range(n_wall):
        specs.append((N + k, H + (k + 0.5)*dy, None, True))

    x, rho, wall, _, _ = _lattice_rows(
        specs, N, Nx, dy, Lx_eff, True, 0, rho0, diamond,
    )
    B = rho0*c0**2/gm
    return Layout(x, np.zeros_like(x), rho, wall, dy, (Lx_eff, None), H, B)


def dam_break_2d(
    dx, H, Lx, rho0=1000., gm=7., c0=None, n_wall=3,
    fluid_width=0.4, fluid_height=0.5, diamond=True,
):
    """Fluid block in a rigid rectangular tank with a free surface."""
    if c0 is None:
        raise ValueError("pass c0 for the dam break")

    N, Nx, dy, Lx_eff = _counts(dx, H, Lx)
    specs = []
    for k in range(n_wall - 1, -1, -1):
        specs.append((-n_wall + k, -(k + 0.5)*dy, None, True))
    for j in range(N):
        specs.append((j, (j + 0.5)*dy, rho0, False))
    for k in range(n_wall):
        specs.append((N + k, H + (k + 0.5)*dy, None, True))

    x, rho, wall, row, col = _lattice_rows(
        specs, N, Nx, dy, Lx_eff, False, n_wall, rho0, diamond,
    )
    nfx = max(1, int(Nx*fluid_width))
    nfy = max(1, int(N*fluid_height))
    fluid_cell = (
        (row >= 0) & (row < nfy) & (col >= 0) & (col < nfx)
    )
    keep = wall | fluid_cell

    B = rho0*c0**2/gm
    L = (Lx_eff + 2*n_wall*dy, H + 2*n_wall*dy)
    return Layout(
        x[keep], np.zeros((keep.sum(), 2)), rho[keep], wall[keep],
        dy, L, H, B,
    )
