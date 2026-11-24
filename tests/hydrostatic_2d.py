"""2D hydrostatic column, with a periodic strip and with rigid side walls.

Particles start on a staggered (diamond) lattice.  The only discretisation
input is ``DX``; particle counts follow from ``DX`` and the domain size.

Run from anywhere:  python tests/hydrostatic_2d.py [--no-show]
"""

import numpy as np
import matplotlib.pyplot as plt

from common import finish
from hydrostatic import hydrostatic_density, hydrostatic_pressure
from src.eqs import tait
from src.particles import hydrostatic_column_2d
from src.wcsph import SPHSolver


DX, H, LX = 0.025, 1.0, 0.2
N_WALL = 4
RHO0, GM, G, DRHO_MAX = 1000.0, 7.0, 9.81, 0.01
C0 = np.sqrt(G*H/DRHO_MAX)
T = 0.4


def top_ghost_density(xw):
    """Exact Tait density on top ghosts; leave bottom walls to Adami."""
    return np.where(xw[:, 1] > H,
                    hydrostatic_density(xw[:, 1], H, RHO0, GM, C0, G),
                    np.nan)


# Rigid walls need the wider support to sample enough wall neighbours.
CASES = (
    ("periodic", True, "bottom", 1.3),
    ("rigid walls", False, "bottom", 2.0),
)
STARTS = ("hydrostatic", "uniform")

fig, ax = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")

for row, (name, periodic, walls, h_factor) in enumerate(CASES):
    for start in STARTS:
        layout = hydrostatic_column_2d(
            DX, H, LX, rho0=RHO0, gm=GM, c0=C0, g=G, start=start,
            n_wall=N_WALL, walls=walls, periodic=periodic, diamond=True,
        )
        rho_ref = lambda x: hydrostatic_density(x[:, 1], H, RHO0, GM, C0, G)
        solver = SPHSolver(
            dim=2, dx=layout.dx, h_factor=h_factor, H=H,
            rho0=RHO0, gm=GM, c0=C0, g=G, alpha=0.1, delta=0.1,
            rho_ref=rho_ref, wall_density=top_ghost_density, T=T,
            periodic_axes=(0,) if periodic else (), L=layout.L,
        )
        out = solver.run(
            layout.x, layout.v, layout.rho, layout.wall,
            save_every=max(1, solver.n_steps//200),
        )

        fluid = ~layout.wall
        xs = out["x"][-1][fluid]
        rho = out["rho"][-1][fluid]
        P = tait(solver.B, solver.rho0, rho, solver.gm)
        err = np.abs(P - hydrostatic_pressure(xs[:, 1], H, RHO0, GM, C0, G))/(RHO0*G*H)
        bottom = err[xs[:, 1] < 0.05*H]
        top = err[xs[:, 1] > 0.95*H]
        KE = 0.5*solver.mass*np.sum(out["v"][:, fluid]**2, axis=(1, 2))

        print(f"[{name} / {start}] dx={layout.dx:.3e}  steps={solver.n_steps}")
        print(f"   max |p - p_exact|/(rho g H) = {err.max():.2e}"
              f"  (near y=0: {bottom.max():.2e}, near y=H: {top.max():.2e})")
        print(f"   max |vx| = {np.abs(out['v'][-1][fluid, 0]).max():.2e}"
              f"  max |vy| = {np.abs(out['v'][-1][fluid, 1]).max():.2e}")

        label = start
        ax[row, 0].plot(xs[:, 1], P/1e3, ".", ms=2, label=label)
        ax[row, 1].semilogy(xs[:, 1], err + 1e-30, ".", ms=2, label=label)
        ax[row, 2].semilogy(out["t"][1:], KE[1:], label=label)

    yy = np.linspace(0, H, 200)
    ax[row, 0].plot(yy, hydrostatic_pressure(yy, H, RHO0, GM, C0, G)/1e3,
                    "k--", lw=2, alpha=0.6, label="analytic")
    ax[row, 0].set_title(name)
    ax[row, 0].set(xlabel="y", ylabel="p (kPa)")
    ax[row, 1].set(xlabel="y", ylabel=r"$|p-p_{exact}|/(\rho g H)$")
    ax[row, 2].set(xlabel="t", ylabel="kinetic energy")
    for a in ax[row]:
        a.legend()

finish(fig, "hydrostatic_2d.png")
