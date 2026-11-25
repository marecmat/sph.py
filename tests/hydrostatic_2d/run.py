import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.eqs import hydrostatic_density
from src.particles import hydrostatic_column_2d
from src.wcsph import SPHSolver
from src.utils import save_data

DX, H, LX = 0.025, 1.0, 0.2
N_WALL = 4
RHO0, GM, G, DRHO_MAX = 1000.0, 7.0, 9.81, 0.01
C0 = np.sqrt(G*H/DRHO_MAX)
T = 0.4
N_SNAPSHOTS = 200

CASES = (
    ("periodic", True, "bottom", 1.3),
    ("rigid", False, "bottom", 2.0),
)
STARTS = ("hydrostatic", "uniform")


def top_ghost_density(xw):
    """Exact Tait density on top ghosts; leave bottom walls to Adami."""
    return np.where(xw[:, 1] > H,
                    hydrostatic_density(xw[:, 1], H, RHO0, GM, C0, G),
                    np.nan)


def run_case(case_name, periodic, walls, h_factor, start):
    layout = hydrostatic_column_2d(
        DX, H, LX, rho0=RHO0, gm=GM, c0=C0, g=G, start=start,
        n_wall=N_WALL, walls=walls, periodic=periodic, diamond=True,
    )
    rho_ref = lambda x: hydrostatic_density(x[:, 1], H, RHO0, GM, C0, G)
    solver = SPHSolver(
        dim=2, dx=layout.dx, h_factor=h_factor, H=H,
        rho0=RHO0, gm=GM, c0=C0, accel=np.array([0., -G]),
        alpha=0.1, delta=0.1, rho_ref=rho_ref,
        wall_density=top_ghost_density, T=T,
        periodic_axes=(0,) if periodic else (), L=layout.L,
    )
    out = solver.run(
        layout.x, layout.v, layout.rho, layout.wall,
        save_every=max(1, solver.n_steps//N_SNAPSHOTS),
    )
    return dict(
        t=out["t"], x=out["x"], v=out["v"], rho=out["rho"],
        fluid=~layout.wall, wall=out["wall"], L=np.asarray(layout.L, dtype=float),
        h=H, B=solver.B, mass=solver.mass, dx=solver.dx, dt=solver.dt,
        steps=solver.n_steps, rho0=RHO0, gm=GM, g=G, c0=C0,
        alpha=solver.alpha, delta=solver.delta,
        case=case_name, start=start, periodic=periodic, walls=walls,
        h_factor=h_factor, n_snapshots=N_SNAPSHOTS,
    )


if __name__ == "__main__":
    for name, periodic, walls, h_factor in CASES:
        for start in STARTS:
            save_data(f"data_{name}_{start}",
                      run_case(name, periodic, walls, h_factor, start))
