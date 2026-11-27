import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.particles import dam_break_2d
from src.wcsph import SPHSolver
from src.utils import save_data

dx = 0.01
h = 1.0
lx = 2.0
n_wall = 3
fluid_width, fluid_height = 0.4, 0.5
rho0, gm, g, drho_max = 1000.0, 7.0, 9.81, 0.01
C0 = np.sqrt(g*h/drho_max)
T = 10
N_SNAPSHOTS = 200


def run_case():
    layout = dam_break_2d(
        dx, h, lx, rho0=rho0, n_wall=n_wall,
        fluid_width=fluid_width, fluid_height=fluid_height, diamond=True,
    )
    solver = SPHSolver(
        dim=2, dx=layout.dx, H=h, rho0=rho0, gm=gm, c0=C0,
        accel=np.array([0., -g]), viscosity="artificial",
        alpha=0.1, delta=0.1, rho_ref=None,
        periodic_axes=(), L=layout.L, T=T,
    )
    out = solver.run(
        layout.x, layout.v, layout.rho, layout.wall,
        save_every=max(1, solver.n_steps//N_SNAPSHOTS),
    )
    return dict(
        t=out["t"], x=out["x"], v=out["v"], rho=out["rho"],
        fluid=~layout.wall, wall=out["wall"],
        L=np.asarray(layout.L, dtype=float), h=H, LX=LX,
        n_wall=N_WALL, fluid_width=FLUID_WIDTH, fluid_height=FLUID_HEIGHT,
        B=solver.B, mass=solver.mass, dx=solver.dx, dt=solver.dt,
        steps=solver.n_steps, rho0=RHO0, gm=GM, g=G, c0=C0,
        alpha=solver.alpha, delta=solver.delta, n_snapshots=N_SNAPSHOTS,
    )


if __name__ == "__main__":
    save_data("data", run_case())
