import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.particles import channel_2d
from src.wcsph import SPHSolver
from src.utils import save_data

DX, H, LX = 0.05, 1.0, 0.4
N_WALL = 3
RHO0, GM, NU, F, MACH = 1000.0, 7.0, 0.1, 1.0, 0.1
U_MAX = F*H**2/(8*NU)
C0 = U_MAX/MACH
T = 3.0
N_SNAPSHOTS = 200


def run_case():
    layout = channel_2d(DX, H, LX, rho0=RHO0, n_wall=N_WALL, diamond=True)
    solver = SPHSolver(
        dim=2, dx=layout.dx, rho0=RHO0, gm=GM, c0=C0,
        accel=np.array([F, 0.0]), viscosity="laminar", nu_phys=NU,
        delta=0.0, wall_noslip=True,
        periodic_axes=(0,), L=layout.L, T=T,
    )
    out = solver.run(
        layout.x, layout.v, layout.rho, layout.wall,
        save_every=max(1, solver.n_steps//N_SNAPSHOTS),
    )
    return dict(
        t=out["t"], x=out["x"], v=out["v"], rho=out["rho"],
        fluid=~layout.wall, wall=out["wall"],
        L=np.asarray(layout.L, dtype=float), H=H, LX=LX,
        B=solver.B, mass=solver.mass, dx=solver.dx, dt=solver.dt,
        steps=solver.n_steps, rho0=RHO0, gm=GM, c0=C0,
        nu=NU, body_force=F, u_max=U_MAX, n_snapshots=N_SNAPSHOTS,
    )


if __name__ == "__main__":
    save_data("data", run_case())
