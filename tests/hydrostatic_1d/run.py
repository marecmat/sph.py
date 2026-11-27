import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.eqs import hydrostatic_density, hydrostatic_pressure, tait
from src.particles import hydrostatic_column_1d
from src.wcsph import SPHSolver
from src.utils import save_data

dx = 0.02
t  = 15
h  = 1.0
rho0, gm, g, drho_max = 1000.0, 7.0, 9.81, 0.01
c0 = np.sqrt(g*h/drho_max)
starts = ("hydrostatic", "uniform")
n_snapshots = 200 # target number of saved snapshots per start


def top_ghost_density(xw): #Exact Tait density on top ghosts; leave bottom walls to Adami.
    return np.where(xw[:, 0] > h, hydrostatic_density(xw[:, 0], h, rho0, gm, c0, g), np.nan)


def run(start):
    case = hydrostatic_column_1d(dx, h, rho0=rho0, gm=gm, c0=c0, g=g, start='hydrostatic_tait', n_wall=4)
    rho_ref = lambda x: hydrostatic_density(x[:, 0], h, rho0, gm, c0, g)
    
    solver = SPHSolver(
        dim=1, dx=case.dx, H=h, rho0=rho0, gm=gm, c0=c0, 
        h_factor=2., alpha=0.01, delta=0.01, rho_ref=rho_ref,
        accel=np.array([-g]), wall_density=top_ghost_density, T=t, periodic_axes=(),
    )
    
    out = solver.run(case.x, case.v, case.rho, case.wall, save_every=max(1, solver.n_steps//n_snapshots))

    fluid = ~case.wall
    y = out["x"][-1][fluid, 0]
    p = tait(solver.B, solver.rho0, out["rho"][-1][fluid], solver.gm)
    err = np.abs(p - hydrostatic_pressure(y, h, rho0, gm, c0, g))/(rho0*g*h)
    bottom = err[y < 0.05*h]
    ke = 0.5*solver.mass*np.sum(out["v"][:, fluid]**2, axis=(1, 2))
    
    return {
        "t": out["t"], "x": out["x"], "v": out["v"], "rho": out["rho"], "fluid": fluid, "h": h,
        "B": solver.B, "mass": solver.mass, "dx": solver.dx, "dt": solver.dt, "steps": solver.n_steps, "starts": np.array(starts), 
        "rho0": solver.rho0, "gm": solver.gm, "g": np.abs(solver.accel[-1]), "c0": solver.c0,
        "alpha": solver.alpha,"delta": solver.delta,"n_snapshots": n_snapshots
    }


if __name__ == "__main__":
    # for start in starts:
    #     data = run(start)
    #     save_data(f"data_{start}", data)
    data = run("hydrostatic_tait")
    save_data(f"data_hydrostatic_tait", data)