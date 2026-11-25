import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.eqs import hydrostatic_density, hydrostatic_pressure, tait
from src.particles import hydrostatic_column_1d
from src.wcsph import SPHSolver

dx, h = 0.02, 1.0
n_wall = 4
rho0, gm, g, drho_max = 1000.0, 7.0, 9.81, 0.01
c0 = np.sqrt(g*h/drho_max)
h_factor, alpha, delta = 2.0, 0.01, 0.1
t = 5.0
starts = ("hydrostatic", "uniform")
n_snapshots = 200 # target number of saved snapshots per start


def top_ghost_density(xw): #Exact Tait density on top ghosts; leave bottom walls to Adami.
    return np.where(xw[:, 0] > h, hydrostatic_density(xw[:, 0], h, rho0, gm, c0, g), np.nan)


def run(start):
    case = hydrostatic_column_1d(dx, h, rho0=rho0, gm=gm, c0=c0, g=g, start=start, n_wall=n_wall)
    rho_ref = lambda x: hydrostatic_density(x[:, 0], h, rho0, gm, c0, g)
    
    solver = SPHSolver(
        dim=1, dx=case.dx, H=h, rho0=rho0, gm=gm, c0=c0, 
        h_factor=h_factor, alpha=alpha, delta=delta, rho_ref=rho_ref,
        accel=np.array([-g]), wall_density=top_ghost_density, T=t, periodic_axes=(),
    )
    
    out = solver.run(
        case.x, case.v, case.rho, case.wall,
        save_every=max(1, solver.n_steps//n_snapshots),
    )

    fluid = ~case.wall
    y = out["x"][-1][fluid, 0]
    p = tait(solver.B, solver.rho0, out["rho"][-1][fluid], solver.gm)
    err = np.abs(p - hydrostatic_pressure(y, h, rho0, gm, c0, g))/(rho0*g*h)
    bottom = err[y < 0.05*h]
    ke = 0.5*solver.mass*np.sum(out["v"][:, fluid]**2, axis=(1, 2))

    return {
        f"{start}_t": out["t"],
        f"{start}_x": out["x"],
        f"{start}_v": out["v"],
        f"{start}_rho": out["rho"],
        f"{start}_fluid": fluid,
        f"{start}_B": solver.B,
        f"{start}_mass": solver.mass,
        f"{start}_dx": solver.dx,
        f"{start}_dt": solver.dt,
        f"{start}_steps": solver.n_steps,
    }


if __name__ == "__main__":
    # Simulation parameters travel with the snapshots: the figure scripts
    # read them back from the npz instead of importing a config module.
    data = {
        "starts": np.array(starts),
        "dx": dx,
        "h": h,
        "n_wall": n_wall,
        "rho0": rho0,
        "gm": gm,
        "g": g,
        "drho_max": drho_max,
        "c0": c0,
        "h_factor": h_factor,
        "alpha": alpha,
        "delta": delta,
        "t": t,
        "n_snapshots": n_snapshots,
    }
    for start in starts:
        data.update(run(start))
    np.savez_compressed('data.npz', **data)
    print(f"wrote data.npz")
