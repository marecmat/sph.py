import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.eqs import hydrostatic_pressure, tait
from src.particles import hydrostatic_column_1d
from src.wcsph import SPHSolver
import numpy as np

dx = 0.01
h = 1.0
t = 50
n_wall = 4
rho0 = 1000.0
gm = 7.0
g = 9.81 
drho_max = 0.01
c0 = np.sqrt(g*h/drho_max)
alpha = 0.1
n_snapshots = 200 # target number of saved snapshots per start

def run(start):
    case = hydrostatic_column_1d(dx, h, rho0=rho0, gm=gm, c0=c0, g=g, start='uniform', n_wall=n_wall)
    
    s = SPHSolver(dim=1, dx=case.dx, H=h, rho0=rho0, gm=gm, c0=c0, wall_density=None, # top_ghost_density, 
        h_factor=2.0, alpha=alpha, delta=0, accel=np.array([-g]), T=t, periodic_axes=())
    o = s.run(case.x, case.v, case.rho, case.wall, save_every=max(1, s.n_steps//n_snapshots))

    fluid = ~case.wall
    y = o["x"][-1][fluid, 0]
    p = tait(s.B, s.rho0, o["rho"][-1][fluid], s.gm)
    err = np.abs(p - hydrostatic_pressure(y, h, rho0, gm, c0, g))/(rho0*g*h)
    ke = 0.5*s.mass*np.sum(o["v"][:, fluid]**2, axis=(1, 2))

    return { "t": o["t"], "x": o["x"], "v": o["v"], "rho": o["rho"], "p":p, "ke": ke, "err": err,
                "fluid": fluid, "B": s.B, "mass": s.mass, "dx": s.dx, "dt": s.dt, "steps": s.n_steps}


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from src.eqs import hydrostatic_pressure, tait
    from src.utils import load

    if True:
        data = { "dx": dx, "h": h, "n_wall": n_wall, "rho0": rho0, "gm": gm,
            "g": g, "drho_max": drho_max, "c0": c0, "alpha": alpha,
            "t": t, "n_snapshots": n_snapshots}

        data.update(run('uniform'))
        np.savez_compressed('data.npz', **data)
        print(f"wrote data.npz")


    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    d = load("data.npz")
    h, rho0, gm = d["h"], d["rho0"], d["gm"]
    C0, G = d["c0"], d["g"]
    fluid = d["fluid"]
    y = d["x"][-1][fluid, 0]
    print(y.shape, d["p"].shape)
    ax[0].plot(y, d["p"]/1e3, ".", ms=3)
    ax[1].semilogy(y, d["err"], ".", ms=3)
    ax[2].semilogy(d["t"][1:], d["ke"][1:])

    yy = np.linspace(0, h, 200)
    ax[0].plot(yy, hydrostatic_pressure(yy, h, rho0, gm, C0, G)/1e3,
               "k--", lw=2, alpha=0.6, label="analytic")
    ax[0].set(xlabel="y", ylabel="p (kPa)")
    ax[1].set(xlabel="y", ylabel=r"$|p-p_{exact}|/(\rho g h)$")
    ax[2].set(xlabel="t", ylabel="kinetic energy")
    plt.show()
