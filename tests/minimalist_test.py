import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.eqs import hydrostatic_pressure, tait
from src.particles import hydrostatic_column_1d
from src.wcsph import SPHSolver
import numpy as np

dx = 0.08
h = 1.0
t = 8
rho0 = 1000.0
gm = 7.0
g = 9.81 

def run(start):
    case = hydrostatic_column_1d(dx, h, rho0=rho0, gm=gm, g=g, c0=np.sqrt(g*h/0.01), start='uniform', n_wall=4, n_ghost=2)
    s = SPHSolver(
        dim=1, dx=case.dx, H=h, rho0=rho0, gm=gm, 
        alpha=0.1, delta=0.1, accel=np.array([-g]), T=t
    )
    o = s.run(case.x, case.v, case.rho, case.wall, save_every=max(1, s.n_steps//200))

    y = o["x"][-1][~case.wall, 0]
    p = tait(s.B, s.rho0, o["rho"][-1][~case.wall], s.gm)
    err = np.abs(p - hydrostatic_pressure(y, h, rho0, gm, s.c0, g))/(rho0*g*h)
    ke = 0.5*s.mass*np.sum(o["v"][:, ~case.wall]**2, axis=(1, 2))

    return {"h": h, "rho0": rho0, "gm": gm, "g": g, "alpha": s.alpha, 
            "t": o["t"], "x": o["x"], "v": o["v"], "rho": o["rho"], "p":p, "ke": ke, "err": err, "c0": s.c0,
                "fluid": ~case.wall, "B": s.B, "mass": s.mass, "dx": s.dx, "dt": s.dt, "steps": s.n_steps}


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from src.eqs import hydrostatic_pressure, tait
    from src.utils import load

    data = run('uniform')
    np.savez_compressed('data.npz', **data)
    print(f"wrote data.npz")


    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    d = load("data.npz")
    h, rho0, gm = d["h"], d["rho0"], d["gm"]
    c0, g = d["c0"], d["g"]
    fluid = d["fluid"]
    y = d["x"][-1][fluid, 0]

    ax[0].plot(y, d["p"]/1e3, ".", ms=3)
    ax[1].semilogy(y, d["err"], ".", ms=3)
    ax[2].semilogy(d["t"][1:], d["ke"][1:])

    yy = np.linspace(0, h, 200)
    ax[0].plot(yy, hydrostatic_pressure(yy, h, rho0, gm, c0, g)/1e3,
               "k--", lw=2, alpha=0.6, label="analytic")
    ax[0].set(xlabel="y", ylabel="p (kPa)")
    ax[1].set(xlabel="y", ylabel=r"$|p-p_{exact}|/(\rho g h)$")
    ax[2].set(xlabel="t", ylabel="kinetic energy")
    plt.show()
