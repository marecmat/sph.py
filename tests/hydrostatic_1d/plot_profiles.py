import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.eqs import hydrostatic_pressure, tait
from src.utils import load


if __name__ == "__main__":
    starts = ("uniform", "hydrostatic")
    data = {start: load(f"data_{start}.npz") for start in starts}
    first = data[starts[0]]
    h, rho0, gm = float(first["h"]), float(first["rho0"]), float(first["gm"])
    C0, G = float(first["c0"]), float(first["g"])

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")

    for start, d in data.items():
        fluid = d["fluid"]
        y = d["x"][-1][fluid, 0]
        p = tait(float(d["B"]), rho0, d["rho"][-1][fluid], gm)
        err = np.abs(p - hydrostatic_pressure(y, h, rho0, gm, C0, G))/(rho0*G*h)
        ke = 0.5*float(d["mass"])*np.sum(d["v"][:, fluid]**2, axis=(1, 2))

        ax[0].plot(y, p/1e3, ".", ms=3, label=start)
        ax[1].semilogy(y, err + 1e-30, ".", ms=3, label=start)
        ax[2].semilogy(d["t"][1:], ke[1:], label=start)

    yy = np.linspace(0, h, 200)
    ax[0].plot(yy, hydrostatic_pressure(yy, h, rho0, gm, C0, G)/1e3,
               "k--", lw=2, alpha=0.6, label="analytic")
    ax[0].set(xlabel="y", ylabel="p (kPa)")
    ax[1].set(xlabel="y", ylabel=r"$|p-p_{exact}|/(\rho g h)$")
    ax[2].set(xlabel="t", ylabel="kinetic energy")
    for a in ax:
        a.legend()
    plt.show()
