import numpy as np
import matplotlib.pyplot as plt
from src.eqs import hydrostatic_pressure, tait
from src.utils import load


if __name__ == "__main__":
    d = load("data.npz")
    starts = tuple(str(s) for s in d["starts"])
    h, rho0, gm = float(d["h"]), float(d["rho0"]), float(d["gm"])
    C0, G = float(d["c0"]), float(d["g"])

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")

    for start in starts:
        fluid = d[f"{start}_fluid"]
        y = d[f"{start}_x"][-1][fluid, 0]
        p = tait(float(d[f"{start}_B"]), rho0,
                 d[f"{start}_rho"][-1][fluid], gm)
        err = np.abs(p - hydrostatic_pressure(y, h, rho0, gm, C0, G))/(rho0*G*h)
        ke = (0.5*float(d[f"{start}_mass"])
              * np.sum(d[f"{start}_v"][:, fluid]**2, axis=(1, 2)))

        ax[0].plot(y, p/1e3, ".", ms=3, label=start)
        ax[1].semilogy(y, err + 1e-30, ".", ms=3, label=start)
        ax[2].semilogy(d[f"{start}_t"][1:], ke[1:], label=start)

    yy = np.linspace(0, h, 200)
    ax[0].plot(yy, hydrostatic_pressure(yy, h, rho0, gm, C0, G)/1e3,
               "k--", lw=2, alpha=0.6, label="analytic")
    ax[0].set(xlabel="y", ylabel="p (kPa)")
    ax[1].set(xlabel="y", ylabel=r"$|p-p_{exact}|/(\rho g h)$")
    ax[2].set(xlabel="t", ylabel="kinetic energy")
    for a in ax:
        a.legend()
    plt.show()