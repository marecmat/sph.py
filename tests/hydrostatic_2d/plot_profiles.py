import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import matplotlib.pyplot as plt

from common import finish
from src.eqs import hydrostatic_pressure, tait
from src.utils import load

CASES = ("periodic", "rigid")
STARTS = ("hydrostatic", "uniform")


if __name__ == "__main__":
    data = {(case, start): load(f"data_{case}_{start}.npz")
            for case in CASES for start in STARTS}
    first = data[CASES[0], STARTS[0]]
    h, rho0, gm = float(first["h"]), float(first["rho0"]), float(first["gm"])
    C0, G = float(first["c0"]), float(first["g"])

    fig, ax = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    yy = np.linspace(0, h, 200)

    for row, case in enumerate(CASES):
        for start in STARTS:
            d = data[case, start]
            fluid = d["fluid"]
            xs = d["x"][-1][fluid]
            p = tait(float(d["B"]), float(d["rho0"]),
                     d["rho"][-1][fluid], float(d["gm"]))
            err = np.abs(p - hydrostatic_pressure(
                xs[:, 1], h, rho0, gm, C0, G))/(rho0*G*h)
            ke = 0.5*float(d["mass"])*np.sum(d["v"][:, fluid]**2, axis=(1, 2))

            print(f"[{case} / {start}] dx={float(d['dx']):.3e}  "
                  f"steps={int(d['steps'])}")
            print(f"   max error = {err.max():.2e}, "
                  f"final KE = {ke[-1]:.2e}")

            ax[row, 0].plot(xs[:, 1], p/1e3, ".", ms=2, label=start)
            ax[row, 1].semilogy(xs[:, 1], err + 1e-30, ".", ms=2, label=start)
            ax[row, 2].semilogy(d["t"][1:], ke[1:], label=start)

        ax[row, 0].plot(yy, hydrostatic_pressure(yy, h, rho0, gm, C0, G)/1e3,
                        "k--", lw=2, alpha=0.6, label="analytic")
        ax[row, 0].set_title(case)
        ax[row, 0].set(xlabel="y", ylabel="p (kPa)")
        ax[row, 1].set(xlabel="y", ylabel=r"$|p-p_{exact}|/(\rho g h)$")
        ax[row, 2].set(xlabel="t", ylabel="kinetic energy")
        for a in ax[row]:
            a.legend()

    finish(fig, "hydrostatic_2d_profiles.png")
