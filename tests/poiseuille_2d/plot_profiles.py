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
from src.utils import load


if __name__ == "__main__":
    d = load("data.npz")
    fluid = d["fluid"]
    H, NU = float(d["H"]), float(d["nu"])
    F, U_MAX = float(d["body_force"]), float(d["u_max"])

    xs = d["x"][-1][fluid]
    vx = d["v"][-1][fluid, 0]
    u_exact = F/(2*NU)*xs[:, 1]*(H - xs[:, 1])
    err = np.abs(vx - u_exact)/U_MAX
    ke = 0.5*float(d["mass"])*np.sum(d["v"][:, fluid]**2, axis=(1, 2))

    print(f"[poiseuille] dx={float(d['dx']):.3e}  steps={int(d['steps'])}")
    print(f"   max |u - u_exact|/u_max = {err.max():.2e}")
    print(f"   max |vy| = {np.abs(d['v'][-1][fluid, 1]).max():.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    ax[0].scatter(xs[:, 1], vx, s=10, label="SPH")
    yy = np.linspace(0, H, 200)
    ax[0].plot(yy, F/(2*NU)*yy*(H - yy), "k--", lw=2, alpha=0.6,
               label="analytic")
    ax[0].set(xlabel="y", ylabel="vx")
    ax[0].legend()
    ax[1].semilogy(xs[:, 1], err + 1e-30, ".")
    ax[1].set(xlabel="y", ylabel=r"$|u-u_{exact}|/u_{max}$")
    ax[2].semilogy(d["t"][1:], ke[1:])
    ax[2].set(xlabel="t", ylabel="kinetic energy")

    finish(fig, "poiseuille_2d.png")
