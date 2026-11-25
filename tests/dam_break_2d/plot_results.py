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
    wall = ~fluid
    x0, xf = d["x"][0], d["x"][-1]
    speed = np.linalg.norm(d["v"][-1][fluid], axis=1)
    ke = 0.5*float(d["mass"])*np.sum(d["v"][:, fluid]**2, axis=(1, 2))

    print(f"[dam break] dx={float(d['dx']):.3e}  steps={int(d['steps'])}")
    print(f"   fluid particles = {fluid.sum()}")
    print(f"   front position: {x0[fluid, 0].max():.3f} -> {xf[fluid, 0].max():.3f}")
    print(f"   max speed = {speed.max():.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for a, pos, title in ((ax[0], x0, "initial"), (ax[1], xf, "final")):
        a.scatter(pos[wall, 0], pos[wall, 1], c="0.6", s=4, label="wall")
        a.scatter(pos[fluid, 0], pos[fluid, 1], c="C0", s=6, label="fluid")
        a.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        a.legend(fontsize=8)
    ax[2].semilogy(d["t"][1:], ke[1:])
    ax[2].set(xlabel="t", ylabel="kinetic energy")

    finish(fig, "dam_break_2d.png")
