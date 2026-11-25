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
from src.eqs import tait
from src.utils import load

CASES = ("periodic", "rigid")
STARTS = ("hydrostatic", "uniform")


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for row, case in enumerate(CASES):
        for col, start in enumerate(STARTS):
            d = load(f"data_{case}_{start}.npz")
            fluid = d["fluid"]
            x = d["x"][-1][fluid]
            p = tait(float(d["B"]), float(d["rho0"]),
                     d["rho"][-1][fluid], float(d["gm"]))/1e3
            ax = axes[row, col]
            m = ax.tricontourf(x[:, 0], x[:, 1], p, levels=40, cmap="viridis")
            fig.colorbar(m, ax=ax, label="p (kPa)")
            ax.set(title=f"{case} / {start}", xlabel="x", ylabel="y",
                   aspect="equal")
    finish(fig, "hydrostatic_2d_pcolormesh.png")
