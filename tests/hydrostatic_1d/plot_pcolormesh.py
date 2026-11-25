import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load
from src.eqs import tait

map_levels = 120


if __name__ == "__main__":
    starts = ("uniform", "hydrostatic")
    data = {start: load(f"data_{start}.npz") for start in starts}
    first = data[starts[0]]
    h, rho0, gm = float(first["h"]), float(first["rho0"]), float(first["gm"])

    maps = {}
    y_map = np.linspace(0, h, map_levels)
    for start, d in data.items():
        fluid = d["fluid"]
        p_map = np.empty((len(d["t"]), y_map.size))
        v_map = np.empty_like(p_map)
        for k in range(len(d["t"])):
            ys = d["x"][k][fluid, 0]
            order = np.argsort(ys)
            p_map[k] = np.interp(y_map, ys, tait(float(d["B"]), rho0,
                                d["rho"][k][fluid][order], gm)/1e3)
            v_map[k] = np.interp(y_map, ys, d["v"][k][fluid, 0][order])
        maps[start] = (p_map, v_map)

    pmin = min(m[0].min() for m in maps.values())
    pmax = max(m[0].max() for m in maps.values())
    vmax = max(np.abs(d["v"][:, d["fluid"], 0]).max() for d in data.values())
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    for row, start in enumerate(starts):
        p_map, v_map = maps[start]
        t = data[start]["t"]
        m_p = axes[row, 0].pcolormesh(t, y_map, p_map.T, shading="nearest",
                                     cmap="viridis", vmin=pmin, vmax=pmax)
        fig.colorbar(m_p, ax=axes[row, 0], label="p (kPa)")
        axes[row, 0].set_title(f"{start}: pressure")

        m_v = axes[row, 1].pcolormesh(t, y_map, v_map.T, shading="nearest",
                                     cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(m_v, ax=axes[row, 1], label=r"$v_y$ (m/s)")
        axes[row, 1].set_title(f"{start}: velocity")

        for a in axes[row]:
            a.set(xlabel="t", ylabel="y", ylim=(0, h))

    plt.show()
