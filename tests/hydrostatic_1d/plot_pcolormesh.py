import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load
from src.eqs import tait

map_levels = 120

if __name__ == "__main__":
    d = load('data.npz')
    starts = tuple(str(s) for s in d["starts"])
    h, rho0, gm = float(d["h"]), float(d["rho0"]), float(d["gm"])

    maps = {}
    y_map = np.linspace(0, h, map_levels)
    for start in starts:
        fluid = d[f"{start}_fluid"]
        p_map = np.empty((len(d[f"{start}_t"]), y_map.size))
        v_map = np.empty_like(p_map)
        for k in range(len(d[f"{start}_t"])):
            ys = d[f"{start}_x"][k][fluid, 0]
            order = np.argsort(ys)
            p_map[k] = np.interp(y_map, ys, tait(float(d[f"{start}_B"]), rho0, 
                                    d[f"{start}_rho"][k][fluid][order], gm)/1e3,)
            v_map[k] = np.interp(y_map, ys, d[f"{start}_v"][k][fluid, 0][order])
        maps[start] = (p_map, v_map)

    pmin = min(m[0].min() for m in maps.values())
    pmax = max(m[0].max() for m in maps.values())
    vmax = max(np.abs(d[f"{start}_v"][:, d[f"{start}_fluid"], 0]).max() for start in starts)
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    for row, start in enumerate(starts):
        p_map, v_map = maps[start]
        t = d[f"{start}_t"]
        m_p = axes[row, 0].pcolormesh(t, y_map, p_map.T, shading="nearest", 
                                    cmap="viridis", vmin=pmin, vmax=pmax)
        fig.colorbar(m_p, ax=axes[row, 0], label="p (kPa)")
        axes[row, 0].set_title(f"{start}: pressure")

        m_v = axes[row, 1].pcolormesh(t, y_map, v_map.T, shading="nearest", 
                                    cmap="RdBu_r", vmin=-vmax, vmax=vmax,)
        fig.colorbar(m_v, ax=axes[row, 1], label=r"$v_y$ (m/s)")
        axes[row, 1].set_title(f"{start}: velocity")

        for a in axes[row]:
            a.set(xlabel="t", ylabel="y", ylim=(0, h))

    plt.show()