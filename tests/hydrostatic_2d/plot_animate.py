import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.eqs import tait
from src.utils import load

CASES = ("periodic", "rigid")
STARTS = ("hydrostatic", "uniform")
keys = [(case, start) for case in CASES for start in STARTS]


if __name__ == "__main__":
    data = {key: load(f"data_{key[0]}_{key[1]}.npz") for key in keys}
    n_frames = len(data[keys[0]]["t"])

    def pressure(d, k, fluid):
        return tait(float(d["B"]), float(d["rho0"]),
                    d["rho"][k][fluid], float(d["gm"]))/1e3

    pmax = max(pressure(d, -1, d["fluid"]).max() for d in data.values())
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    scatters = {}
    for ax, key in zip(axes.flat, keys):
        d = data[key]
        fluid = d["fluid"]
        x = d["x"][0][fluid]
        scatters[key] = ax.scatter(
            x[:, 0], x[:, 1], c=pressure(d, 0, fluid), s=8,
            cmap="viridis", vmin=0, vmax=pmax,
        )
        ax.set(title=f"{key[0]} / {key[1]}", xlabel="x", ylabel="y",
               aspect="equal")
    fig.colorbar(scatters[keys[0]], ax=axes, label="p (kPa)", shrink=0.8)
    title = fig.suptitle("")

    def update(k):
        for key, sc in scatters.items():
            d = data[key]
            fluid = d["fluid"]
            sc.set_offsets(d["x"][k][fluid])
            sc.set_array(pressure(d, k, fluid))
        title.set_text(f"t = {data[keys[0]]['t'][k]:.2f}")
        return [*scatters.values(), title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100)
    plt.show()
