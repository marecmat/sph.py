import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils import load


if __name__ == "__main__":
    d = load("data.npz")
    fluid = d["fluid"]
    wall = ~fluid
    n_frames = len(d["t"])
    margin = 0.05
    xmin, xmax = d["x"][:, :, 0].min(), d["x"][:, :, 0].max()
    ymin, ymax = d["x"][:, :, 1].min(), d["x"][:, :, 1].max()

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    sc_f = ax.scatter(d["x"][0][fluid, 0], d["x"][0][fluid, 1],
                      c="C0", s=8, label="fluid")
    sc_w = ax.scatter(d["x"][0][wall, 0], d["x"][0][wall, 1],
                      c="0.6", s=4, label="wall")
    ax.set(xlabel="x", ylabel="y", aspect="equal",
           xlim=(xmin - margin, xmax + margin),
           ylim=(ymin - margin, ymax + margin))
    ax.legend(fontsize=8)
    title = fig.suptitle("")

    def update(k):
        sc_f.set_offsets(d["x"][k][fluid])
        sc_w.set_offsets(d["x"][k][wall])
        title.set_text(f"t = {d['t'][k]:.2f}")
        return sc_f, sc_w, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=15)
    plt.show()
