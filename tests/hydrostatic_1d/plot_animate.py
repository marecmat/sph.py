import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from src.utils import load
from src.eqs import hydrostatic_pressure, tait

anim_frames = 300
starts = ("uniform", "hydrostatic")


def main():
    files = {s: load(f"data_{s}.npz") for s in starts}

    def get(start, key):
        return files[start][key]

    first = files[starts[0]]
    H, RHO0, GM = first["h"], first["rho0"], first["gm"]
    C0, G = first["c0"], first["g"]

    n_snap = len(get(starts[0], "t"))
    frame_ids = np.unique(
        np.linspace(0, n_snap - 1, min(anim_frames, n_snap), dtype=int)
    )
    vmax = max(
        np.abs(get(s, "v")[:, get(s, "fluid"), 0]).max() for s in starts
    )
    vmax = max(vmax, 1e-6)

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    lines_p, lines_v = {}, {}
    for start in starts:
        lines_p[start], = ax_p.plot([], [], "o-", ms=3, lw=1, label=start)
        lines_v[start], = ax_v.plot([], [], "o-", ms=3, lw=1, label=start)

    yy = np.linspace(0, H, 200)
    ax_p.plot(hydrostatic_pressure(yy, H, RHO0, GM, C0, G)/1e3, yy,
              "k--", lw=2, alpha=0.6, label="analytic")
    ax_p.set(xlabel="p (kPa)", ylabel="y",
             xlim=(0, 1.05*hydrostatic_pressure(0, H, RHO0, GM, C0, G)/1e3),
             ylim=(0, H))
    ax_v.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax_v.set(xlabel=r"$v_y$ (m/s)", ylabel="y",
             xlim=(-1.1*vmax, 1.1*vmax), ylim=(0, H))
    for a in (ax_p, ax_v):
        a.legend()
    title = fig.suptitle("")

    def update(k):
        idx = frame_ids[k]
        for start in starts:
            fluid = get(start, "fluid")
            ys = get(start, "x")[idx][fluid, 0]
            order = np.argsort(ys)
            ps = tait(float(get(start, "B")), RHO0,
                      get(start, "rho")[idx][fluid][order], GM)/1e3
            vs = get(start, "v")[idx][fluid, 0][order]
            lines_p[start].set_data(ps, ys[order])
            lines_v[start].set_data(vs, ys[order])
        title.set_text(f"t = {get(starts[0], 't')[idx]:.2f}")
        return [*lines_p.values(), *lines_v.values(), title]

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=100)
    # anim.save("animation.gif", writer=PillowWriter(fps=30), dpi=110)
    plt.show()


if __name__ == "__main__":
    main()