import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.utils import load
from src.eqs import hydrostatic_pressure, tait

anim_frames = 60    # number of frames in animation.gif

def main():
    d = load('data.npz')
    starts = tuple(str(s) for s in d["starts"])
    H, RHO0, GM = float(d["h"]), float(d["rho0"]), float(d["gm"])
    C0, G = float(d["c0"]), float(d["g"])

    n_snap = len(d[f"{starts[0]}_t"])
    frame_ids = np.unique(
        np.linspace(0, n_snap - 1, min(anim_frames, n_snap), dtype=int)
    )
    vmax = max(
        np.abs(d[f"{start}_v"][:, d[f"{start}_fluid"], 0]).max()
        for start in starts
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
            fluid = d[f"{start}_fluid"]
            ys = d[f"{start}_x"][idx][fluid, 0]
            order = np.argsort(ys)
            ps = tait(float(d[f"{start}_B"]), RHO0,
                      d[f"{start}_rho"][idx][fluid][order], GM)/1e3
            vs = d[f"{start}_v"][idx][fluid, 0][order]
            lines_p[start].set_data(ps, ys[order])
            lines_v[start].set_data(vs, ys[order])
        title.set_text(f"t = {d[f'{starts[0]}_t'][idx]:.2f}")
        return [*lines_p.values(), *lines_v.values(), title]

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=100)
    # anim.save("animation.gif", writer=PillowWriter(fps=30), dpi=110)
    plt.show()


if __name__ == "__main__":
    main()
