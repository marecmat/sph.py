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
    H, NU = float(d["H"]), float(d["nu"])
    F, U_MAX = float(d["body_force"]), float(d["u_max"])
    n_frames = len(d["t"])

    fig, (ax_space, ax_profile) = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained")
    x0 = d["x"][0][fluid]
    sc = ax_space.scatter(x0[:, 0], x0[:, 1], c=d["v"][0][fluid, 0],
                          cmap="RdBu_r", s=10, vmin=-U_MAX, vmax=U_MAX)
    ax_space.set(xlabel="x", ylabel="y", aspect="equal", title="vx field")
    fig.colorbar(sc, ax=ax_space, label=r"$v_x$ (m/s)")

    yy = np.linspace(0, H, 200)
    ax_profile.plot(F/(2*NU)*yy*(H - yy), yy, "k--", lw=2, alpha=0.6,
                    label="analytic")
    points, = ax_profile.plot([], [], "o", ms=4, label="SPH")
    ax_profile.set(xlabel=r"$v_x$ (m/s)", ylabel="y", xlim=(-0.1*U_MAX, 1.1*U_MAX),
                   ylim=(0, H), title="profile")
    ax_profile.legend(fontsize=8)
    title = fig.suptitle("")

    def update(k):
        x = d["x"][k][fluid]
        vx = d["v"][k][fluid, 0]
        sc.set_offsets(x)
        sc.set_array(vx)
        order = np.argsort(x[:, 1])
        points.set_data(vx[order], x[order, 1])
        title.set_text(f"t = {d['t'][k]:.2f}")
        return sc, points, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100)
    plt.show()
