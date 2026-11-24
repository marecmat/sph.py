"""2D hydrostatic column in a periodic strip: compare with the analytic equilibrium.

Run:  python run_hydrostatic.py          (add --no-show to only write the PNG)
"""
import sys
import numpy as np
import matplotlib
if "--no-show" in sys.argv: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wcsph import Params, tait, run

N, Nx, T = 50, 8, 0.5            # resolution; Nx*dy must be >= 2*support (Nx >= 6 for h = 1.3 dy)
STARTS = ["hydrostatic", "uniform"]

fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
for start in STARTS:
    p = Params(N=N, Nx=Nx, T=T, start=start, alpha=0.1, delta=0.1, h_factor=1.3)
    print(f"[{start}] dt={p.dt:.2e}  steps={p.n_steps}  particles={len(p.wall)}")
    out = run(p, save_every=max(1, p.n_steps//200))
    fl, xs, rs = out["fluid"], out["x"][-1][out["fluid"]], out["rho"][-1][out["fluid"]]
    P = tait(p, rs)
    err = np.abs(P - p.analytical_pressure(xs[:, 1]))/(p.rho0*p.g*p.H)       # normalised by rho g H
    rows = P.reshape(p.N, p.Nx)
    print(f"   max |p - p_analytic|/(rho g H) = {err.max():.2e}")
    print(f"   column spread (max-min)/mean   = {((rows.max(1) - rows.min(1))/rows.mean(1)).max():.1e}   (should be ~1e-12)")
    print(f"   max |vx| = {np.abs(out['v'][-1][fl, 0]).max():.1e}   max |vy| = {np.abs(out['v'][-1][fl, 1]).max():.1e}")
    ax[0, 0].plot(xs[:, 1], P/1e3, ".", ms=3, label=start)
    ax[0, 1].semilogy(xs[:, 1], err + 1e-30, ".", ms=3, label=start)
    ax[1, 0].semilogy(out["t"][1:], out["KE"][1:], label=start)
    if start == STARTS[0]:
        sc = ax[1, 1].scatter(xs[:, 0], xs[:, 1], c=P/1e3, s=10, cmap="viridis")
        fig.colorbar(sc, ax=ax[1, 1], label="p (kPa)")
yy = np.linspace(0, p.H, 200)
ax[0, 0].plot(yy, p.analytical_pressure(yy)/1e3, "k--", lw=3, alpha=.5, label="analytic")
ax[0, 0].set(xlabel="y", ylabel="p (kPa)"); ax[0, 0].legend()
ax[0, 1].set(xlabel="y", ylabel=r"$|p-p_a|/(\rho g H)$"); ax[0, 1].legend()
ax[1, 0].set(xlabel="t", ylabel="kinetic energy"); ax[1, 0].legend()
ax[1, 1].set(xlabel="x", ylabel="y", title=f"final particles ({STARTS[0]} start)", aspect="equal")
fig.savefig("hydrostatic_2D.png", dpi=130)
print("wrote hydrostatic_2D.png")

plt.show()
