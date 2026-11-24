"""Plane Poiseuille flow in a 2D periodic channel vs the analytic transient solution.

Run:  python run_poiseuille.py          (add --no-show to only write the PNG)
"""
import sys
import numpy as np
import matplotlib
if "--no-show" in sys.argv: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wcsph import poiseuille_params, run

H, NU, F = 1.0, 0.1, 1.0         # gap, kinematic viscosity, driving acceleration
N, Nx, T = 20, 8, 4.0       # T = 4 ~ 4 relaxation times H^2/(pi^2 nu): flow is close to steady
U_MAX = F*H**2/(8*NU)
# Monaghan's nu = alpha h c0 / 8 assumes every pair dissipates; the (v.r < 0) switch removes about half of it in
# shear flow, so the nominal alpha (factor 1) gives a peak velocity ~70% too high. 2.3 was tuned empirically for
# this setup (N=20, h = sqrt(2) dy); it is NOT a general constant - re-tune if you change resolution or h_factor.
ARTIFICIAL_ALPHA_FACTOR = 2.3


def u_exact(y, t, nterms=200):
    u = F/(2*NU)*y*(H - y)
    for n in range(1, 2*nterms, 2):
        u = u - 4*F*H**2/(NU*np.pi**3*n**3)*np.sin(n*np.pi*y/H)*np.exp(-n**2*np.pi**2*NU*t/H**2)
    return u


fig, ax = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
yy = np.linspace(0, H, 200)
for row, visc in zip(ax, ("laminar", "artificial")):
    p = poiseuille_params(H=H, N=N, Nx=Nx, nu=NU, F=F, T=T, viscosity=visc)
    if visc == "artificial":
        p.alpha = ARTIFICIAL_ALPHA_FACTOR*8*NU/(p.h*p.c0)
    print(f"[{visc}] dt={p.dt:.2e} steps={p.n_steps} c0={p.c0:.2f}" + (f" alpha={p.alpha:.3f}" if visc == "artificial" else f" nu={p.nu:.3f}"))
    out = run(p, save_every=max(1, p.n_steps//300))
    t, fl = out["t"], out["fluid"]

    def profile(k):                                   # column-averaged u(y) at saved frame k
        u = out["v"][k][fl, 0].reshape(p.N, p.Nx)
        y = out["x"][k][fl, 1].reshape(p.N, p.Nx)
        return y.mean(1), u.mean(1), np.abs(u - u.mean(1, keepdims=True)).max()

    for frac, col in zip((0.1, 0.3, 1.0), ("C0", "C1", "C2")):
        k = int(frac*(len(t) - 1))
        y, u, _ = profile(k)
        row[0].plot(y, u, ".", color=col, label=f"t={t[k]:.2f}")
        row[0].plot(yy, u_exact(yy, t[k]), "k-", lw=.8)
    y, u, spread = profile(-1)
    err = np.abs(u - u_exact(y, t[-1]))/U_MAX
    umax_t = np.array([profile(k)[1].max() for k in range(len(t))])
    print(f"   peak velocity {u.max():.4f} (exact {U_MAX:.4f}),  max profile error/u_max = {err.max():.3f}")
    print(f"   column spread {spread:.1e},  max|vy| = {np.abs(out['v'][-1][fl, 1]).max():.1e},  "
          f"max |rho/rho0-1| = {np.abs(out['rho'][-1][fl]/p.rho0 - 1).max():.1e}")
    row[0].set(xlabel="y", ylabel="u", title=f"{visc}: profiles (lines = analytic)"); row[0].legend()
    row[1].plot(t, umax_t, label="SPH")
    row[1].plot(t, [u_exact(H/2, tt) for tt in t], "k--", label="analytic")
    row[1].set(xlabel="t", ylabel="peak u", title="peak velocity vs time"); row[1].legend()
    sc = row[2].scatter(out["x"][-1][fl, 0], out["x"][-1][fl, 1], c=out["v"][-1][fl, 0], s=12, cmap="viridis")
    fig.colorbar(sc, ax=row[2], label="u"); row[2].set(xlabel="x", ylabel="y", title="final u", aspect="equal")
fig.savefig("poiseuille_2D.png", dpi=130)
print("wrote poiseuille_2D.png")
if "--no-show" not in sys.argv: plt.show()
