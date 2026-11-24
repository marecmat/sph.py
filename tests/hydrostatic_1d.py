"""1D hydrostatic column: compare the computed pressure with the analytic profile.

Run from anywhere:  python tests/hydrostatic_1d.py [--no-show]
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import finish
from _hydrostatic import hydrostatic_density, hydrostatic_pressure
from src.eqs import tait
from src.particles import hydrostatic_column_1d
from src.wcsph import SPHSolver


DX, H = 0.02, 1.0
N_WALL = 4
RHO0, GM, G, DRHO_MAX = 1000.0, 7.0, 9.81, 0.01
C0 = np.sqrt(G*H/DRHO_MAX)
# The old 1D case used h = 2 dx: with h = 1.3 dx the first fluid layer only
# sees two wall layers, which biases the pressure near y = 0.
H_FACTOR, ALPHA, DELTA = 2.0, 0.01, 0.1
T = 5.0

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")

for start in ("hydrostatic", "uniform"):
    layout = hydrostatic_column_1d(
        DX, H, rho0=RHO0, gm=GM, c0=C0, g=G, start=start, n_wall=N_WALL,
    )
    rho_ref = lambda x: hydrostatic_density(x[:, 0], H, RHO0, GM, C0, G)
    solver = SPHSolver(
        dim=1, dx=layout.dx, H=H, rho0=RHO0, gm=GM, c0=C0, g=G,
        h_factor=H_FACTOR, alpha=ALPHA, delta=DELTA, rho_ref=rho_ref,
        T=T, periodic_axes=(),
    )
    out = solver.run(
        layout.x, layout.v, layout.rho, layout.wall,
        save_every=max(1, solver.n_steps//200),
    )

    fluid = ~layout.wall
    y = out["x"][-1][fluid, 0]
    rho = out["rho"][-1][fluid]
    P = tait(solver.B, solver.rho0, rho, solver.gm)
    err = np.abs(P - hydrostatic_pressure(y, H, RHO0, GM, C0, G))/(RHO0*G*H)
    bottom = err[y < 0.05*H]
    KE = 0.5*solver.mass*np.sum(out["v"][:, fluid]**2, axis=(1, 2))

    print(f"[{start}] dx={layout.dx:.3e}  steps={solver.n_steps}")
    print(f"   max |p - p_exact|/(rho g H) = {err.max():.2e}"
          f"  (near y=0: {bottom.max():.2e})")
    print(f"   max |v| = {np.abs(out['v'][-1][fluid]).max():.2e}")

    ax[0].plot(y, P/1e3, ".", ms=3, label=start)
    ax[1].semilogy(y, err + 1e-30, ".", ms=3, label=start)
    ax[2].semilogy(out["t"][1:], KE[1:], label=start)

yy = np.linspace(0, H, 200)
ax[0].plot(yy, hydrostatic_pressure(yy, H, RHO0, GM, C0, G)/1e3,
           "k--", lw=2, alpha=0.6, label="analytic")
ax[0].set(xlabel="y", ylabel="p (kPa)")
ax[1].set(xlabel="y", ylabel=r"$|p-p_{exact}|/(\rho g H)$")
ax[2].set(xlabel="t", ylabel="kinetic energy")
for a in ax:
    a.legend()

finish(fig, "hydrostatic_1d.png")
