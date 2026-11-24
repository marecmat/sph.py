"""2D Poiseuille flow: periodic channel driven by a body force.

Particles start on a staggered (diamond) lattice.  The only discretisation
input is ``DX``; particle counts follow from ``DX`` and the domain size.

Run from anywhere:  python tests/poiseuille_2d.py [--no-show]
"""

import numpy as np
import matplotlib.pyplot as plt

from common import finish
from src.particles import channel_2d
from src.wcsph import SPHSolver


DX, H, LX = 0.05, 1.0, 0.4
N_WALL = 3
RHO0, GM, NU, F, MACH = 1000.0, 7.0, 0.1, 1.0, 0.1
U_MAX = F*H**2/(8*NU)
C0 = U_MAX/MACH
T = 3.0

layout = channel_2d(
    DX, H, LX, rho0=RHO0, n_wall=N_WALL, diamond=True,
)
solver = SPHSolver(
    dim=2, dx=layout.dx, rho0=RHO0, gm=GM, c0=C0, g=0.0, body_force=F,
    viscosity="laminar", nu_phys=NU, delta=0.0, wall_noslip=True,
    periodic_axes=(0,), L=layout.L, T=T,
)
out = solver.run(
    layout.x, layout.v, layout.rho, layout.wall,
    save_every=max(1, solver.n_steps//200),
)

fluid = ~layout.wall
xs = out["x"][-1][fluid]
vx = out["v"][-1][fluid, 0]
u_exact = F/(2*NU)*xs[:, 1]*(H - xs[:, 1])
err = np.abs(vx - u_exact)/U_MAX
KE = 0.5*solver.mass*np.sum(out["v"][:, fluid]**2, axis=(1, 2))

print(f"[poiseuille] dx={layout.dx:.3e}  steps={solver.n_steps}")
print(f"   max |u - u_exact|/u_max = {err.max():.2e}")
print(f"   max |vy| = {np.abs(out['v'][-1][fluid, 1]).max():.2e}")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
ax[0].scatter(xs[:, 1], vx, s=10, label="SPH")
yy = np.linspace(0, H, 200)
ax[0].plot(yy, F/(2*NU)*yy*(H - yy), "k--", lw=2, alpha=0.6, label="analytic")
ax[0].set(xlabel="y", ylabel="vx")
ax[1].semilogy(xs[:, 1], err + 1e-30, ".", ms=3)
ax[1].set(xlabel="y", ylabel=r"$|u-u_{exact}|/u_{max}$")
ax[2].semilogy(out["t"][1:], KE[1:])
ax[2].set(xlabel="t", ylabel="kinetic energy")
ax[0].legend()

finish(fig, "poiseuille_2d.png")
