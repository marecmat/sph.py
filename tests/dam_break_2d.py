"""2D dam break in a rigid tank.

Particles start on a staggered (diamond) lattice.  The only discretisation
input is ``DX``; particle counts follow from ``DX`` and the tank size.

This case currently checks that the solver produces a stable free-surface flow;
validation against reference front-position data is still TODO.

Run from anywhere:  python tests/dam_break_2d.py [--no-show]
"""

import numpy as np
import matplotlib.pyplot as plt

from common import finish
from src.particles import dam_break_2d
from src.wcsph import SPHSolver


DX, H, LX = 1.0/30.0, 1.0, 2.0
N_WALL = 3
RHO0, GM, G, DRHO_MAX = 1000.0, 7.0, 9.81, 0.01
C0 = np.sqrt(G*H/DRHO_MAX)
T = 1.0

layout = dam_break_2d(
    DX, H, LX, rho0=RHO0, n_wall=N_WALL,
    fluid_width=0.4, fluid_height=0.5, diamond=True,
)
solver = SPHSolver(
    dim=2, dx=layout.dx, H=H, rho0=RHO0, gm=GM, c0=C0,
    accel=np.array([0., -G]),
    viscosity="artificial", alpha=0.1, delta=0.1, rho_ref=None,
    periodic_axes=(), L=layout.L, T=T,
)
out = solver.run(
    layout.x, layout.v, layout.rho, layout.wall,
    save_every=max(1, solver.n_steps//200),
)

fluid = ~layout.wall
x0 = layout.x[fluid]
xf = out["x"][-1][fluid]
speed = np.linalg.norm(out["v"][-1][fluid], axis=1)
KE = 0.5*solver.mass*np.sum(out["v"][:, fluid]**2, axis=(1, 2))

print(f"[dam break] dx={layout.dx:.3e}  steps={solver.n_steps}")
print(f"   fluid particles = {fluid.sum()}")
print(f"   front position: {x0[:, 0].max():.3f} -> {xf[:, 0].max():.3f}")
print(f"   max speed = {speed.max():.3f}")

fig, ax = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
for a, pos, title in ((ax[0], x0, "initial"), (ax[1], xf, "final")):
    a.scatter(pos[:, 0], pos[:, 1], c="C0", s=6)
    a.set(title=title, xlabel="x", ylabel="y", aspect="equal")
ax[2].semilogy(out["t"][1:], KE[1:])
ax[2].set(xlabel="t", ylabel="kinetic energy")

finish(fig, "dam_break_2d.png")
