
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from wcsph import Params, tait, run
from matplotlib.animation import FuncAnimation, PillowWriter

N, Nx, T = 20, 8, 20
STARTS = ["hydrostatic", "uniform"]
results = []
for ii, start in enumerate(STARTS):

    p = Params(N=N, Nx=Nx, T=T, start=start, alpha=0.1, delta=0.3, h_factor=1.3)
    out = run(p, save_every=max(1, p.n_steps//200))

    particle_positions = out["x"][:, out["fluid"], :]
    particle_pressures = np.array([
        tait(p, rho[out["fluid"]]) / 1e3 for rho in out["rho"]
    ])

    results.append({
        "start": start,
        "p": p,
        "out": out,
        "positions": particle_positions,
        "pressures": particle_pressures
    })


fig, ax = plt.subplots(1, 2, figsize=(8, 5), layout="constrained")
vmin = min(r["pressures"].min() for r in results)
vmax = max(r["pressures"].max() for r in results)

scatters = []
time_texts = []

for ii, result in enumerate(results):
    p = result["p"]
    positions = result["positions"]
    pressures = result["pressures"]
    sc = ax[ii].scatter( positions[0, :, 0], positions[0, :, 1], c=pressures[0], 
                                                s=10, cmap="viridis", vmin=vmin, vmax=vmax)
    scatters.append(sc)

    ax[ii].set( xlabel="x", ylabel="y", title=f"({result['start']} start)", xlim=(0, p.Lx), ylim=(0, p.H))
    time_texts.append(ax[ii].text( 0.03, 0.97, "", transform=ax[ii].transAxes, va="top"))
fig.colorbar( scatters[0], ax=ax, label="p (kPa)", shrink=0.9)

def update(frame):
    for ii, result in enumerate(results):
        pos = result["positions"][frame]
        pressure = result["pressures"][frame]
        scatters[ii].set_offsets(pos[:, :2])
        scatters[ii].set_array(pressure)
        time_texts[ii].set_text( f"t = {result['out']['t'][frame]:.4f} s")
    return scatters + time_texts

ani = FuncAnimation(fig, update, frames=min(len(r["out"]["t"]) for r in results), 
                                            interval=40, blit=False, repeat=True)
ani.save("compar.gif", writer=PillowWriter(fps=10), dpi=150)
plt.show()
