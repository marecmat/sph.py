import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/")
import kernels 
from matplotlib import rcParams
rcParams['text.usetex'] = True

from misc import save_to_logs

def sph_approx(smoothing_param, kernel, neighbour_search: bool = False):
    L = 1.0
    N = 100
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    V = dx
    h = smoothing_param * dx

    if neighbour_search:
        sph_sum = np.zeros(N)

        pairs = cKDTree(x.reshape(-1, 1)).query_pairs(r=2.*h, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]

        W = kernel(x[i] - x[j], h, dim=1)
        sph_sum += np.bincount(i, weights=V*W, minlength=N) \
                + np.bincount(j, weights=V*W, minlength=N)
        sph_sum += V * kernel(np.array([0.0]), h, dim=1)[0]
        
    else: 
        # distance between every lattice point pair
        r = x[:, None] - x[None, :]
        W = kernel(r, h, dim=1)
        sph_sum = V * np.sum(W, axis=1)
    return x, sph_sum


if __name__ == "__main__":
    # x, S = sph_approx(1)
    fig, ax = plt.subplots(2,2, figsize=(6,6), constrained_layout=True)
    nb_pts = 8
    cmap = plt.get_cmap('viridis', nb_pts + 1)
    kerns = ["wendland", "cubic spline"]
    for jj, kernel in enumerate([kernels.wendland, kernels.cubic_spline]):
        ax[0, jj].set_title(rf"$dx \cdot \sum_i W(r, h)$, {kerns[jj]}")
        ax[1, jj].set_title(r"error to unity")

        for ii, i in enumerate(np.linspace(1, 4, nb_pts)):
            x, S = sph_approx(i, kernel=kernel, neighbour_search=True)
            ax[0, jj].plot(x, S, 'o--', color=cmap(ii), label=f"{i:.2f}")
            ax[1, jj].semilogy(x, np.abs(1 - S), 'o--', color=cmap(ii), label=f"{i:.2f}")
    
    for i in range(2):
        ax[0, i].legend(title="smoothing * dx")
        ax[0, i].set(ylim=(.7, 1.035), xlim=(0, 1))

    save_to_logs(fig, "202611/partition_of_unity_test.pdf", dpi=100)