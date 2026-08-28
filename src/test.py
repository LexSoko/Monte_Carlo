import matplotlib.pyplot as plt 
from modules.TSP import create_cities, construct_D_matrix_dtype, calculate_total_distance_with_D, euclidian_dist, make_loop, create_tour_ids
from solve_system import solve_system_from_permutations
from os.path import join
import numpy as np
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
results_path = join("..","results","solve_system_N11")
plots_path = join(results_path,"plots")
perm_path = join(results_path,"permutation.npy")

tour=create_cities(
    11, 
    low= -40,
    high=40, 
    seed= 30121999
    )
tour_ids =  create_tour_ids(tour)

D = construct_D_matrix_dtype(
        tour,
        euclidian_dist,
        dtype=np.float64
        )

perm = np.load(perm_path)
print(f"{len(perm)} permutations loaded")



lengths = np.empty(len(perm))
for n,ids in enumerate(perm):
    lengths[n] = calculate_total_distance_with_D(D,ids)

min_idx = np.argmin(lengths)
max_idx = np.argmax(lengths)

tour_ids_min = perm[min_idx]
min_tour = tour.copy()
min_tour = min_tour[tour_ids_min]
min_tour = make_loop(min_tour)
min_length = lengths[min_idx]
max_length = lengths[max_idx]
mean_l = np.mean(lengths)
#data1 = solve_system_from_permutations(perm_path,tour_ids,D,(0.5,10000,1),results_path,add_info="gen_preciser2",dtype=np.float64)
print(min_length,max_length)
lenghs = np.linspace(min_length,max_length,40)
data = np.loadtxt(join(results_path,"solved_system_data_0.5_10000_1_gen_preciser2.csv"), delimiter=";", skiprows=1, dtype=np.float64).T
T = data[0]
Z = data[1]
#plt.figure(figsize=(10,10))
#plt.yscale('log')
#plt.xscale('log')
#plt.ylim(1e-50,1.2)
#probs = []
#for l in lenghs:
#    prob = np.exp(-l/T)/Z
#    probs.append(prob)
#    plt.plot(T,prob, label = f"{l:.2f}")
##plt.legend()
#plt.show()



def plot_length_probabilities(
        T,
        lengths,
        log_Z,
        cmap_name="viridis",
        xscale="log",
        yscale="log",
        linewidth=1.0,
        alpha=0.8,
        ax=None,
):
    """
    Plot p(tau | T) = exp(-L/T) / Z(T) for multiple tour lengths.

    Parameters
    ----------
    T : array, shape (n_temperatures,)
        Temperatures.
    lengths : array, shape (n_lengths,)
        Lengths for which probability curves are plotted.
    log_Z : array, shape (n_temperatures,)
        Natural logarithm of the partition function.
    cmap_name : str
        Matplotlib colormap, e.g. 'viridis', 'plasma', 'turbo'.
    """
    T = np.asarray(T, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    log_Z = np.asarray(log_Z, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure

    cmap = plt.colormaps[cmap_name]

    norm = mpl.colors.Normalize(
        vmin=np.min(lengths),
        vmax=np.max(lengths),
    )

    for length in lengths:
        # log(p) is numerically safer than exp(-L/T)/Z.
        log_probability = -length / T - log_Z
        probability = np.exp(log_probability)

        ax.plot(
            T,
            probability,
            color=cmap(norm(length)),
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_xlabel(r"Temperature $T$")
    ax.set_ylabel(r"$p_L(\tau\mid T)$")
    ax.grid(alpha=0.25)
    ax.set_ylim((1e-20, 1.2))
    color_mapper = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    color_mapper.set_array([])

    colorbar = fig.colorbar(
        color_mapper,
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label(r" $L(\tau)$")

    fig.tight_layout()

    return fig, ax

lengths_high_resolution1 = np.linspace(
    mean_l,
    max_length,
    200,
)
lengths_high_resolution2 = np.linspace(
    min_length,
    max_length,
    1000,
)
lengths_high_resolution = np.concatenate([lengths_high_resolution2,lengths_high_resolution1])
log_Z = np.log(Z)
fig, ax = plot_length_probabilities(
    T,
    lengths_high_resolution,
    log_Z,
    cmap_name="turbo",
)
ax.hlines([1/len(perm)],xmin=min(T),xmax=max(T), colors="k", linestyles="--", label=r"$\frac{1}{N_{\tau}}$")
ax.legend(loc='lower right')
fig.savefig(
    join(results_path, "probabilities_colormap2.pdf"),
    bbox_inches="tight",
)

plt.show()