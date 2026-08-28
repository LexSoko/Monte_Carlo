import matplotlib.pyplot as plt 
from modules.TSP import create_cities, construct_D_matrix, calculate_total_distance_with_D, euclidian_dist, make_loop

from os.path import join
import numpy as np

results_path = join("..","results","solve_system_N11")
plots_path = join(results_path,"plots")
perm_path = join(results_path,"permutation.npy")

tour=create_cities(
    11, 
    low= -40,
    high=40, 
    seed= 30121999
    )


D = construct_D_matrix(
        tour,
        euclidian_dist
        )

perm = np.load(perm_path)
print(f"{len(perm)} permutations loaded")
lengths = np.empty(len(perm))
for n,ids in enumerate(perm):
    lengths[n] = calculate_total_distance_with_D(D,ids)

min_idx = np.argmin(lengths)
tour_ids_min = perm[min_idx]
min_tour = tour.copy()
min_tour = min_tour[tour_ids_min]
min_tour = make_loop(min_tour)
min_lenght = lengths[min_idx]



plt.xlim(-50,50)
plt.ylim(-50,50)
plt.plot(*min_tour, "b-", label=fr"min($L(\tau)$) = {min_lenght:.3f}",zorder=1)
plt.scatter(*tour.T , marker= "+", c = "r",s= 120.0, label = "Nodes", zorder=20)
plt.grid(alpha=0.25)
plt.legend()
plt.axis("equal")
plt.savefig(join("..","results","solve_system_N11","cities.pdf"))
plt.show()