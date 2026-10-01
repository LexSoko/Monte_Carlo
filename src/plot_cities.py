import matplotlib.pyplot as plt 
from modules.TSP import create_cities, construct_D_matrix, calculate_total_distance_with_D, euclidian_dist, make_loop, create_tour_ids, unique_tsp_permutations
import argparse
from os.path import join
import numpy as np
import os

results_path = join("..","results","solve_system_N11")
plots_path = join(results_path,"plots")
perm_path = join(results_path,"permutation.npy")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results"
    )
    parser.add_argument(
        "--perm-path"
    )
    parser.add_argument(
        "--N-cities"
    )
    parser.add_argument(
        "--low"
    )
    parser.add_argument(
        "--high"
    )
    parser.add_argument(
        "--seed"
    )
    return parser.parse_args()

def simple_random_city_plot(
        results_path,
        permutation_path="results/permutation.npy",
        N_cities=11,
        low=-40,
        high=40,
        seed =30121999 
):
    tour=create_cities(
        N_cities, 
        low= low,
        high=high, 
        seed= seed
        )


    D = construct_D_matrix(
            tour,
            euclidian_dist
            )
    if os.path.exists(permutation_path):
        perm = np.load(perm_path)
    else:
        perm = unique_tsp_permutations()
    print(f"{len(perm)} permutations loaded")
    lengths = np.empty(len(perm))
   
    for n,ids in enumerate(perm):
        lengths[n] = calculate_total_distance_with_D(D,tour)

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
    plt.savefig(join(results_path,"plots","cities.pdf"))
    plt.show()
def main():
    args = parse_args()
    simple_random_city_plot(
        args.results,
        permutation_path=args.perm_path,
        N_cities=args.N_cities,
        low=args.low,
        high= args.high,
        seed=args.seed
    )
if __name__ == "__main__":
    main()