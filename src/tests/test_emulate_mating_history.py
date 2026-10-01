import matplotlib.pyplot as plt 
from modules.TSP import create_cities, construct_D_matrix_dtype, calculate_total_distance_with_D, euclidian_dist, make_loop, create_tour_ids, create_diversity_ids ,subseq_dist_2
from solve_system import solve_system_from_permutations
from os.path import join
import numpy as np
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numba as nb
from modules.plotForReport import plot_best_population_index_history

#for k in range(n_sweeps):    
#    dead_idx[k] = np.random.default_rng().choice(population, size= population//2, replace=False)
#
#mate_idx = np.empty((n_sweeps,population//2,3),dtype=np.int32)
#for i in range(n_sweeps):
#    for j in range(population//2):
#        a = np.random.randint(low=0, high=population -1)
#        b = np.random.randint(low=a, high=population)
#        c = np.random
#        mate_idx[i,j] = np.random.randint(1, 100, size = 3)

@nb.njit(cache=True)
def emulate_mating_history(n_sweeps, population_size):

    if population_size % 4 != 0:
        raise ValueError("population_size must be divisible by 4")

    n_survivors = population_size // 2

    dead_ids_history = np.empty(
        (n_sweeps, n_survivors),
        dtype=np.int32
    )

    mating_ids_history = np.empty(
        (n_sweeps, n_survivors, 3),
        dtype=np.int32
    )

    for sweep in range(n_sweeps):

        # Random battles
        population_indices = np.arange(population_size, dtype=np.int32)
        np.random.shuffle(population_indices)

        surviving_ids = np.empty(n_survivors, dtype=np.int32)
        dead_ids = np.empty(n_survivors, dtype=np.int32)

        for i in range(n_survivors):

            id_1 = population_indices[2 * i]
            id_2 = population_indices[2 * i + 1]

            # Randomly select the winner
            if np.random.randint(0, 2) == 0:
                surviving_ids[i] = id_1
                dead_ids[i] = id_2
            else:
                surviving_ids[i] = id_2
                dead_ids[i] = id_1

        dead_ids_history[sweep] = dead_ids

        # Randomly pair the survivors
        indices = np.arange(n_survivors, dtype=np.int32)
        np.random.shuffle(indices)

        for i in range(0, n_survivors, 2):

            parent_1 = surviving_ids[indices[i]]
            parent_2 = surviving_ids[indices[i + 1]]

            mating_ids_history[sweep, i, 0] = parent_1
            mating_ids_history[sweep, i, 1] = parent_2
            mating_ids_history[sweep, i, 2] = dead_ids[i]

            mating_ids_history[sweep, i + 1, 0] = parent_2
            mating_ids_history[sweep, i + 1, 1] = parent_1
            mating_ids_history[sweep, i + 1, 2] = dead_ids[i + 1]

    return mating_ids_history, dead_ids_history

n_sweeps = 10
population = 16


mating_ids, dead_ids = emulate_mating_history(n_sweeps,population)

best_ids = np.empty(n_sweeps)
mates_N = mating_ids.shape[1]
for i,m in enumerate(mating_ids):
    best_ids[i] = m[np.random.randint(mates_N), 0]

plot_best_population_index_history(
    best_ids,
    dead_idx = dead_ids,
    mate_idx=mating_ids,
    population= population
)
plt.show()