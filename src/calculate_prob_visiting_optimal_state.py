import matplotlib.pyplot as plt 
from modules.TSP import create_cities, construct_D_matrix_dtype, calculate_total_distance_with_D, euclidian_dist, make_loop, create_tour_ids
from solve_system import solve_system_from_permutations
from os.path import join
import numpy as np

results_path = join("..","results","solve_system_N11")
plots_path = join(results_path,"plots")
perm_path = join(results_path,"permutation.npy")
def main():
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

    data = np.loadtxt(join(results_path,"solved_system_data_0.5_10000_1_gen_preciser2.csv"), delimiter=";", skiprows=1, dtype=np.float64).T
    T = data[0]
    Z = data[1]

    prob_0 = np.exp(-min_length/T)
    prob = prob_0/Z

    prob_max_0 = np.exp(-max_length/T)
    prob_max = prob_max_0/Z

    p_mean = np.exp(-mean_l/T)/Z

    fig, ax = plt.subplots(2, figsize=(8,8))
    line1 = ax[0].plot(T,prob,label=r"$p_L(\tau_{\mathrm{min}} | T)$")

    ax[0].grid(alpha=0.25)
    ax[0].set_xlabel("$T$")

    ax[0].tick_params(axis="y", labelcolor="b")
    ax_twin = ax[0].twinx()
    line2 = ax_twin.plot(T,prob_max,"g-", label=r"$ p_L(\tau_{\mathrm{max}} | T)$")


    ax_twin.set_xlabel("$T$")
    ax_twin.tick_params(axis="y", labelcolor="g")


    ax_twin_2 = ax[0].twinx()
    line3 = ax_twin_2.plot(T,p_mean,"r-",label= r"$p_L(\tau_{\mathrm{mean}} | T)$")

    ax_twin_2.set_xlabel("$T$")
    ax_twin_2.tick_params(axis="y", labelcolor="r")
    ax_twin_2.spines['right'].set_position(('axes', 1.07))
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax[0].legend(lines, labels, loc='center right')
    c,b,patches = ax[1].hist(lengths,bins = 1000, density=True, alpha = 0.5, label= r"$L$ distribution",color="cornflowerblue")

    ax[1].vlines([min_length],color="b",linestyles="--",ymin = 0, ymax = max(c), label= fr"min($L(\tau)$) = {min_length:.3f}")
    ax[1].vlines([max_length],color="g",linestyles="--",ymin = 0, ymax = max(c), label= fr"max($L(\tau)$) = {max_length:.3f}")
    ax[1].vlines([np.mean(lengths)],color="r",linestyles="--",ymin = 0, ymax = max(c), label= fr"$\langle L(\tau) \rangle$ = {np.mean(lengths):.3f}")
    ax[1].legend()
    ax[1].set_xlabel("$L$")
    fig.tight_layout()
    fig.savefig(join(results_path,"prob_visit_optimal_1000.pdf"))
    plt.show()

if __name__ == "__main__":
    main()