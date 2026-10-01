import modules.TSP as tsp
import modules.plotForReport as pr
import numpy as np 
import matplotlib.pyplot as plt
import numba as nb
import os 
import math
from tqdm import tqdm
from time import perf_counter

from solve_system import (
    calculate_total_distance_with_D,
    temp_func_const,
    save_to_file_system_solved,
)
@nb.njit
def temp_func_warmup(n,ids,lenght):
    return 3000 - 5*(n+1)  + ids 
def plot_quantities2(
        fig,
        ax,
        data,
        temp,
        colums,
        path,
        t_lims,
        tsp=False,
        save=True,
        add_info="plot"
):
    data = np.asarray(data)

    # Allow both full data and data without the temperature row.
    if data.shape[0] == 6:
        data = data[1:]

    y_labels = colums
    fig.set_size_inches(10, 12)

    if tsp:
        # Number of population curves already present.
        population_index = len(ax[1].lines)
        color = plt.get_cmap("Paired")(population_index % 10)
        label = f"population id {population_index}"
        linestyle = "-"
        linewidth = 0.8
        alpha = 0.6
    else:
        color = "red"
        label = "exact"
        linestyle = "--"
        linewidth = 2.0
        alpha = 1.0

    for d in range(len(data)):
        if np.isnan(data[d, 0]):
            continue

        ax[d].plot(
            temp,
            data[d],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label
        )

        ax[d].set_ylabel(y_labels[d])
        ax[d].set_xlim(temp[0], temp[-1])
        ax[d].grid(alpha=0.25)

    # The heat capacity covers a very large range.
    ax[3].set_yscale(
        "symlog",
        linthresh=1.0
    )

    # Show the x label only on the bottom subplot.
    for current_ax in ax[:-1]:
        current_ax.set_xlabel("")

    ax[-1].set_xlabel(r"$T$")

    if save:
        # Remove separate subplot legends.
        for current_ax in ax:
            legend = current_ax.get_legend()

            if legend is not None:
                legend.remove()

        # Remove an older figure legend if the function is called again.
        for legend in list(fig.legends):
            legend.remove()

        # All population colors are the same across the subplots,
        # so the handles from the <L> subplot are sufficient.
        handles, labels = ax[1].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(5, len(labels)),
            fontsize=9,
            frameon=False
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.93],
            h_pad=1.2
        )

        fig.savefig(
            os.path.join(
                path,
                (
                    f"solved_system_"
                    f"{t_lims[0]}_{t_lims[1]}_{t_lims[2]}_"
                    f"{add_info}.pdf"
                )
            ),
            bbox_inches="tight"
        )

    return fig, ax




def plot_quantity(
        x,
        y,
        dy = [None],
        labels = ["T","func"],
        path = "",
        save = True,
        add_info = "",
        fig = None,
        ax = None,
        twinx = False
        ):
    if fig == None and ax == None:
        fig, ax = plt.subplots(1,1, figsize=(6,10))

    if twinx:
        axtwin = ax.twinx()
        ax = axtwin
    ax.plot(x,y, label = labels[1])
    ax.set_xlabel(labels[0])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    #if dy[0] != None:
    #    ax.fill_between(x, y -dy , y + dy, label = r"$1\sigma$", alpha = 0.3)
    fig.tight_layout()
    if save:
        
        legend = ax.get_legend()

        if legend is not None:
            legend.remove()
        
        # Remove an older figure legend if the function is called again.
        for legend in list(fig.legends):
            legend.remove()

        # All population colors are the same across the subplots,
        # so the handles from the <L> subplot are sufficient.
        handles, labels = ax.get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(5, len(labels)),
            fontsize=9,
            frameon=False
        )
        fig.tight_layout(
                    rect=[0, 0, 1, 0.85],
                    h_pad=1.5
                )
        fig.savefig(os.path.join(path,f"solved_system_{add_info}.pdf"))

    return fig, ax



def solve_system_with_mixed_TSP(
        tours_ids,
        D,
        temperatures,
        nsweeps,
        population_size,
        mutations_per_sweep,
        mutations_interval=1,
        warmup=1500,
        detailed=False,
        seed = 1234
):
    nsweeps = nsweeps - warmup
    N_cities = len(tours_ids)

    Tour_id_matrix = np.empty(
        (population_size, N_cities),
        dtype=np.int32
    )

    for p in range(population_size):
        Tour_id_matrix[p] = tours_ids

    Tour_id_matrix = tsp.create_diversity_ids(Tour_id_matrix, specific_population_members=[-1])

    #for p in range(1, population_size):
    #    np.random.shuffle(Tour_id_matrix[p])

    Lengths_0 = np.empty(population_size, dtype=np.float64)
    for p in range(population_size):
        Lengths_0[p] = calculate_total_distance_with_D(
            D,
            Tour_id_matrix[p]
        )

    Lengths_T = np.empty(
        (
            population_size,
            len(temperatures),
            nsweeps + 1
        ),
        dtype=np.float32
    )
    acceptance_rate_T = np.empty(
        (len(temperatures), nsweeps,population_size),
        dtype=np.float64
    )

    _,_,_,_ = tsp.mixed_annealing_D_const_T_2(
        Tour_id_matrix[:2],
        D,
        2,
        temp_func_const,
        Lengths_0[:2],
        0,
        1,
        1,
        temperatures[0],
        True,
        False,
        seed
    )
    print(f"total expected iterations = {len(temperatures)}")

    for n, t in tqdm(
            enumerate(temperatures),
            desc="solving with mixed TSP"
    ):
        (
            Tour_id_matrix,
            Lengths_warmup,
            temperatures_warmup,
            acceptance_warmup,
        ) = tsp.mixed_annealing_D_const_T_2(
            Tour_id_matrix,
            D,
            population_size,
            temp_func_warmup,
            Lengths_0,
            warmup,
            mutations_per_sweep,
            mutations_interval,
            const_temp=t,
            warm_up=True,
            detailed=detailed,
            seed=seed
        )

        Lengths_0 = Lengths_warmup[-1].copy()

        (
            Tour_id_matrix,
            Lengths,
            temperatures_mixed,
            acceptance_rate,
        ) = tsp.mixed_annealing_D_const_T_2(
            Tour_id_matrix,
            D,
            population_size,
            temp_func_const,
            Lengths_0,
            nsweeps,
            mutations_per_sweep,
            mutations_interval,
            const_temp=t,
            warm_up=False,
            detailed=detailed,
            seed=seed
        )

        Lengths_0 = Lengths[-1].copy()
        Lengths_T[:, n] = Lengths.T
        acceptance_rate_T[n] = acceptance_rate

    data = [
        temperatures,
        Lengths_T,
        acceptance_rate_T,
    ]
    return data




def main():
    results_path = os.path.join("..", "results", "solve_system_N11")
    plots_path = os.path.join(results_path, "plots")

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    t_lims = (
        0.5,
        50,
        0.01
    )

    solved_system_path = os.path.join(
        results_path,
        (
            f"solved_system_data_"
            f"{t_lims[0]}_{t_lims[1]}_{t_lims[2]}_gen.csv"
        )
    )

    data_solved = np.loadtxt(
        solved_system_path,
        delimiter=";",
        skiprows=1
    ).T

    temperatures = data_solved[0]

    tour = tsp.create_cities(
        11,
        low=-40,
        high=40,
        seed=30121999
    )
    tour_ids = tsp.create_tour_ids(tour)
    D = tsp.construct_D_matrix(
        tour,
        tsp.euclidian_dist
    )

    population_size = 16
    mutations_attempts_per_sweep = len(tour_ids)**2
    mutations_interval = 5
    nsweeps = 3000
    warmup = 600
    results_path = os.path.join(results_path,f"{nsweeps}_{warmup}_pop{population_size}_mut{mutations_attempts_per_sweep}_{mutations_interval}")
    plots_path = os.path.join(results_path,"plots")
    print(f"population size: {population_size}")
    print(f"mutations per sweep: {mutations_attempts_per_sweep}")
    print(f"mutations interval: {mutations_interval}")
    print(f"nsweeps: {nsweeps}, warmup: {warmup}")
    temperatures_tsp = temperatures[::10]
    data_TSP = solve_system_with_mixed_TSP(
        tour_ids,
        D,
        temperatures_tsp,
        nsweeps=nsweeps,
        population_size=population_size,
        mutations_per_sweep=mutations_attempts_per_sweep,
        mutations_interval=mutations_interval,
        warmup=warmup,
        detailed=False
    )

    Lengths_T = data_TSP[1]
    acceptance_rate_T = data_TSP[2]

    colums = [
        r"$T$",
        r"$\langle L \rangle_T$",
        r"$\langle L^2 \rangle_T - \langle L \rangle_T^2$",
        r"$C(T)$",
        r"$\langle L^2 \rangle_T$",
    ]

    fig, ax = plt.subplots(
        len(data_solved[2:]),
        figsize=(6, 10)
    )

    os.makedirs(results_path,exist_ok=True)
    os.makedirs(plots_path,exist_ok=True)
    for p in range(population_size):
        L = Lengths_T[p, :, 1:]

        L_mean = np.mean(L, axis=1)
        L_sq = np.mean(L**2, axis=1)
        variance = L_sq - L_mean**2
        heat_capacity = variance / temperatures_tsp**2

        data_population = np.array([
            temperatures_tsp,
            np.full(temperatures_tsp.shape, np.nan),
            L_mean,
            variance,
            heat_capacity,
            L_sq,
        ])

        colums_population = [
            r"$T$",
            rf"$\langle L \rangle_T$, population {p}",
            rf"$\langle L^2 \rangle_T - \langle L \rangle_T^2$, population {p}",
            rf"$C(T)$, population {p}",
            rf"$\langle L^2 \rangle_T$, population {p}",
        ]
        save_to_file_system_solved(
            data_population.T,
            [
                "T",
                "Z(T)",
                "L_mean",
                "var",
                "C(T)",
                "L_sq",
            ],
            results_path,
            t_lims,
            add_info=f"mixed_TSP_population_{p}"
        )

        fig, ax = plot_quantities2(
            fig,
            ax,
            data_population[2:],
            temperatures_tsp,
            colums_population[1:],
            plots_path,
            t_lims,
            tsp=True,
            save=False
        )

    fig, ax = plot_quantities2(
        fig,
        ax,
        data_solved[2:],
        temperatures,
        colums[1:],
        plots_path,
        t_lims,
        save=True,
        add_info="mixed_TSP"
    )

    acceptance_mean = np.mean(
        acceptance_rate_T,
        axis=1
    )
    acceptance_variance = np.var(
        acceptance_rate_T,
        axis=1
    )
    for p in range(population_size):
        save_to_file_system_solved(
            np.column_stack((
                temperatures_tsp,
                acceptance_mean,
                acceptance_variance
            )),
            [
                "T",
                "accept_mean",
                "accept_var"
            ],
            results_path,
            t_lims,
            add_info=f"mixed_TSP_metadata_population_{p}"
        )

    fig_acceptance, ax_acceptance = plt.subplots(1, 1)
    for p in range(population_size):
        fig_acceptance, ax_acceptance = plot_quantity(
            temperatures_tsp,
            acceptance_mean.T[p],
            dy=acceptance_variance.T[p],
            labels=[
                r"$T$",
                fr"$\langle A \rangle_T$ id {p}",
            ],
            path=results_path,
            add_info="mixed_TSP_acceptance",
            save=True,
            fig=fig_acceptance,
            ax=ax_acceptance
        ) 

    plt.show()    





def plotting():
    population_size = 8
    population_size = 8
    mutations_attempts_per_sweep = 121
    nsweeps = 4000
    warmup = 500
    results_path = os.path.join("..", "results", "solve_system_N11")
    plots_path = os.path.join(results_path, "plots")
    results_path = os.path.join(results_path,f"{nsweeps}_{warmup}_pop{population_size}_mut{mutations_attempts_per_sweep}")
    plots_path = os.path.join(results_path,"plots")
    t_lims = (
                0.5,
                50,
                0.01
            )
    solved_system_path = os.path.join(
            results_path,
            (
                f"solved_system_data_"
                f"{t_lims[0]}_{t_lims[1]}_{t_lims[2]}_gen.csv"
            )
        )
    
    data_solved = np.loadtxt(
        solved_system_path,
        delimiter=";",
        skiprows=1
    ).T
    
    temperatures = data_solved[0]
    colums = [
            r"$T$",
            r"$\langle L \rangle_T$",
            r"$\langle L^2 \rangle_T - \langle L \rangle_T^2$",
            r"$C(T)$",
            r"$\langle L^2 \rangle_T$",
        ]
    fig, ax = plt.subplots(
        len(data_solved[2:]),
        figsize=(6, 10),
        sharex=True
        )
    for p in range(population_size):
       
        data_population =  np.loadtxt(os.path.join(results_path,f"solved_system_data_0.5_50_0.01_mixed_TSP_population_{p}.csv"), skiprows=1,delimiter=";").T
        colums_population = [
            r"$T$",
            rf"$\langle L \rangle_T$, population {p}",
            rf"$\langle L^2 \rangle_T - \langle L \rangle_T^2$, population {p}",
            rf"$C(T)$, population {p}",
            rf"$\langle L^2 \rangle_T$, population {p}",
        ]

       

        fig, ax = plot_quantities2(
            fig,
            ax,
            data_population[2:],
            temperatures,
            colums_population[1:],
            plots_path,
            t_lims,
            tsp=True,
            save=False,
            
        )
    
    fig, ax = plot_quantities2(
        fig,
        ax,
        data_solved[2:],
        temperatures,
        colums[1:],
        plots_path,
        t_lims,
        save=True,
        add_info="mixed_TSP",
        
    )

  
   

   
    fig_acceptance, ax_acceptance = plt.subplots(1, 1)
    for p in range(population_size):
        meta_data= np.loadtxt(os.path.join(results_path,f"solved_system_data_0.5_50_0.01_mixed_TSP_metadata_population_{p}.csv"), skiprows=1,delimiter=";").T
        temperatures = meta_data[0]
        acceptance_mean = meta_data[1]
        acceptance_variance = meta_data[2]
        fig_acceptance, ax_acceptance = plot_quantity(
            temperatures,
            acceptance_mean.T[p],
            dy=acceptance_variance.T[p],
            labels=[
                r"$T$",
                fr"$\langle A \rangle_T$ id {p}",
            ],
            path=results_path,
            add_info="mixed_TSP_acceptance",
            save=True,
            fig=fig_acceptance,
            ax=ax_acceptance
        ) 

    plt.show()


if __name__ == "__main__":
    #plotting()
    main()