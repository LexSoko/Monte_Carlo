from timeit import timeit
from time import perf_counter
import modules.TSP as tsp
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt
import inspect
import numba as nb
from modules.stdout import Stdout_parse


@nb.njit(cache=True)
def benchmark_temperature_function(n, state, length):
    return np.float64(1000.0)

def print_exectimes(variable, times, function, time_meas = "s", variable_name = "n"):
    max_lenght = len(f"{max(variable)}: {times[-1]:.10f} {time_meas}\n")
    max_lenght1 = len(f"function: {function}\n")
    max_lenght = max(max_lenght,max_lenght1)
    delimiter = "#"*max_lenght
    print(f"\n{delimiter}\n")
    print(f"function: {function}\n")
    print(f"{delimiter}\n")
    print(f"{variable_name}"+" "*(max_lenght//2-len(variable_name)-1) + ": " + f"{time_meas} \n")
    print("-"*max_lenght + "\n")
    for n in range(len(times)):
        variable_str = f"{variable[n]}"
        print(f"{variable_str}" + " "*(max_lenght//2-len(variable_str)-1) + ": " + f"{times[n]} \n")
    print("-"*max_lenght + "\n")

def plot_performance(results_dict:dict, dst_path="", time = "s"):
    for key, value in results_dict.items():
        fig, ax = plt.subplots(1,1, figsize=(10,10))

        times = np.asarray(value["times"], dtype=np.float64)
        try:
            standard_deviation = np.asarray(
                value["standard_deviation"],
                dtype=np.float64
            )
        except:
            standard_deviation = np.zeros(times.shape)

        if time == "ms":
            times = times*10**3
            standard_deviation = standard_deviation*10**3

        fig.suptitle(f"function: {key}")
        parameter_values = {
            parameter_key: parameter_value
            for parameter_key, parameter_value in value.items()
            if parameter_key not in ("times", "standard_deviation")
        }

        if len(parameter_values) == 2:
            keys = list(parameter_values.keys())
            value_values = list(parameter_values.values())
            value_label = np.array(value_values[0])

            if len(value_label)> 10:
                down = len(value_label)//10
                value_label = value_label[::down]
                times = times[:,::down]
                standard_deviation = standard_deviation[:,::down]

            for i, v in enumerate(value_label):
                mean = times[:,i]
                deviation = standard_deviation[:,i]
                ax.plot(
                    value_values[1],
                    mean,
                    label = f"performance for {keys[0]} = {v}"
                )
                ax.fill_between(
                    value_values[1],
                    mean-deviation,
                    mean+deviation,
                    alpha=0.2
                )
            ax.set_ylabel(time)
            ax.set_xlabel(keys[1])
            ax.legend()
        else:
            keys = list(parameter_values.keys())
            value_values = list(parameter_values.values())
            ax.plot(value_values[0], times, label = "performance")
            ax.fill_between(
                value_values[0],
                times-standard_deviation,
                times+standard_deviation,
                alpha=0.2
            )
            ax.set_ylabel(time)
            ax.set_xlabel(keys[0])
            ax.legend()

        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(dst_path,f"{key}.pdf"))
        plt.close(fig)


def warmup(N,p):
    N_c_warmup = np.int32(N)
    pop_warmup = np.int32(p)

    tour_warmup = np.asarray(
        tsp.create_cities(int(N_c_warmup)),
        dtype=np.float64,
    )

    tour_ids_warm = np.ascontiguousarray(
        tsp.create_tour_ids(tour_warmup),
        dtype=np.int32,
    )

    D_warmup = np.ascontiguousarray(
        tsp.construct_D_matrix(
            tour_warmup,
            tsp.euclidian_dist,
        ),
        dtype=np.float64,
    )

    tour_id_m_warmup = np.ascontiguousarray(
        np.tile(
            tour_ids_warm,
            (int(pop_warmup), 1),
        ),
        dtype=np.int32,
    )

    cities_pos_pop_warmup = np.ascontiguousarray(
        np.repeat(
            tour_warmup[np.newaxis,:,:],
            int(pop_warmup),
            axis=0
        ),
        dtype=np.float64
    )

    length_warm = np.float64(5.0)

    lengths_warm = np.full(
        int(pop_warmup),
        length_warm,
        dtype=np.float64,
    )        
    warmup_values = {
    "seed": np.int64(30121999),

    "tour": tour_warmup,
    "distance_metric": tsp.euclidian_dist,
    "dtype": np.float64,

    "D": D_warmup,
    "tour_ids": tour_ids_warm,
    "tours_ids": tour_ids_warm,
    "city_ids": tour_ids_warm,

    "arrays": cities_pos_pop_warmup,
    "lenght_function": tsp.length_func,
    "ordered_coordinates": tour_warmup,

    "a": np.int32(1),
    "b": np.int32(N_c_warmup-2),

    "Tour_ID_Matrix": tour_id_m_warmup,
    "specific_population_members": np.array([-1], dtype=np.int64),

    "temperature": np.float64(1000.0),
    "temperature_function": benchmark_temperature_function,
    "N_cities": N_c_warmup,
    "Length": length_warm,
    "length": length_warm,
    "n_sweeps": np.int64(1),
    "const_temp": np.float64(1000.0),
    "warm_up": False,
    "detailed": True,

    "cities_pos_pop": cities_pos_pop_warmup,
    "prob_mut": np.float64(0.9),

    "old_generation": tour_id_m_warmup,
    "lenghts": lengths_warm,
    "population_size": pop_warmup,

    "survivors": np.ascontiguousarray(
        tour_id_m_warmup[:2],
        dtype=np.int32,
    ),
    "D_matrix": D_warmup,

    "Lengths_0": lengths_warm,
    "mutations_per_sweep": np.int64(N_c_warmup),
    }
    return warmup_values
    
def measure_performance():
    BENCHMARK_FUNCTIONS = [
        tsp.construct_D_matrix,
        tsp.construct_D_matrix_dtype,
        tsp.calculate_total_distance_with_D,
        tsp.create_tour_id_matrix,
        tsp.calculate_lenghts,
        tsp.length_func,
        tsp.reverse_subsequence,
        tsp.create_diversity_ids,
        tsp.annealing_step_Dmatrix,
        tsp.annealing_D,
        tsp.annealing_D_detailed,
        tsp.mutation_genetic,
        tsp.choose_survivors_ids,
        tsp.mate_ids,
        tsp.mixed_annealing_D,
        tsp.mixed_annealing_D_after,
        tsp.mixed_annealing_D_const_T,
    ]
    warmup_values = warmup(
         6,
         4
    )
    print(warmup_values)
    pbar = tqdm(BENCHMARK_FUNCTIONS,desc= "warming up")
    for func in pbar:
        parameter_names =  inspect.signature(
            func
        ).parameters
        func_name = func.__name__
        pbar.set_description(f"warming up {func_name}")
        function_arguments = {
            name: warmup_values[name] for name in parameter_names
            }
        _ = func(**function_arguments)

    print("\n After warm-up: \n")
    print(tsp.annealing_step_Dmatrix.signatures)

    time_now = datetime.now().strftime('%m_%d_%H_%M')
    
    repeat = 10
    N_cities = np.arange(
        10,
        200,
        1,
        dtype=np.int32
        )
    population_sizes = np.arange(
        2,
        18,
        2,
        dtype=np.int32
        )
    print(population_sizes)
    nsweeps = 1
    performance_benchmark_path = os.path.join("..","results","performance_benchmark",f"run_{time_now}_{N_cities[-1]}_{population_sizes[-1]}")
        
    temp = np.float64(10000.0)
    tours = [tsp.create_cities(n) for n in N_cities]
    
    
    tours_ids =  [tsp.create_tour_ids(tour) for tour in tours]

    tour_ids_matrices = []
    

    results = {}
    
    for tour_id in tours_ids:
        tour_ids_matrices_p = []
        for p in population_sizes:
            tour_ids_matrices_p.append(tsp.create_tour_id_matrix(tour_id,p))
        tour_ids_matrices.append(tour_ids_matrices_p)

    
    D_matrices = []        
   
    results["construct_D_matrix"] = {
         "N_cities": N_cities,
         "times": [],
         "standard_deviation": []
    }

    for tour in tqdm(tours,desc="construct_D_matrix"):
        time_samples = []
        for r in range(repeat):
            start = perf_counter()
            D = tsp.construct_D_matrix(tour,tsp.euclidian_dist)
            end = perf_counter()
            time_samples.append(end-start)
        time_avg = np.mean(time_samples)
        time_std = np.std(time_samples)
        D_matrices.append(D)
        results["construct_D_matrix"]["times"].append(time_avg)
        results["construct_D_matrix"]["standard_deviation"].append(time_std)


    results["construct_D_matrix_dtype"] = {
         "N_cities": N_cities,
         "times": [],
         "standard_deviation": []
    }

    for tour in tqdm(tours,desc="construct_D_matrix_dtype"):
        time_samples = []
        for r in range(repeat):
            start = perf_counter()
            _ = tsp.construct_D_matrix_dtype(
                tour,
                tsp.euclidian_dist,
                np.float64
            )
            end = perf_counter()
            time_samples.append(end-start)
        results["construct_D_matrix_dtype"]["times"].append(
            np.mean(time_samples)
        )
        results["construct_D_matrix_dtype"]["standard_deviation"].append(
            np.std(time_samples)
        )


    results["create_cities"] = {
        "N_cities": N_cities,
        "times": [],
        "standard_deviation": []
    }

    for n_cities in tqdm(N_cities, desc="create_cities"):
        time_samples = []
        for r in range(repeat):
            start = perf_counter()
            _ = tsp.create_cities(n_cities)
            end = perf_counter()
            time_samples.append(end-start)
        results["create_cities"]["times"].append(np.mean(time_samples))
        results["create_cities"]["standard_deviation"].append(
            np.std(time_samples)
        )


    results["create_tour_ids"] = {
        "N_cities": N_cities,
        "times": [],
        "standard_deviation": []
    }

    for tour in tqdm(tours, desc="create_tour_ids"):
        time_samples = []
        for r in range(repeat):
            start = perf_counter()
            _ = tsp.create_tour_ids(tour)
            end = perf_counter()
            time_samples.append(end-start)
        results["create_tour_ids"]["times"].append(np.mean(time_samples))
        results["create_tour_ids"]["standard_deviation"].append(
            np.std(time_samples)
        )


    results["calculate_lenght"] = {
        "N_cities": N_cities,
        "times": [],
        "standard_deviation": []
    }

    for tour in tqdm(tours, desc="calculate_lenght"):
        time_samples = []
        for r in range(repeat):
            start = perf_counter()
            _ = tsp.calculate_lenght(tour)
            end = perf_counter()
            time_samples.append(end-start)
        results["calculate_lenght"]["times"].append(np.mean(time_samples))
        results["calculate_lenght"]["standard_deviation"].append(
            np.std(time_samples)
        )


    results["length_func"] = {
        "N_cities": N_cities,
        "times": [],
        "standard_deviation": []
    }

    for tour in tqdm(tours, desc="length_func"):
        time_samples = []
        for r in range(repeat):
            start = perf_counter()
            _ = tsp.length_func(tour)
            end = perf_counter()
            time_samples.append(end-start)
        results["length_func"]["times"].append(np.mean(time_samples))
        results["length_func"]["standard_deviation"].append(
            np.std(time_samples)
        )


    results["make_loop"] = {
        "N_cities": N_cities,
        "times": [],
        "standard_deviation": []
    }

    for tour in tqdm(tours, desc="make_loop"):
        time_samples = []
        for r in range(repeat):
            start = perf_counter()
            _ = tsp.make_loop(tour)
            end = perf_counter()
            time_samples.append(end-start)
        results["make_loop"]["times"].append(np.mean(time_samples))
        results["make_loop"]["standard_deviation"].append(
            np.std(time_samples)
        )


    results["reverse_subsequence"] = {
        "N_cities": N_cities,
        "times": [],
        "standard_deviation": []
    }

    for i, tour_id in tqdm(
        enumerate(tours_ids),
        desc="reverse_subsequence"
    ):
        time_samples = []
        for r in range(repeat):
            tour_id_current = tour_id.copy()
            start = perf_counter()
            _ = tsp.reverse_subsequence(
                tour_id_current,
                np.int32(1),
                np.int32(N_cities[i]-2)
            )
            end = perf_counter()
            time_samples.append(end-start)
        results["reverse_subsequence"]["times"].append(
            np.mean(time_samples)
        )
        results["reverse_subsequence"]["standard_deviation"].append(
            np.std(time_samples)
        )

    

    Lenghts = np.empty(len(N_cities), dtype=np.float64)
    Lenghts_p = []
    results["calculate_total_distance_with_D"] = {
         "N_cities": N_cities,
         "times": [],
         "standard_deviation": []
    }
    for i, tour_id in tqdm(enumerate(tours_ids),desc="calculate_total_distance_with_D"):
            time_samples = []
            for r in range(repeat):
                start = perf_counter()
                L = tsp.calculate_total_distance_with_D(
                     D_matrices[i],
                     tour_id
                )
                end = perf_counter()
                time_samples.append(end-start)
            time_avg = np.mean(time_samples)
            time_std = np.std(time_samples)
            Lenghts[i] = L 
            Lenths_inner = []
            for p in population_sizes:
                Lenths_inner.append([L for pi in range(p)])
            Lenghts_p.append(Lenths_inner)
            results["calculate_total_distance_with_D"]["times"].append(time_avg)
            results["calculate_total_distance_with_D"]["standard_deviation"].append(
                time_std
            )


    results["create_tour_id_matrix"] = {
        "population": population_sizes,
        "N_cities": N_cities,
        "times": np.empty((len(N_cities),len(population_sizes))),
        "standard_deviation": np.empty((len(N_cities),len(population_sizes)))
    }

    for i, tour_id in tqdm(enumerate(tours_ids), desc="create_tour_id_matrix"):
        for j, pop_size in enumerate(population_sizes):
            time_samples = []
            for r in range(repeat):
                start = perf_counter()
                _ = tsp.create_tour_id_matrix(tour_id,pop_size)
                end = perf_counter()
                time_samples.append(end-start)
            results["create_tour_id_matrix"]["times"][i,j] = np.mean(time_samples)
            results["create_tour_id_matrix"]["standard_deviation"][i,j] = np.std(time_samples)


    results["calculate_lenghts"] = {
        "population": population_sizes,
        "N_cities": N_cities,
        "times": np.empty((len(N_cities),len(population_sizes))),
        "standard_deviation": np.empty((len(N_cities),len(population_sizes)))
    }

    for i, tour in tqdm(enumerate(tours), desc="calculate_lenghts"):
        for j, pop_size in enumerate(population_sizes):
            cities_pos_pop = np.repeat(
                tour[np.newaxis,:,:],
                int(pop_size),
                axis=0
            )
            time_samples = []
            for r in range(repeat):
                start = perf_counter()
                _ = tsp.calculate_lenghts(cities_pos_pop,tsp.length_func)
                end = perf_counter()
                time_samples.append(end-start)
            results["calculate_lenghts"]["times"][i,j] = np.mean(time_samples)
            results["calculate_lenghts"]["standard_deviation"][i,j] = np.std(time_samples)


    results["create_diversity_ids"] = {
        "population": population_sizes,
        "N_cities": N_cities,
        "times": np.empty((len(N_cities),len(population_sizes))),
        "standard_deviation": np.empty((len(N_cities),len(population_sizes)))
    }

    for i, tours_id_matrix in tqdm(
        enumerate(tour_ids_matrices),
        desc="create_diversity_ids"
    ):
        for j, pop_size in enumerate(population_sizes):
            tour_id_m_current = np.array(tours_id_matrix[j])
            time_samples = []
            for r in range(repeat):
                start = perf_counter()
                _ = tsp.create_diversity_ids(
                    tour_id_m_current,
                    np.array([-1],dtype=np.int64)
                )
                end = perf_counter()
                time_samples.append(end-start)
            results["create_diversity_ids"]["times"][i,j] = np.mean(time_samples)
            results["create_diversity_ids"]["standard_deviation"][i,j] = np.std(time_samples)


    results["mutation_genetic"] = {
        "population": population_sizes,
        "N_cities": N_cities,
        "times": np.empty((len(N_cities),len(population_sizes))),
        "standard_deviation": np.empty((len(N_cities),len(population_sizes)))
    }

    for i, tour in tqdm(enumerate(tours), desc="mutation_genetic"):
        for j, pop_size in enumerate(population_sizes):
            cities_pos_pop = np.repeat(
                tour[np.newaxis,:,:],
                int(pop_size),
                axis=0
            )
            time_samples = []
            for r in range(repeat):
                cities_pos_pop_current = cities_pos_pop.copy()
                start = perf_counter()
                _ = tsp.mutation_genetic(
                    cities_pos_pop_current,
                    np.float64(0.9)
                )
                end = perf_counter()
                time_samples.append(end-start)
            results["mutation_genetic"]["times"][i,j] = np.mean(time_samples)
            results["mutation_genetic"]["standard_deviation"][i,j] = np.std(time_samples)

    
        
    
    results["annealing_step_Dmatrix"] = {
         "N_cities": N_cities,
         "times": [],
         "standard_deviation": []
    }

    for i,tour_id in tqdm(enumerate(tours_ids), desc= "annealing_step_Dmatrix"):
        time_samples = []
        for r in range(repeat):
            start = perf_counter()
            _,_ = tsp.annealing_step_Dmatrix(
                tour_id,
                D_matrices[i],
                temp,
                N_cities[i],
                Lenghts[i]
                )
            end = perf_counter()
            time_samples.append(end-start)

        time_avg = np.mean(time_samples)
        time_std = np.std(time_samples)
        results["annealing_step_Dmatrix"]["times"].append(time_avg)
        results["annealing_step_Dmatrix"]["standard_deviation"].append(time_std)


    results["annealing_D"] = {
        "N_cities": N_cities,
        "times": [],
        "standard_deviation": []
    }

    for i, tour_id in tqdm(enumerate(tours_ids), desc="annealing_D"):
        time_samples = []
        for r in range(repeat):
            tour_id_current = tour_id.copy()
            start = perf_counter()
            _ = tsp.annealing_D(
                tour_id_current,
                D_matrices[i],
                benchmark_temperature_function,
                Lenghts[i],
                nsweeps
            )
            end = perf_counter()
            time_samples.append(end-start)
        results["annealing_D"]["times"].append(np.mean(time_samples))
        results["annealing_D"]["standard_deviation"].append(
            np.std(time_samples)
        )


    results["annealing_D_detailed"] = {
        "N_cities": N_cities,
        "times": [],
        "standard_deviation": []
    }

    for i, tour_id in tqdm(
        enumerate(tours_ids),
        desc="annealing_D_detailed"
    ):
        time_samples = []
        for r in range(repeat):
            tour_id_current = tour_id.copy()
            start = perf_counter()
            _ = tsp.annealing_D_detailed(
                tour_id_current,
                D_matrices[i],
                benchmark_temperature_function,
                Lenghts[i],
                nsweeps,
                temp,
                False,
                True
            )
            end = perf_counter()
            time_samples.append(end-start)
        results["annealing_D_detailed"]["times"].append(
            np.mean(time_samples)
        )
        results["annealing_D_detailed"]["standard_deviation"].append(
            np.std(time_samples)
        )


    results["choose_survivors_ids"] = {
         "population": population_sizes,
         "N_cities": N_cities,
         "times": np.empty((len(N_cities),len(population_sizes))),
         "standard_deviation": np.empty(
             (len(N_cities),len(population_sizes))
         )
    }
    #print(Lenghts_p)
    survivors_matrix = []
    for i, tours_id_matrix in tqdm(enumerate(tour_ids_matrices), desc = "choose_survivors_ids"):
        inner = []
        for j, pop_size in enumerate(population_sizes):
            tour_id_m_current = np.array(tours_id_matrix[j])
            lenght_m_current = np.array(Lenghts_p[i][j])
            time_samples = []
            for r in range(repeat):
                start = perf_counter()
                survivors,survivor_L = tsp.choose_survivors_ids(
                    tour_id_m_current,
                    lenght_m_current,
                    pop_size,
                    N_cities[i]
                )
                end = perf_counter()
                time_samples.append(end-start)
            time_avg = np.mean(time_samples)
            time_std = np.std(time_samples)
            inner.append(survivors)
            results["choose_survivors_ids"]["times"][i,j] = time_avg
            results["choose_survivors_ids"]["standard_deviation"][i,j] = time_std
        survivors_matrix.append(inner)

    
        

    results["mate_ids"] = {
             "population": population_sizes,
             "N_cities": N_cities,
             "times": np.empty((len(N_cities),len(population_sizes))),
             "standard_deviation": np.empty(
                 (len(N_cities),len(population_sizes))
             )
        }

    for i, tours_id_matrix in tqdm(enumerate(tour_ids_matrices), desc = "mate_ids"):
            
            for j, pop_size in enumerate(population_sizes):
                surv_current = np.array(survivors_matrix[i][j])
                time_samples = []
                for r in range(repeat):
                    start = perf_counter()
                    _,_ = tsp.mate_ids(
                         surv_current,
                         N_cities[i],
                         D_matrices[i]
                    )
                    end = perf_counter()
                    time_samples.append(end-start)
                time_avg = np.mean(time_samples)
                time_std = np.std(time_samples)
                results["mate_ids"]["times"][i,j] = time_avg
                results["mate_ids"]["standard_deviation"][i,j] = time_std


    results["mixed_annealing_D"] = {
        "population": population_sizes,
        "N_cities": N_cities,
        "times": np.empty((len(N_cities),len(population_sizes))),
        "standard_deviation": np.empty(
            (len(N_cities),len(population_sizes))
        )
    }

    for i, tours_id_matrix in tqdm(
        enumerate(tour_ids_matrices),
        desc="mixed_annealing_D"
    ):
        for j, pop_size in enumerate(population_sizes):
            tour_id_m_current = np.array(tours_id_matrix[j])
            lenght_m_current = np.array(
                Lenghts_p[i][j],
                dtype=np.float64
            )
            time_samples = []
            for r in range(repeat):
                tour_id_m_repeat = tour_id_m_current.copy()
                start = perf_counter()
                _ = tsp.mixed_annealing_D(
                    tour_id_m_repeat,
                    D_matrices[i],
                    pop_size,
                    benchmark_temperature_function,
                    lenght_m_current,
                    nsweeps
                )
                end = perf_counter()
                time_samples.append(end-start)
            results["mixed_annealing_D"]["times"][i,j] = np.mean(
                time_samples
            )
            results["mixed_annealing_D"]["standard_deviation"][i,j] = (
                np.std(time_samples)
            )


    results["mixed_annealing_D_after"] = {
        "population": population_sizes,
        "N_cities": N_cities,
        "times": np.empty((len(N_cities),len(population_sizes))),
        "standard_deviation": np.empty(
            (len(N_cities),len(population_sizes))
        )
    }

    for i, tours_id_matrix in tqdm(
        enumerate(tour_ids_matrices),
        desc="mixed_annealing_D_after"
    ):
        for j, pop_size in enumerate(population_sizes):
            tour_id_m_current = np.array(tours_id_matrix[j])
            lenght_m_current = np.array(
                Lenghts_p[i][j],
                dtype=np.float64
            )
            time_samples = []
            for r in range(repeat):
                tour_id_m_repeat = tour_id_m_current.copy()
                start = perf_counter()
                _ = tsp.mixed_annealing_D_after(
                    tour_id_m_repeat,
                    D_matrices[i],
                    pop_size,
                    benchmark_temperature_function,
                    lenght_m_current,
                    nsweeps
                )
                end = perf_counter()
                time_samples.append(end-start)
            results["mixed_annealing_D_after"]["times"][i,j] = np.mean(
                time_samples
            )
            results["mixed_annealing_D_after"]["standard_deviation"][i,j] = (
                np.std(time_samples)
            )


    results["mixed_annealing_D_const_T"] = {
        "population": population_sizes,
        "N_cities": N_cities,
        "times": np.empty((len(N_cities),len(population_sizes))),
        "standard_deviation": np.empty(
            (len(N_cities),len(population_sizes))
        )
    }

    for i, tours_id_matrix in tqdm(
        enumerate(tour_ids_matrices),
        desc="mixed_annealing_D_const_T"
    ):
        for j, pop_size in enumerate(population_sizes):
            tour_id_m_current = np.array(tours_id_matrix[j])
            lenght_m_current = np.array(
                Lenghts_p[i][j],
                dtype=np.float64
            )
            time_samples = []
            for r in range(repeat):
                tour_id_m_repeat = tour_id_m_current.copy()
                start = perf_counter()
                _ = tsp.mixed_annealing_D_const_T(
                    tour_id_m_repeat,
                    D_matrices[i],
                    pop_size,
                    benchmark_temperature_function,
                    lenght_m_current,
                    nsweeps,
                    N_cities[i]**2,
                    temp,
                    False,
                    False
                )
                end = perf_counter()
                time_samples.append(end-start)
            results["mixed_annealing_D_const_T"]["times"][i,j] = np.mean(
                time_samples
            )
            results["mixed_annealing_D_const_T"]["standard_deviation"][i,j] = (
                np.std(time_samples)
            )
    
        
    os.makedirs(performance_benchmark_path, exist_ok= True)
    #with Stdout_parse(folder=performance_benchmark_path, writing_type="a+"):
    #        print_exectimes(
    #             N_cities,
    #             results["construct_D_matrix"]["times"],
    #             "tsp.construct_D_matrix()",
    #             variable_name="N_cities"
    #             )
    #        print_exectimes(
    #                     N_cities,
    #                     results["calculate_total_distance_with_D"]["times"],
    #                     "calculate_total_distance_with_D",
    #                     variable_name="N_cities"
    #                     )
    #        print_exectimes(
    #                         N_cities,
    #                         results["annealing_step_Dmatrix"]["times"],
    #                         "annealing_step_Dmatrix",
    #                         variable_name="N_cities"
    #                         )
    #        for i, p in enumerate(population_sizes):
    #            print(f"pop = {p}")
    #            print_exectimes(
    #                results["choose_survivors_ids"]["N_cities"],
    #                results["choose_survivors_ids"]["times"][:,i],
    #                "choose_survivors_ids",
    #                variable_name="population"
    #            )
    #        for i, p in enumerate(population_sizes):
    #            print(f"pop = {p}")
    #            print_exectimes(
    #                results["mate_ids"]["N_cities"],
    #                results["mate_ids"]["times"][:,i],
    #                "mate_ids",
    #                variable_name="population"
    #            )
            
    results2 = {
            "mate_ids only highest N_cities": {
                "population": population_sizes,
                "times": np.array(results["mate_ids"]["times"])[-1,:],
                "standard_deviation": np.array(
                    results["mate_ids"]["standard_deviation"]
                )[-1,:]
            },
            "choose_survivors only highest N_cities": {
                "population": population_sizes,
                "times": np.array(
                    results["choose_survivors_ids"]["times"]
                )[-1,:],
                "standard_deviation": np.array(
                    results["choose_survivors_ids"]["standard_deviation"]
                )[-1,:]
            }
        }
    results3 = {
        "comparison normal annealing and pop 4,8,16": {
            "population": np.array([1,2,4,16]),
            "N_cities": N_cities,
            "times": np.column_stack(
                [
                    np.array(results["annealing_D"]["times"]),
                    np.array(results["mixed_annealing_D"]["times"])[:,0],
                    np.array(results["mixed_annealing_D"]["times"])[:,1],
                    np.array(results["mixed_annealing_D"]["times"])[:,-1]
                ]
            )
        }
    }
    results4 = {
            "comparison normal annealing and pop minus exectime of genetic": {
                "population": np.array([1,2,4,16]),
                "N_cities": N_cities,
                "times": np.column_stack(
                    [
                        np.array(results["annealing_D"]["times"]),
                        (np.array(results["mixed_annealing_D"]["times"])[:,0]  - (np.array(results["choose_survivors_ids"]["times"])[:,0] +np.array(results["mate_ids"]["times"])[:,0]))/2,
                        (np.array(results["mixed_annealing_D"]["times"])[:,1]  - (np.array(results["choose_survivors_ids"]["times"])[:,1] +np.array(results["mate_ids"]["times"])[:,1]))/4,
                        (np.array(results["mixed_annealing_D"]["times"])[:,-1] - (np.array(results["choose_survivors_ids"]["times"])[:,-1] +np.array(results["mate_ids"]["times"])[:,-1]))/16
                    ]
                )
            }
        }
    
    plot_performance(
        results,
        dst_path=performance_benchmark_path,
        time = "ms"
    )
    
    plot_performance(
        results2,
        dst_path=performance_benchmark_path,
        time = "ms"
        )
    plot_performance(
            results3,
            dst_path=performance_benchmark_path,
            time = "ms"
            )
    plot_performance(
            results4,
            dst_path=performance_benchmark_path,
            time = "ms"
            )
            
    
    return

if __name__ == "__main__":
    measure_performance()
