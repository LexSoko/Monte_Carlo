from timeit import timeit
from time import perf_counter
import modules.TSP as tsp
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt
import inspect
from modules.stdout import Stdout_parse

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
            
        times = value.pop("times")
        if time == "ms":
            times = np.array(times)*10**3

        
        fig.suptitle(f"function: {key}")
        if len(value) == 2:
            keys = list(value.keys())
            value_values = list(value.values())
            value_label = np.array(value_values[0])
            
            if len(value_label)> 10:
                down = len(value_label)//10
                value_label = value_label[::down]
                times = times[:,::down]

            for i, v in enumerate(value_label):
                ax.plot(value_values[1],times[:,i], label = f"performance for {keys[0]} = {v}")
            ax.set_ylabel(time)
            ax.set_xlabel(keys[1])
            ax.legend()
        else:
            keys, value_values = list(value.keys()), list(value.values())
            
            ax.plot(value_values[0], times, label = "performance")
            ax.set_ylabel(time)
            ax.set_xlabel(keys[0])
            ax.legend()
        fig.savefig(os.path.join(dst_path,f"{key}.pdf"))
def warmup(N,p,):
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

    length_warm = np.float64(5.0)

    lengths_warm = np.full(
        int(pop_warmup),
        length_warm,
        dtype=np.float64,
    )        
    warmup_values = {
    "tour": tour_warmup,
    "distance_metric": tsp.euclidian_dist,

    "D": D_warmup,
    "tour_ids": tour_ids_warm,
    "city_ids": tour_ids_warm,

    "temperature": np.float64(1000.0),
    "N_cities": N_c_warmup,
    "Length": length_warm,
    "length": length_warm,

    "old_generation": tour_id_m_warmup,
    "lenghts": lengths_warm,
    "population_size": pop_warmup,

    "survivors": np.ascontiguousarray(
        tour_id_m_warmup[:2],
        dtype=np.int32,
    ),
    "D_matrix": D_warmup,
    }
    return warmup_values
    
def measure_performance():
    BENCHMARK_FUNCTIONS = [
    tsp.construct_D_matrix,
    tsp.calculate_total_distance_with_D,
    tsp.annealing_step_Dmatrix,
    tsp.choose_survivors_ids,
    tsp.mate_ids,
        ]
    warmup_values = warmup(
         6,
         4
    )
    

    for func in tqdm(BENCHMARK_FUNCTIONS,desc= "warming up"):
        parameter_names =  inspect.signature(
            func
        ).parameters
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
        500,
        1,
        dtype=np.int32
        )
    population_sizes = np.arange(
        2,
        80,
        2,
        dtype=np.int32
        )
    
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
         "times": []
    }
    for tour in tqdm(tours,desc="construct_D_matrix"):
        time_exec = 0
        for r in range(repeat):
            start = perf_counter()
            D = tsp.construct_D_matrix(tour,tsp.euclidian_dist)
            end = perf_counter()
            time_exec += (end - start)
        time_avg = time_exec/repeat
        D_matrices.append(D)
        results["construct_D_matrix"]["times"].append(time_avg)

    

    Lenghts = np.empty(len(N_cities), dtype=np.float64)
    Lenghts_p = []
    results["calculate_total_distance_with_D"] = {
         "N_cities": N_cities,
         "times": []
    }
    for i, tour_id in tqdm(enumerate(tours_ids),desc="calculate_total_distance_with_D"):
            time_exec = 0
            for r in range(repeat):
                start = perf_counter()
                L = tsp.calculate_total_distance_with_D(
                     D_matrices[i],
                     tour_id
                )
                end = perf_counter()
                time_exec += (end - start)
            time_avg = time_exec/repeat
            Lenghts[i] = L 
            Lenths_inner = []
            for p in population_sizes:
                Lenths_inner.append([L for pi in range(p)])
            Lenghts_p.append(Lenths_inner)
            results["calculate_total_distance_with_D"]["times"].append(time_avg)

    
        
    
    results["annealing_step_Dmatrix"] = {
         "N_cities": N_cities,
         "times": []
    }

    for i,tour_id in tqdm(enumerate(tours_ids), desc= "annealing_step_Dmatrix"):
        time_exec = 0
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
            time_exec += (end - start)
        if i == 0:
            print("\n After first benchamrk: \n")
            print(tsp.annealing_step_Dmatrix.signatures)
        time_avg = time_exec/repeat
        results["annealing_step_Dmatrix"]["times"].append(time_avg)


    results["choose_survivors_ids"] = {
         "population": population_sizes,
         "N_cities": N_cities,
         "times": np.empty((len(N_cities),len(population_sizes)))
    }
    #print(Lenghts_p)
    survivors_matrix = []
    for i, tours_id_matrix in tqdm(enumerate(tour_ids_matrices), desc = "choose_survivors_ids"):
        inner = []
        for j, pop_size in enumerate(population_sizes):
            tour_id_m_current = np.array(tours_id_matrix[j])
            lenght_m_current = np.array(Lenghts_p[i][j])
            time_exec = 0
            for r in range(repeat):
                start = perf_counter()
                survivors,survivor_L = tsp.choose_survivors_ids(
                    tour_id_m_current,
                    lenght_m_current,
                    pop_size,
                    N_cities[i]
                )
                end = perf_counter()


                time_exec += (end - start)
            time_avg = time_exec/repeat
            inner.append(survivors)
            results["choose_survivors_ids"]["times"][i,j] = time_avg 
        survivors_matrix.append(inner)

    
        

    results["mate_ids"] = {
             "population": population_sizes,
             "N_cities": N_cities,
             "times": np.empty((len(N_cities),len(population_sizes)))
        }

    for i, tours_id_matrix in tqdm(enumerate(tour_ids_matrices), desc = "mate_ids"):
            
            for j, pop_size in enumerate(population_sizes):
                surv_current = np.array(survivors_matrix[i][j])
                time_exec = 0
                for r in range(repeat):
                    start = perf_counter()
                    _,_ = tsp.mate_ids(
                         surv_current,
                         N_cities[i],
                         D_matrices[i]
                    )
                    end = perf_counter()
                    time_exec += (end - start)
                time_avg = time_exec/repeat
                results["mate_ids"]["times"][i,j] = time_avg 
            survivors_matrix.append(inner)

    
        
    os.makedirs(performance_benchmark_path, exist_ok= True)
    with Stdout_parse(folder=performance_benchmark_path, writing_type="a+"):
            print_exectimes(
                 N_cities,
                 results["construct_D_matrix"]["times"],
                 "tsp.construct_D_matrix()",
                 variable_name="N_cities"
                 )
            print_exectimes(
                         N_cities,
                         results["calculate_total_distance_with_D"]["times"],
                         "calculate_total_distance_with_D",
                         variable_name="N_cities"
                         )
            print_exectimes(
                             N_cities,
                             results["annealing_step_Dmatrix"]["times"],
                             "annealing_step_Dmatrix",
                             variable_name="N_cities"
                             )
            for i, p in enumerate(population_sizes):
                print(f"pop = {p}")
                print_exectimes(
                    results["choose_survivors_ids"]["N_cities"],
                    results["choose_survivors_ids"]["times"][:,i],
                    "choose_survivors_ids",
                    variable_name="population"
                )
            for i, p in enumerate(population_sizes):
                print(f"pop = {p}")
                print_exectimes(
                    results["mate_ids"]["N_cities"],
                    results["mate_ids"]["times"][:,i],
                    "mate_ids",
                    variable_name="population"
                )
            
    results2 = {
            "mate_ids only highest N_cities": {
                "population": population_sizes,
                "times": np.array(results["mate_ids"]["times"])[-1,:]
            },
            "choose_survivors only highest N_cities": {
                "population": population_sizes,
                "times": np.array(results["choose_survivors_ids"]["times"])[-1,:]
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
    
    
    return

if __name__ == "__main__":
    measure_performance()