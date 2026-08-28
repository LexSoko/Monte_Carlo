import modules.TSP as tsp
import modules.plotForReport as pr
import numpy as np 
import matplotlib.pyplot as plt
import numba as nb
import os 
import math
from tqdm import tqdm
from time import perf_counter
tsp.seed_numba(30121999)

@nb.njit()
def permutations2(A, k):
    r = [[i for i in range(0)]]
    for i in range(k):
        t=[[i for i in range(0)] for i in range(0)]
        for next_num in A:
            for next_chain in r:           
                if (next_num in next_chain)==False:
                    t.append([next_num,] + next_chain)
        r=t
    return  r

@nb.njit(inline='always')
def calculate_total_distance_with_D(D, tour_ids):
    total_lenght = 0
    N_cities = len(tour_ids)
    for i in range(N_cities):
        total_lenght += D[tour_ids[i],tour_ids[(i+1)%N_cities]]
    return total_lenght

@nb.njit
def solve_system(perm,D_matrix,temperatures, dtype = np.float64):
    perm_array = perm
    N_configs = len(perm_array)
    all_lenghts = np.empty(N_configs, dtype=dtype)
    for i, perm in enumerate(perm_array):
        all_lenghts[i] = calculate_total_distance_with_D(D_matrix,perm)
    partition__func = np.empty(temperatures.shape, dtype=dtype)
    expectation_value = np.empty(temperatures.shape, dtype=dtype)
    expectation_value_L_sq = np.empty(temperatures.shape, dtype=dtype)
    for k, t in enumerate(temperatures):
        partion = 0 
        weight_sum_expect = 0
        weight_sum_expect_L_sq = 0
        for L in all_lenghts:
            exp = np.exp(-L/t)
            weight_sum_expect += L*exp
            weight_sum_expect_L_sq += L*L*exp
            partion += exp
        partition__func[k] = partion
        expectation_value[k] = weight_sum_expect/partion
        expectation_value_L_sq[k] = weight_sum_expect_L_sq/partion

    return partition__func, expectation_value, expectation_value_L_sq

@nb.njit
def temp_func_const(n, ids, Length):
    return 1
@nb.njit
def temp_func_warmup(n,ids,lenght):
    return 2000*(1/(n+1)*4) + ids 

def solve_system_with_TSP(tours_ids,D,temperatures, nsweeps, warmup =1500,detailed = False):
    nsweeps = nsweeps- warmup
    N_cities = len(tours_ids)**2
    N_cities = int(N_cities)
    Lengths_T = np.empty((temperatures.shape[0],nsweeps+1), dtype=np.float32)
    Length_samples_T =  np.empty((temperatures.shape[0],nsweeps,N_cities ), dtype=np.float64)
    mean_L_T = np.empty((temperatures.shape[0],nsweeps), dtype=np.float64) 
    mean_L_sq_T = np.empty((temperatures.shape[0],nsweeps), dtype=np.float64)
    variance_L_T = np.empty((temperatures.shape[0],nsweeps), dtype=np.float64)
    heat_capacity = np.empty((temperatures.shape[0],nsweeps), dtype=np.float64)
    acceptance_rate = np.empty((temperatures.shape[0],nsweeps), dtype=np.float64)
    lenght = calculate_total_distance_with_D(D,tours_ids)
    #warmup_sweep
    
    print(f"total expected iterations = {len(temperatures)}")
    
    for n,t in tqdm(enumerate(temperatures),desc="solving with TSP"):
        tours_ids_1,_, _,_, _,_, _, _, _ = tsp.annealing_D_detailed(
                tours_ids,
                D,
                temp_func_warmup,
                lenght,
                warmup,
                const_temp=t,
                warm_up=True,
                detailed=detailed
            )
        lenght = calculate_total_distance_with_D(D,tours_ids_1)
        #temp_func_const = make_temp_func_const(t)
        tours_ids_2 = tours_ids_1.copy()
        tours_ids_notused,Lengths, temperatures,lenght_samples, mean_L,mean_L_sq, variance, heat_C, accept = tsp.annealing_D_detailed(
            tours_ids_2,
            D,
            temp_func_const,
            lenght,
            nsweeps,
            const_temp=t,
            warm_up=False,
            detailed=detailed

        )
        
        Lengths_T[n] = Lengths
        Length_samples_T[n] = lenght_samples
        mean_L_T[n] = mean_L
        mean_L_sq_T[n] = mean_L_sq
        variance_L_T[n] = variance
        heat_capacity[n] = heat_C
        acceptance_rate[n] =accept

    data1 = [
        temperatures,
        Lengths_T,
        acceptance_rate,
        
    ]
    data2=    [mean_L_T,
        variance_L_T,
        heat_capacity,
        mean_L_sq_T,
        Length_samples_T
        ]
    return data1,data2


def save_to_file_system_solved(data,columns, results_path, temp_disc, add_info = "gen", dtype = np.float64):
    header = ""
    for c in columns:
        header += f"{c};"
    header = header[:-1]
    np.savetxt(
        os.path.join(results_path,f"solved_system_data_{temp_disc[0]}_{temp_disc[1]}_{temp_disc[2]}_{add_info}.csv"),
        data,
        delimiter=";",
        header=header
        )
def solve_system_from_permutations(
        perm_path,
        tour_ids, 
        D,
        t_lims,
        results_path, 
        temperature = None,
        seed=30121999,
        add_info= "gen",
        dtype = np.float64
        ):
    N_cities = len(tour_ids)
    expected_perm = math.factorial(N_cities-1)/2
    if os.path.exists(perm_path):
        perm = np.load(perm_path)
        print(f"loaded successfully")
        if len(perm) != expected_perm:
            print(f"length {len(perm)} dont match to expected {expected_perm}")
            #perm = permutations2(tour_ids,N_cities)
            perm = unique_tsp_permutations(N_cities)
        else:
            print("perms mathc")
    else:
        print(f"couldnt find file , reverting to manual calc")
        #perm = permutations2(tour_ids,N_cities)  
        perm = unique_tsp_permutations(N_cities)
        perm = np.array(perm)
        #print(perm[:5])
        np.save(perm_path,perm)
    if temperature is None:
        temperatures = np.arange(
        t_lims[0],
        t_lims[1],
        t_lims[2],
        dtype=dtype,
        )
    else:
        temperatures = np.atleast_1d(
        np.asarray(temperature, dtype=dtype)
        )
       
    print(f"solved system N={len(perm)}")
    colums = [
            "T",
            "Z(T)",
            "L_mean",
            "var",
            "C(T)",
            "L_sq",
        ]

    if not os.path.exists(os.path.join(results_path,f"solved_system_data_{t_lims[0]}_{t_lims[1]}_{t_lims[2]}_{add_info}.csv")):
        print("didnt find a file, calculating manually")
        
        
        partition_func, expectation_value, expectation_value_L_sq = solve_system(
            perm,
            D,
            temperatures=temperatures,
            dtype=dtype
            )

        variance =expectation_value_L_sq-expectation_value**2
        heat_cap = variance/temperatures**2

        data = np.column_stack((
        temperatures,
        partition_func,
        expectation_value,
        variance,
        heat_cap,
        expectation_value_L_sq,
        ))
        save_to_file_system_solved(data,colums,results_path, t_lims,dtype=dtype,add_info=add_info)
    else:
        print("loading from data")
        data = np.loadtxt(os.path.join(results_path,f"solved_system_data_{t_lims[0]}_{t_lims[1]}_{t_lims[2]}_{add_info}.csv"),delimiter=";",skiprows=1,dtype=dtype )
    print(data)
    data = data.T
    return data 

def plot_quantities(fig,ax,data,temp,colums, path, t_lims,tsp=False ,save = False):
    i=0
    if tsp == True:
        i=1
    for d in range(0,len(data)+i):
        if tsp == True:
            if d == 0:
                continue
        ax[d].plot(temp,data[d-i], label=colums[d+1])
        ax[d].legend()
        ax[d].set_xlabel(colums[0])
        ax[d].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(path,f"solved_system_{t_lims[0]}_{t_lims[1]}_{t_lims[2]}.pdf"))
    return fig,ax


def plot_quantities2(fig,ax,data,temp,colums, path, t_lims,tsp = False ,save = True, add_info = "plot", fmt = "b"):
   
    for d in range(0,len(data)):
        if np.isnan(data[d, 0]):
            continue
        ax[d].plot(temp,data[d],fmt, label=colums[d+1])
        if tsp == True and colums[d+1] == r"$\langle L \rangle_T$ (TSP)":
            ax[d].fill_between(temp, data[d]- np.sqrt(data[d+1]),data[d]+  np.sqrt(data[d+1]), label= r"$ 1\sigma$", alpha= 0.3)

        ax[d].legend()
        ax[d].set_xlabel(colums[0])
        ax[d].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(path,f"solved_system_{t_lims[0]}_{t_lims[1]}_{t_lims[2]}_{add_info}.pdf"))
    return fig,ax





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
        fig, ax = plt.subplots(1,1)

    if twinx:
        axtwin = ax.twinx()
        ax = axtwin
    ax.plot(x,y, label = labels[1])
    ax.set_xlabel(labels[0])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    if dy[0] != None:
        ax.fill_between(x, y -dy , y + dy, label = r"$1\sigma$", alpha = 0.3)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(path,f"solved_system_{add_info}.pdf"))

    return fig, ax






@nb.njit(cache=True)
def factorial_int(n):
    result = 1

    for i in range(2, n + 1):
        result *= i

    return result


@nb.njit(cache=True)
def next_permutation(values):
    """Change values to its next lexicographic permutation.

    Returns False when the current permutation is the last one.
    """
    n = len(values)

    i = n - 2

    while i >= 0 and values[i] >= values[i + 1]:
        i -= 1

    if i < 0:
        return False

    j = n - 1

    while values[j] <= values[i]:
        j -= 1

    temporary = values[i]
    values[i] = values[j]
    values[j] = temporary

    left = i + 1
    right = n - 1

    while left < right:
        temporary = values[left]
        values[left] = values[right]
        values[right] = temporary

        left += 1
        right -= 1

    return True


@nb.njit(cache=True)
def unique_tsp_permutations(N_cities):
    """Generate unique tours for a symmetric TSP.

    City 0 is fixed at the beginning to remove rotational duplicates.
    A tour and its reversed version are counted only once.
    """
    if N_cities < 1:
        raise ValueError("N_cities must be at least 1")

    if N_cities == 1:
        tours = np.empty((1, 1), dtype=np.int32)
        tours[0, 0] = 0
        return tours

    if N_cities == 2:
        tours = np.empty((1, 2), dtype=np.int32)
        tours[0, 0] = 0
        tours[0, 1] = 1
        return tours

    number_of_tours = factorial_int(N_cities - 1) // 2

    tours = np.empty(
        (number_of_tours, N_cities),
        dtype=np.int32
    )

    remaining_cities = np.arange(
        1,
        N_cities,
        dtype=np.int32
    )

    tour_index = 0
    permutations_remaining = True

    while permutations_remaining:
        # Between a tour and its reverse, exactly one satisfies this.
        if remaining_cities[0] < remaining_cities[-1]:
            tours[tour_index, 0] = 0

            for city_index in range(N_cities - 1):
                tours[tour_index, city_index + 1] = (
                    remaining_cities[city_index]
                )

            tour_index += 1

        permutations_remaining = next_permutation(
            remaining_cities
        )

    return tours













def main():

    results_path = os.path.join("..","results","solve_system_N11")
    plots_path = os.path.join(results_path,"plots")
    perm_path = os.path.join(results_path,"permutation.npy")

    tour=tsp.create_cities(
    11, 
    low= -40,
    high=40, 
    seed= 30121999
    )

    tour_ids = tsp.create_tour_ids(tour)
    N_cities = len(tour_ids)
    D = tsp.construct_D_matrix(
        tour,
        tsp.euclidian_dist
        )
    t_lims = (
                0.5,
                50, 
                0.01
                )
    temperatures = np.arange(t_lims[0],t_lims[1], t_lims[2], dtype=np.float64)
    data_solve  = solve_system_from_permutations(
        perm_path,
        tour_ids,
        D,
        t_lims,
        results_path,
        temperature=temperatures
    )
    data_rel = data_solve[1:]

    
    temp = np.arange(t_lims[0],t_lims[1],t_lims[2])
    data_TSP1,data_TSP_plots = solve_system_with_TSP(
        tour_ids,
        D,
        np.arange(t_lims[0],t_lims[1],t_lims[2],dtype=np.float64),
        10000,
        warmup=2000,
        detailed=False
    )

    L = data_TSP1[1]

    data_tsp_calc = np.array([
        temp,
        np.full(temp.shape, np.nan),
        np.mean(L,axis=1),
        np.mean(L**2, axis=1)-np.mean(L,axis=1)**2,
        (np.mean(L**2, axis=1)-np.mean(L,axis=1)**2)/temperatures**2,
        np.mean(L**2, axis=1)

    ])
   
    

    colums_tsp = [
                r"$T$ (TSP)",
                r"$Z(T)$ (TSP)",
                r"$\langle L \rangle_T$ (TSP)",
                r"$\langle L^2 \rangle_T - \langle L \rangle_T^2 $ (TSP)",
                r"$C(T)$ (TSP)",
                r"$\langle L^2 \rangle_T$ (TSP)",
            ]
    colums_TSP_data = [
                "T",
                "Z(T)",
                "L_mean",
                "var",
                "C(T)",
                "L_sq",
            ]
    colums_metadata = [
        "T",
        "accept_mean"
    ]

    data_tsp_save = data_tsp_calc.T
    save_to_file_system_solved(
        data_tsp_save,
        colums_TSP_data,
        results_path,
        t_lims,
        add_info="TSP"
        )
    
    metadata_TSP_mean = np.mean(data_TSP1[-1], axis=1)
    metadata_TSP_mean = np.column_stack([temp, metadata_TSP_mean])
    
    save_to_file_system_solved(
        metadata_TSP_mean,
        colums_metadata,
        results_path,
        t_lims,
        add_info="TSP_metadata"
    )
    metadata_TSP_mean = metadata_TSP_mean.T


    colums = [
                    r"$T$",
                    r"$Z(T)$",
                    r"$\langle L \rangle_T$",
                    r"$\langle L^2 \rangle_T - \langle L \rangle_T^2 $",
                    r"$C(T)$",
                    r"$\langle L^2 \rangle_T$",
                ]
        
    fig, ax = plt.subplots(len(data_rel),figsize=(8,10))
    
    
    fig, ax = plot_quantities2(
        fig,
        ax,
        data_tsp_calc[1:],
        temp,
        colums_tsp,
        plots_path,
        t_lims=t_lims,
        tsp=True,

    )
    fig, ax = plot_quantities2(
            fig,
            ax,
            data_rel,
            temp,
            colums,
            plots_path,
            t_lims
        )
    plt.show()

    plt.cla()

    fig1, ax1 = plot_quantity(
        metadata_TSP_mean[0],
        metadata_TSP_mean[1],
        dy = np.var(metadata_TSP_mean[1]),
        labels=[colums[0], r"$\langle A \rangle_T$"],
        path=results_path,
        add_info="acceptance",
        save=True

    )

    
def plotting():

    results_path = os.path.join("..","results","solve_system_N11")
    plots_path = os.path.join(results_path,"plots")
    perm_path = os.path.join(results_path,"permutation.npy")

    data_solved = np.loadtxt(os.path.join(results_path,"solved_system_data_0.5_50_0.01_gen.csv"),skiprows=1,delimiter=";").T
    data_TSP = np.loadtxt(os.path.join(results_path,"solved_system_data_0.5_50_0.01_TSP.csv"),skiprows=1,delimiter=";").T
    metadata_TSP_mean = np.loadtxt(os.path.join(results_path,"solved_system_data_0.5_50_0.01_TSP_metadata.csv"),skiprows=1,delimiter=";").T

    temp = data_solved[0]
    data_tsp_calc = data_TSP[1:]
    data_rel = data_solved[1:]
    

    t_lims = (
                    0.5,
                    50, 
                    0.01
                    )
    
    colums_tsp = [
                    r"$T$ (TSP)",
                    r"$Z(T)$ (TSP)",
                    r"$\langle L \rangle_T$ (TSP)",
                    r"$\langle L^2 \rangle_T - \langle L \rangle_T^2 $ (TSP)",
                    r"$C(T)$ (TSP)",
                    r"$\langle L^2 \rangle_T$ (TSP)",
                ]
    colums = [
                        r"$T$",
                        r"$Z(T)$",
                        r"$\langle L \rangle_T$",
                        r"$\langle L^2 \rangle_T - \langle L \rangle_T^2 $",
                        r"$C(T)$",
                        r"$\langle L^2 \rangle_T$",
                    ]
            
    fig, ax = plt.subplots(len(data_rel),figsize=(8,10))
    
    
    fig, ax = plot_quantities2(
        fig,
        ax,
        data_tsp_calc,
        temp,
        colums_tsp,
        plots_path,
        t_lims=t_lims,
        tsp=True,

    )
    fig, ax = plot_quantities2(
            fig,
            ax,
            data_rel,
            temp,
            colums,
            plots_path,
            t_lims,
            fmt="r--",
            save=True
        )
    plt.show()

    plt.cla()

    fig1, ax1 = plot_quantity(
        metadata_TSP_mean[0],
        metadata_TSP_mean[1],
        dy = np.var(metadata_TSP_mean[1]),
        labels=[colums[0], r"$\langle A \rangle_T$"],
        path=results_path,
        add_info="acceptance",
        save=True

    )

    
    





if __name__ == "__main__":
    #main()
    plotting()
    