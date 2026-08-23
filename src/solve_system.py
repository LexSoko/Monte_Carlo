import modules.TSP as tsp
import modules.plotForReport as pr
import numpy as np 
import matplotlib.pyplot as plt
import numba as nb
import os 
import math
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
def solve_system(perm,D_matrix,temperatures):
    perm_array = perm
    N_configs = len(perm_array)
    all_lenghts = np.empty(N_configs, dtype=np.float32)
    for i, perm in enumerate(perm_array):
        all_lenghts[i] = calculate_total_distance_with_D(D_matrix,perm)
    partition__func = np.empty(temperatures.shape, dtype=np.float32)
    expectation_value = np.empty(temperatures.shape, dtype=np.float32)
    expectation_value_L_sq = np.empty(temperatures.shape, dtype=np.float32)
    for k, t in enumerate(temperatures):
        partion = 0 
        weight_sum_expect = 0
        weight_sum_expect_L_sq = 0
        for L in all_lenghts:
            exp = np.exp(-L/t)
            weight_sum_expect += L*exp
            weight_sum_expect_L_sq += L*L*exp
            partion += exp
        partition__func[k] = partion/(2*N_configs)
        expectation_value[k] = weight_sum_expect/partion
        expectation_value_L_sq[k] = weight_sum_expect_L_sq/partion

    return partition__func, expectation_value, expectation_value_L_sq

def save_to_file_system_solved(data,columns, results_path, temp_disc):
    header = ""
    for c in columns:
        header += f"{c};"
    header = header[:-1]
    np.savetxt(
        os.path.join(results_path,f"solved_system_data_{temp_disc[0]}_{temp_disc[1]}_{temp_disc[2]}.csv"),
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
        seed=30121999
        ):
    N_cities = len(tour_ids)
    expected_perm = math.factorial(N_cities-1)
    if os.path.exists(perm_path):
        perm = np.load(perm_path)
        print(f"loaded successfully")
        if len(perm) != expected_perm:
            print(f"length {len(perm)} dont match to expected {expected_perm}")
            perm = permutations2(tour_ids,N_cities)
        else:
            pass
            
    else:
        print(f"couldnt find file , reverting to manual calc")
        perm = permutations2(tour_ids,N_cities)  
        perm = np.array(perm)
        np.save(perm_path,perm)
        
    
       
        print(f"solved system N={len(perm)}")
    
        
    
        colums = [
                r"$T$",
                r"$Z(T)$",
                r"$\langle L \rangle_T$",
                r"$\langle L^2 \rangle_T - \langle L \rangle_T^2 $",
                r"$C(T)$",
                r"$\langle L^2 \rangle_T$",
            ]
    
        if not os.path.exists(os.path.join(results_path,f"solved_system_data_{t_lims[0]}_{t_lims[1]}_{t_lims[2]}.csv")):
    
            temperatures = np.arange(t_lims[0],t_lims[1], t_lims[2])
            partition_func, expectation_value, expectation_value_L_sq = solve_system(
                perm,
                D,
                temperatures=temperatures
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
            save_to_file_system_solved(data,colums,results_path, t_lims)
        else:
            data = np.loadtxt(os.path.join(results_path,f"solved_system_data_{t_lims[0]}_{t_lims[1]}_{t_lims[2]}.csv"),delimiter=";",skiprows=1)
    
        data = data.T
        return data ,colums
   
def plot_quantities(fig,ax,data,colums, path, t_lims):
    
        
    for d in range(1,len(data)):
        ax[d-1].plot(data[0],data[d], label=colums[d])
        ax[d-1].legend()
        ax[d-1].set_xlabel(colums[0])
        ax[d-1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(path,f"solved_system_{t_lims[0]}_{t_lims[1]}_{t_lims[2]}.pdf"))
    return fig,ax



def main():

    results_path = os.path.join("..","results","solve_system")
    plots_path = os.path.join(results_path,"plots")
    perm_path = os.path.join(results_path,"permutation.npy")

    tour=tsp.create_cities(
    10, 
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
                1,
                100, 
                0.05
                )

    data , colums = solve_system_from_permutations(
        perm_path,
        tour_ids,
        D,
        t_lims,
        results_path
    )
    
   
    colums = [
                r"$T$",
                r"$Z(T)$",
                r"$\langle L \rangle_T$",
                r"$\langle L^2 \rangle_T - \langle L \rangle_T^2 $",
                r"$C(T)$",
                r"$\langle L^2 \rangle_T$",
            ]
    
    fig, ax = plt.subplots(len(data)-1,figsize=(8,10))
    
    fig, ax = plot_quantities(
        fig,
        ax,
        data,
        colums,
        plots_path,
        t_lims
    )

    
    





if __name__ == "__main__":
    main()