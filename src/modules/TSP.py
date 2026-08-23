import matplotlib.pyplot as plt
import os 
import random as rd
import numpy as np
import numba as nb
from time import perf_counter
import modules.plotForReport as pr
from warnings import deprecated
from tqdm import tqdm

@nb.njit(cache=True)
def seed_numba(seed):
    """Seed Numba's random-number generator."""
    np.random.seed(seed)
def converge_critiria_true(best_histories):
    """Dummy stopping criterion.

    Replace this later. For example, stop when the best result has not
    improved over several independent runs. Returning False means that all
    configured runs are performed.
    """
    return False    
def create_cities(N, low= -10, high= 10, seed = 12345):
    np.random.seed(seed)
    city_pos = []
    for i in range(N):
        x = np.random.randint(low=low,high=high) + np.random.normal(loc=0, scale= np.sqrt(np.abs(high)/2)) 
        y = np.random.randint(low=low,high=high) + np.random.normal(loc=0, scale= np.sqrt(np.abs(high)/2))
        city_pos.append(np.array([x,y]))
    return np.array(city_pos)
def create_tour_ids(tour):
    return np.arange(0,len(tour), dtype=np.int32)
@nb.njit
def calculate_lenghts(
        arrays, 
        lenght_function = lambda cities: np.linalg.norm(cities)
        ):
    """
    calculates total lenght of a poplulation of city arrays

    Args:
        arrays (np.ndarray): cities array

    Returns:
        np.ndarray: lenghts for each city array
    """
    lenghts = np.empty(len(arrays))
    for i, array in enumerate(arrays):
        lenght = lenght_function(array)  # Vectorized computation
        lenghts[i] = lenght
    return lenghts

def calculate_lenght(cities):
    """
    calculates total lenght of a poplulation of city arrays

    Args:
        arrays (np.ndarray): cities array

    Returns:
        np.ndarray: lenghts for each city array
    """
   
    shifted_array = np.roll(cities, -1, axis=0)
    delta = shifted_array - cities
    lenght = np.sum(np.sqrt(np.sum(delta**2, axis=1)))  # Vectorized computation
        
    return lenght

def calculate_length_tsplib(array):
    return


def validate_lenght(cities):
    lenght1 = calculate_lenght(cities)
    
@nb.njit(cache=True)
def length_func(ordered_coordinates):
    """Closed-tour length using the rounded TSPLIB EUC_2D convention.

    This must use exactly the same distance definition as construct_D_matrix().
    Remove the rounding here if construct_D_matrix() uses raw Euclidean lengths.
    """
    n_cities = ordered_coordinates.shape[0]
    length = 0.0

    for i in range(n_cities):
        next_i = (i + 1) % n_cities
        dx = ordered_coordinates[i, 0] - ordered_coordinates[next_i, 0]
        dy = ordered_coordinates[i, 1] - ordered_coordinates[next_i, 1]
        distance = np.sqrt(dx * dx + dy * dy)
        length += np.floor(distance + 0.5)

    return length

def make_loop(new_path):
    """intended for plotting to close the gap between the start city and the end city

    Args:
        new_path (np.ndarray): path to create loop

    Returns:
        np.ndarray: looped path
    """
    new_path = np.concatenate([new_path, [new_path[0]]],axis=0)
    
    x = new_path.T[0]
    y = new_path.T[1]
    return np.array([x,y])


@nb.njit(inline='always')
def reverse_subsequence(city_ids, a , b):

    while a < b:
        temp = city_ids[a]
        city_ids[a] = city_ids[b]
        city_ids[b] = temp
        a+=1
        b-=1
    return city_ids

@nb.njit
def create_diversity_ids(Tour_ID_Matrix:np.ndarray,specific_population_members = [-1]):
    Tour_ID_Matrix_shuffled = np.empty(Tour_ID_Matrix.shape, dtype= Tour_ID_Matrix.dtype)
    for n, tour in enumerate(Tour_ID_Matrix):
        tour_shuffled = tour.copy()
        if specific_population_members[0] != -1:
            if n in specific_population_members:
                np.random.shuffle(tour_shuffled)
            else:
                pass    
        else:
            np.random.shuffle(tour_shuffled)

        Tour_ID_Matrix_shuffled[n] = tour_shuffled
    return Tour_ID_Matrix_shuffled    


@nb.njit(inline='always')
def annealing_step_Dmatrix(tour_ids,D,temperature,N_cities,Length):
    r1 = np.random.randint(0, N_cities)
    r2 = np.random.randint(0, N_cities)
    
    while r2 == r1:
        r2 = np.random.randint(0, N_cities)
    a = min(r1, r2)
    b = max(r1, r2)
    if a == 0 and b == N_cities -1:
        return tour_ids,Length
    id_a_minus_one =  tour_ids[(a-1)%N_cities]
    id_a = tour_ids[a]
    id_b_plus_one = tour_ids[(b+1)%N_cities]
    id_b = tour_ids[b] 
    
    dL = D[id_a_minus_one,id_b] - D[id_a_minus_one,id_a] + D[id_b_plus_one,id_a] -D[id_b_plus_one,id_b]
    
    U = np.random.random()
    accept = dL <= 0.0
    if not accept and temperature > 0.0:
        accept = (
            U < np.exp(-dL/temperature)
        )
    
    if accept:
        Length = Length + dL
        tour_ids = reverse_subsequence(tour_ids,a,b)
    return tour_ids,Length

@nb.njit(cache=True)
def annealing_D(
        tour_ids,
        D,
        temperature_function,
        Length,
        n_sweeps,
):
    N_cities = len(tour_ids)
    N_cities_sq = N_cities**2
    
    
    
    
    Lengths = np.empty(n_sweeps+1,dtype=np.float32)
    Lengths[0] = Length
    temperatures = np.empty(n_sweeps+1,dtype=np.float32)
    for n in range(n_sweeps):
        temperature = temperature_function(n,tour_ids,Lengths[n])
        current_L = Lengths[n]
        #print(Lengths[n])
        
        for k in range(N_cities_sq):
            r1 = np.random.randint(0, N_cities)
            r2 = np.random.randint(0, N_cities)

            
            while r2 == r1:
                r2 = np.random.randint(0, N_cities)
            a = min(r1, r2)
            b = max(r1, r2)
            if a == 0 and b == N_cities -1:
                continue
            id_a_minus_one =  tour_ids[(a-1)%N_cities]
            id_a = tour_ids[a]
            id_b_plus_one = tour_ids[(b+1)%N_cities]
            id_b = tour_ids[b] 
            
            dL = D[id_a_minus_one,id_b] - D[id_a_minus_one,id_a] + D[id_b_plus_one,id_a] -D[id_b_plus_one,id_b]
            
            U = np.random.random()
            accept = dL <= 0.0
            if not accept and temperature > 0.0:
                accept = (
                    U < np.exp(-dL/temperature)
                )
            
            if accept:
                current_L = current_L + dL
                tour_ids = reverse_subsequence(tour_ids,a,b)
            
        Lengths[n+1] = current_L
        temperatures[n] = temperature
        
    if n_sweeps != 0:
        temperatures[-1] = temperatures[-2]
    return tour_ids, Lengths, temperatures


@nb.njit
def mutation_genetic(
        cities_pos_pop: np.ndarray,
        prob_mut:float = 0.9
):
    
    if prob_mut > 100.0 or prob_mut < 0.0:
        print("probability must be between 0 and 100")
        return cities_pos_pop
    
    for n, city_indv in enumerate(cities_pos_pop):
        if np.random.randint(0,1000) < prob_mut*10:
            a = np.random.randint(0,len(cities_pos_pop[0]))
            b = np.random.randint(0,len(cities_pos_pop[0]))
            city_indv[a] , city_indv[b] = city_indv[b] , city_indv[a]       
            
    return cities_pos_pop

   
@nb.njit(inline='always')
def choose_survivors_ids(old_generation,lenghts,population_size, N_cities):
    """
    chooses random pairs of the population and compares them
    the one which has the shorter path survives.
    kind of a battle to the death in the colosseum as i imagine it

    Args:
        old_generation (np.ndarray): old population
        lenghts (np.ndarray): lenghts of old population

    Returns:
        np.ndarray: survivors half of the old population
    """
    mid = population_size//2
    indeces = np.arange(0,population_size)
    np.random.shuffle(indeces)
    old_generation = old_generation[indeces]
    lenghts = lenghts[indeces]
    survivor_lengths = np.empty((mid,N_cities), dtype=np.float64)
    survivors = np.empty((mid,N_cities),dtype=np.int32)
    for i in range(mid):
        if lenghts[i] < lenghts[i + mid]:
            survivors[i] = old_generation[i]
            survivor_lengths[i] = lenghts[i]
        else:
            survivors[i] = old_generation[i + mid]
            survivor_lengths[i] = lenghts[i+ mid]
   
    return survivors , survivor_lengths

@nb.njit(inline='always')
def mate_ids(survivors, N_cities,  D_matrix):
    """
    the kinky part of the algorithm
    chooses random pairs of a population and exchanges the genetic information
    a sub sequence of the path of each parent is taken out and injected into the other one
    the other citys are rearanged to incorparate the subsequence path

    Args:
        survivors (np.ndarray): survivors of the battle to the death

    Returns:
        np.ndarray: new population double the survivors
    """
    N_survivors = len(survivors)
    offspring = np.empty((N_survivors*2, N_cities), dtype=np.int32)
    offspring[0:N_survivors,:] = survivors
    
    indices = np.arange(N_survivors, dtype=np.int32)  
    np.random.shuffle(indices)  
    pairs = np.zeros((N_survivors, 2), dtype=np.int32)

    for i in range(0, N_survivors - 1, 2):
        pairs[i] = (indices[i], indices[i + 1])
        pairs[i + 1] = (indices[i + 1], indices[i])

    for n, (i , j) in enumerate(pairs):
        a = np.random.randint(0,N_cities - 1)
        b = np.random.randint(a,N_cities)
       
        sub_path_i = list(survivors[i][a:b])
        remaining_path_j = np.empty(N_cities - len(sub_path_i),dtype=np.int32)
        
        count = 0
        
        for item in survivors[j]:
            found = False
            for sub_item in sub_path_i:
                if item == sub_item:
                    
                    found = True
                    break
            if not found:
                remaining_path_j[count] = item
                count += 1
            
        remaining_path_j = list(remaining_path_j)
        
        for k in range(0, N_cities):
            if a <= k < b:
                offspring[n +N_survivors,k] = sub_path_i.pop(0)
                
            else:
                offspring[n+N_survivors,k] = remaining_path_j.pop(0)

    offspring_lengths = np.empty(N_survivors*2,dtype=np.float64)
    for kid in range(N_survivors*2):
        length = 0
        for n_c in range(N_cities):
            length += D_matrix[offspring[kid][n_c],offspring[kid][(n_c+1)%N_cities]]
        offspring_lengths[kid] = length

    return offspring , offspring_lengths




@nb.njit(parallel=True)
def mixed_annealing_D(
        Tour_id_matrix,
        D,
        population_size,
        temperature_function,
        Lengths_0,
        n_sweeps
):
    N_cities = Tour_id_matrix.shape[1]
    N_cities_sq = N_cities**2
    

    
    
    Lengths = np.empty((n_sweeps+1,population_size),dtype=np.float32)

    for i in range(population_size):
        Lengths[0][i] = Lengths_0[i]

    temperatures = np.empty(n_sweeps+1,dtype=np.float32)
    
    for n in range(n_sweeps):
        temperature = temperature_function(
            n,
            Tour_id_matrix,
            Lengths[n,0]
            )
        
        current_lengths = Lengths[n].copy()
        survivors, survivor_lengths = choose_survivors_ids(
                    Tour_id_matrix,
                    current_lengths,
                    population_size,
                    N_cities
                    )
        Tour_id_matrix, new_generation_length = mate_ids(
            survivors,
            N_cities,
            D
            )
        current_lengths = new_generation_length
        for p in nb.prange(population_size):
            current_L = current_lengths[p]
            for _ in range(N_cities_sq):
            
                Tour_id_matrix[p], current_L =annealing_step_Dmatrix(
                    Tour_id_matrix[p],
                    D,
                    temperature,
                    N_cities,
                    current_L
                    )
            current_lengths[p] = current_L
        
        Lengths[n+1] = current_lengths
        temperatures[n] = temperature
    if n_sweeps != 0:
        temperatures[-1] = temperatures[-2]
    
    return Tour_id_matrix,Lengths, temperatures

@nb.njit(parallel=True)
def mixed_annealing_D_after(
        Tour_id_matrix,
        D,
        population_size,
        temperature_function,
        Lengths_0,
        n_sweeps
):
    N_cities = Tour_id_matrix.shape[1]
    N_cities_sq = N_cities**2
    

    
    
    Lengths = np.empty((n_sweeps+1,population_size),dtype=np.float32)

    for i in range(population_size):
        Lengths[0][i] = Lengths_0[i]

    temperatures = np.empty(n_sweeps+1,dtype=np.float32)
    
    for n in range(n_sweeps):
        temperature = temperature_function(
            n,
            Tour_id_matrix,
            Lengths[n,0]
            )
        
        current_lengths = Lengths[n].copy()
        for p in nb.prange(population_size):
            current_L = current_lengths[p]
            for _ in range(N_cities_sq):
            
                Tour_id_matrix[p], current_L =annealing_step_Dmatrix(
                    Tour_id_matrix[p],
                    D,
                    temperature,
                    N_cities,
                    current_L
                    )
            current_lengths[p] = current_L
        survivors, survivor_lengths = choose_survivors_ids(
            Tour_id_matrix,
            current_lengths,
            population_size,
            N_cities
            )
        Tour_id_matrix, new_generation_length = mate_ids(
            survivors,
            N_cities,
            D
            )
        Lengths[n+1] = new_generation_length
        temperatures[n] = temperature
    if n_sweeps != 0:
        temperatures[-1] = temperatures[-2]
    
    return Tour_id_matrix,Lengths, temperatures

    
@nb.njit()
def euclidian_dist(city1,city2):
    dist = np.linalg.norm((city1- city2))
    return dist
@nb.njit()
def tsplib_dist(city1,city2):
    dx = city1[0]- city2[0]
    dy = city1[1]- city2[1]
    distance = np.sqrt(dx * dx + dy * dy)
    dist = np.floor(distance + 0.5)
    return dist

@nb.njit(cache=True)
def construct_D_matrix(tour , distance_metric):
    N_cities = len(tour)
    D_matrix = np.empty((N_cities,N_cities))
    tour2 = tour
    
    for p1 in range(N_cities):
        for p2 in range(N_cities):
            D_matrix[p1,p2] = distance_metric(tour[p1], tour2[p2])
    return D_matrix

def save_aggregate_results(directory, histories, final_lengths, runtimes):
    """Save all completed runs after every new result."""
    np.save(directory / "all_length_histories.npy", np.stack(histories))
    np.save(directory / "all_final_lengths.npy", np.asarray(final_lengths))
    np.save(directory / "all_runtimes_seconds.npy", np.asarray(runtimes))

def run_annealing(
        coordinates, 
        distance_matrix, 
        temp_func,
        length_func,
        OUTPUT_DIRECTORY = os.getcwd(),
        WARM_UP_NUMBA = True,
        BASE_RANDOM_SEED=28041999,
        NUMBER_OF_RUNS = 1,
        NUMBER_OF_SWEEPS = 100,
        KNOWN_OPTIMUM = None,
        MAKE_PLOTS_FOR_EACH_RUN =False
        ):
    """Run repeated independent annealing experiments."""
    output_directory = OUTPUT_DIRECTORY / "annealing"
    output_directory.mkdir(parents=True, exist_ok=True)

    histories = []
    final_lengths = []
    runtimes = []
    tour = coordinates.copy()
    starting_length = length_func(tour)#
    #starting_length = calculate_lenght(tour)    
    
    N_cities = len(tour)
    tour_ids = np.arange(0,N_cities, dtype=np.int32)
    if WARM_UP_NUMBA:
        seed_numba(BASE_RANDOM_SEED)
        annealing_D(
            tour_ids,
            distance_matrix,
            temp_func,
            0,
            0,
        )

    tour = coordinates.copy()
    starting_length = length_func(tour)#
    starting_length = calculate_lenght(tour)    

    N_cities = len(tour)
    tour_ids = np.arange(0,N_cities, dtype=np.int32)

    for run_index in tqdm(range(NUMBER_OF_RUNS),desc="Running Annealing"):
        seed = BASE_RANDOM_SEED + run_index
        seed_numba(seed)
        
        start_time = perf_counter()
        final_tour, length_history, temperatures = annealing_D(
            tour_ids,
            distance_matrix,
            temp_func,
            starting_length,
            NUMBER_OF_SWEEPS,
        )
        INITIAL_TEMPERATURE = INITIAL_TEMPERATURE/(run_index+1)
        #print(final_tour,"finaltour")
        tour = tour[final_tour]
        tour_ids = final_tour.copy()
        starting_length = length_history[-1]
        runtime = perf_counter() - start_time

        length_history = np.asarray(length_history)
        temperatures = np.asarray(temperatures)
        best_history = np.minimum.accumulate(length_history)

        run_directory = output_directory / f"run_{run_index:03d}"
        run_directory.mkdir(exist_ok=True)

        np.save(run_directory / "final_tour.npy", final_tour)
        np.save(run_directory / "length_history.npy", length_history)
        np.save(run_directory / "temperature_history.npy", temperatures)
        np.save(run_directory / "runtime_seconds.npy", np.array(runtime))
        np.save(run_directory / "random_seed.npy", np.array(seed))

        histories.append(best_history)
        final_lengths.append(float(best_history[-1]))
        runtimes.append(runtime)
        save_aggregate_results(
            output_directory, histories, final_lengths, runtimes
        )

        if MAKE_PLOTS_FOR_EACH_RUN:
            figure, _ = pr.plot_diagnostics(
                best_history=best_history,
                temperature_history=pr.one_temperature_history(temperatures),
                optimum=KNOWN_OPTIMUM,
                title=f"Annealing run {run_index}",
                save_path=run_directory / "diagnostics.png",
            )
            plt.close(figure)

            figure, _ = pr.plot_returned_tour(
                coordinates,
                final_tour,
                title=f"Annealing final tour, run {run_index}",
                save_path=run_directory / "final_tour.png",
            )
            plt.close(figure)

        print(
            f"Annealing run {run_index}: "
            f"best={best_history[-1]:.3f}, time={runtime:.3f} s"
        )

        if converge_critiria_true(np.asarray(histories)):
            break

    return {
        "histories": np.asarray(histories),
        "final_lengths": np.asarray(final_lengths),
        "runtimes": np.asarray(runtimes),
    }


def run_mixed(
        coordinates, 
        distance_matrix, 
        temp_func,
        length_func,
        OUTPUT_DIRECTORY = os.getcwd(),
        WARM_UP_NUMBA = True,
        BASE_RANDOM_SEED=28041999,
        NUMBER_OF_RUNS = 1,
        NUMBER_OF_SWEEPS = 100,
        POPULATION_SIZE = 4,
        KNOWN_OPTIMUM = None,
        MAKE_PLOTS_FOR_EACH_RUN =False,
        MODIFY_INITIAL_TEMP = lambda run_idx,last_temps: last_temps[0]
        ):
    """Run repeated independent mixed-population experiments."""
    output_directory = OUTPUT_DIRECTORY / "mixed"
    output_directory.mkdir(parents=True, exist_ok=True)

    histories = []
    final_lengths = []
    runtimes = []
    tour = coordinates.copy()
    starting_length = length_func(tour)#
    Lengths = np.empty(POPULATION_SIZE,dtype=np.float32)
        #starting_length = calculate_lenght(tour)    

    N_cities = len(tour)
   
    #if N_cities < 65500:
    #    tour_ids = np.arange(0,N_cities,dtype=np.int16)    
    #    Tour_id_matrix = np.empty((POPULATION_SIZE,N_cities),dtype=np.int16)
    #else:
    tour_ids = np.arange(0,N_cities,dtype=np.int32)
    Tour_id_matrix = np.empty((POPULATION_SIZE,N_cities),dtype=np.int32)
    for n_p in range(POPULATION_SIZE):
        Tour_id_matrix[n_p] = tour_ids
        Lengths[n_p] = starting_length
    print("Input dtype:", Tour_id_matrix.dtype)
    if WARM_UP_NUMBA:
        seed_numba(BASE_RANDOM_SEED)
        mixed_annealing_D(
            Tour_id_matrix,
            distance_matrix,
            POPULATION_SIZE,
            temp_func,
            Lengths,
            0,
        )

    for run_index in tqdm(range(NUMBER_OF_RUNS), desc = "Running Mixed"):
        seed = BASE_RANDOM_SEED + run_index
        seed_numba(seed)

        start_time = perf_counter()
        final_population, length_history, temperatures = mixed_annealing_D(
            Tour_id_matrix,
            distance_matrix,
            POPULATION_SIZE,
            temp_func,
            Lengths,
            NUMBER_OF_SWEEPS,
        )
        
        Tour_id_matrix = final_population.copy()
        print("Input dtype:", Tour_id_matrix.dtype)
        Lengths = length_history[-1]

        runtime = perf_counter() - start_time

        final_population = np.asarray(final_population)
        length_history = np.asarray(length_history)
        temperatures = np.asarray(temperatures)

        if length_history.ndim == 1:
            population_history = length_history[:, None]
        elif length_history.ndim == 2:
            population_history = length_history
        else:
            raise ValueError("Mixed length history must be 1D or 2D")

        best_history = np.minimum.accumulate(
            np.min(population_history, axis=1)
        )
        mean_history = np.mean(population_history, axis=1)
        variance_history = np.var(population_history, axis=1)
        best_population_index = int(np.argmin(population_history[-1]))

        run_directory = output_directory / f"run_{run_index:03d}"
        run_directory.mkdir(exist_ok=True)

        np.save(run_directory / "final_population.npy", final_population)
        np.save(run_directory / "length_history.npy", population_history)
        np.save(run_directory / "temperature_history.npy", temperatures)
        np.save(run_directory / "runtime_seconds.npy", np.array(runtime))
        np.save(run_directory / "random_seed.npy", np.array(seed))

        histories.append(best_history)
        final_lengths.append(float(best_history[-1]))
        runtimes.append(runtime)
        save_aggregate_results(
            output_directory, histories, final_lengths, runtimes
        )

        if MAKE_PLOTS_FOR_EACH_RUN:
            figure, _ = pr.plot_population(
                population_history,
                optimum=KNOWN_OPTIMUM,
                title=f"Mixed population, run {run_index}",
                save_path=run_directory / "population.png",
            )
            plt.close(figure)

            figure, _ = pr.plot_diagnostics(
                best_history=best_history,
                mean_history=mean_history,
                variance_history=variance_history,
                temperature_history=pr.one_temperature_history(temperatures),
                optimum=KNOWN_OPTIMUM,
                title=f"Mixed solver run {run_index}",
                save_path=run_directory / "diagnostics.png",
            )
            plt.close(figure)

            best_tour = final_population[best_population_index]
            figure, _ = pr.plot_returned_tour(
                coordinates,
                best_tour,
                title=f"Mixed final tour, run {run_index}",
                save_path=run_directory / "final_tour.png",
            )
            plt.close(figure)

        print(
            f"Mixed run {run_index}: "
            f"best={best_history[-1]:.3f}, time={runtime:.3f} s"
        )

        if converge_critiria_true(np.asarray(histories)):
            break

    return {
        "histories": np.asarray(histories),
        "final_lengths": np.asarray(final_lengths),
        "runtimes": np.asarray(runtimes),
    }




@deprecated("This function has been deprecated")
def annealing_change(
    cities_pos: np.ndarray,
    temperature: np.float64,
    start_lenght: np.ndarray
    ):
    """mutations that are applied to the cities array population. This mutation is an annealing process.
        different to a genetic algorithm the mutation acceptance is based on the temperature of the system.
        the idea was to have sort of an outside parameter which can affect the population

    Args:
        cities_pos_pop (np.ndarray): population of cities arrays
        temperature (np.float64): temperature of the system
        start_lenghts (np.ndarray): lenghts of city paths which is modified after each mutation

    Returns:
        np.ndarray , np.ndarray: modified population, new lenghts
    """
    N_cities = len(cities_pos)
    city_ids = np.arange(0,N_cities,dtype=np.int32)
    #cities_pos = np.copy(cities_pos)
    r1 = np.random.randint(0, N_cities)
    r2 = np.random.randint(0, N_cities)
    
    while r2 == r1:
        r2 = np.random.randint(0, N_cities)
    
    a = np.min([r1, r2])
    b = np.max([r1, r2])
    
    id_a_minus_one =  city_ids[(a-1)%N_cities]
    id_a = city_ids[a]
    id_b_plus_one = city_ids[(b+1)%N_cities]
    id_b = city_ids[b] 
    
    if ((a,b) != (0,len(cities_pos)-1)):
        e_Kprime = np.sqrt(
            np.sum((
                cities_pos[(a-1)%len(cities_pos)] - cities_pos[(b)%len(cities_pos)])**2)) + \
                        np.sqrt(np.sum((cities_pos[(a)%len(cities_pos)] -cities_pos[(b+1)%len(cities_pos)])**2)) 
        e_K = np.sqrt(np.sum(((cities_pos[(a-1)%len(cities_pos)]- cities_pos[(a)%len(cities_pos)])**2)) + np.sqrt(np.sum((cities_pos[(b)%len(cities_pos)]- cities_pos[(b+1)%len(cities_pos)]))**2))
        dE = e_Kprime - e_K  
    else:
        dE = 0.0
    
    if np.exp(-dE/temperature) > np.random.random():
        
        city_ids = reverse_subsequence(city_ids,a,b)
        start_lenght += dE
        
            
    
    return cities_pos , start_lenght
@deprecated("This function has been deprecated")
def run_mixed_fixedN(N,temp_func):
    """runnes fixed amount of iteratuons

    Args:
        N (int): number of iterations
        temp_func (Callable): temperature function used

    Returns:
        np.ndarray, np.ndarray: optimized paths, corresponding lenghts
    """
    time1 = time.time()
    for k in range(N):
        temp = temp_func(k)
        survivors = choose_survivors(all_cities_specimen,all_cities_lenghts)
        all_cities_specimen = mate(survivors)
        for n in range(all_cities_specimen.shape[1]**2):        
            all_cities_specimen , all_cities_lenghts = mutation(all_cities_specimen,temp,all_cities_lenghts)
        all_cities_lenghts = calculate_lenghts(all_cities_specimen)
    time2 = time.time()
    print(f"calculation took : {time2 - time1 } s")
    return all_cities_specimen, all_cities_lenghts

@deprecated("This function has been deprecated")
def run_mixed(all_cities_specimen, all_cities_lenghts,number_mutations,temp):
    """
    implementation used for gui, runs one sweep 

    Args:
        all_cities_specimen (np.ndarray): cities population
        all_cities_lenghts (np.ndarray): corresponding lenghts
        number_mutations (Int): number of mutations applied
        temp (Float): temperature for acceptance probability

    Returns:
        np.ndarray,np.ndarray: new cities population, new corresponding lenghts
    """
    survivors = choose_survivors(all_cities_specimen,all_cities_lenghts)
    all_cities_specimen = mate(survivors)
    for n in range(number_mutations):        
        all_cities_specimen , all_cities_lenghts = mutation(all_cities_specimen,temp,all_cities_lenghts)
    all_cities_lenghts = calculate_lenghts(all_cities_specimen)

    return all_cities_specimen, all_cities_lenghts

@deprecated("This function has been deprecated")
def choose_survivors(old_generation,lenghts,population_size, N_cities):
    """
    chooses random pairs of the population and compares them
    the one which has the shorter path survives.
    kind of a battle to the death in the colosseum as i imagine it

    Args:
        old_generation (np.ndarray): old population
        lenghts (np.ndarray): lenghts of old population

    Returns:
        np.ndarray: survivors half of the old population
    """
    mid = population_size//2
    indeces = np.arange(0,population_size)
    np.random.shuffle(indeces)
    old_generation = old_generation[indeces]
    lenghts = lenghts[indeces]
    
    survivors = np.empty((mid,N_cities,2))
    for i in range(mid):
        if lenghts[i] < lenghts[i + mid]:
            survivors[i] = old_generation[i]
        else:
            survivors[i] = old_generation[i + mid]
    return survivors
@deprecated("This function has been deprecated")
def mate(survivors:np.ndarray, N_cities ,length_function):
    """
    the kinky part of the algorithm
    chooses random pairs of a population and exchanges the genetic information
    a sub sequence of the path of each parent is taken out and injected into the other one
    the other citys are rearanged to incorparate the subsequence path

    Args:
        survivors (np.ndarray): survivors of the battle to the death

    Returns:
        np.ndarray: new population double the survivors
    """
    N_survivors = len(survivors)
    offspring = np.empty((int(N_survivors*2), N_cities, 2))
    offspring[0:N_survivors,:,:] = survivors
    
    indices = np.arange(N_survivors, dtype=np.int32)  
    np.random.shuffle(indices)  
    pairs = np.zeros((N_survivors, 2), dtype=np.int32)

    for i in range(0, N_survivors - 1, 2):
        pairs[i] = (indices[i], indices[i + 1])
        pairs[i + 1] = (indices[i + 1], indices[i])

    for n, (i , j) in enumerate(pairs):
        a = np.random.randint(0,N_cities - 1)
        b = np.random.randint(a,N_cities)
       
        sub_path_i = list(survivors[i][a:b])
        remaining_path_j = np.empty((N_cities - len(sub_path_i), survivors[j].shape[1]))
        
        count = 0
        
        for item in survivors[j]:
            found = False
            for sub_item in sub_path_i:
                if np.all(item == sub_item):
                    
                    found = True
                    break
            if not found:
                remaining_path_j[count] = item
                count += 1
            
        remaining_path_j = list(remaining_path_j)
        
        for k in range(0, N_cities):
            if a <= k < b:
                offspring[n +N_survivors,k,:] = sub_path_i.pop(0)
                
            else:
                offspring[n+N_survivors,k,:] = remaining_path_j.pop(0)

    return offspring

@deprecated("This function has been deprecated")
def mutation(
    cities_pos_pop: np.ndarray,
    temperature: np.float64,
    start_lenghts: np.ndarray
    ):
    """mutations that are applied to the cities array population. This mutation is an annealing process.
        different to a genetic algorithm the mutation acceptance is based on the temperature of the system.
        the idea was to have sort of an outside parameter which can affect the population

    Args:
        cities_pos_pop (np.ndarray): population of cities arrays
        temperature (np.float64): temperature of the system
        start_lenghts (np.ndarray): lenghts of city paths which is modified after each mutation

    Returns:
        np.ndarray , np.ndarray: modified population, new lenghts
    """
    
    for n, city_indv in enumerate(cities_pos_pop):
        a = np.random.randint(0,len(city_indv))
        b = np.random.randint(0,len(city_indv))
        
        if a > b:
            tempindex = a
            a = b
            b = tempindex  
        
        if a != b:
            if ((a,b) != (0,len(city_indv)-1)):
                e_Kprime = np.sqrt(
                    np.sum((city_indv[(a-1)%len(city_indv)] -city_indv[(b)%len(city_indv)])**2)) + np.sqrt(np.sum((city_indv[(a)%len(city_indv)] -city_indv[(b+1)%len(city_indv)])**2)) 
                e_K = np.sqrt(np.sum(((city_indv[(a-1)%len(city_indv)]- city_indv[(a)%len(city_indv)])**2))) + np.sqrt(np.sum((city_indv[(b)%len(city_indv)]- city_indv[(b+1)%len(city_indv)]))**2)
                dE = e_Kprime - e_K  
            else:
                dE = 0.0
        else:
            dE = 0.0
        if np.exp(-dE/temperature) > np.random.random():
            
            subarray = city_indv[a:b+1]
            subarray_rev = subarray[::-1]
            city_indv[a:b+1] = subarray_rev
            start_lenghts[n] += dE
        
            
    
    return cities_pos_pop , start_lenghts
@deprecated("This function has been deprecated")
def create_diversity(cities,n = 2):
    """creates random starting sequences from one

    Args:
        cities (np.ndarray): array with cites positions
        n (int, optional):number of new paths created. Defaults to 2.

    Returns:
        np.ndarray: random population
    """
    population = []
    for i in range(n):
        shuffled_arr = cities.copy()
        np.random.shuffle(shuffled_arr)
        population.append(list(shuffled_arr))
    return np.array(population)

@deprecated("This function has been deprecated")
def run_annealing_fixedN(N,cities,energy,temp_func):

    Energy_avg = np.empty(N)
    Energy_var = np.empty(N)        
    Heat_cap = np.empty(N)
    acceptance = np.empty(N)
    
    
    city_pos = np.copy(cities)
    for i in range(N):
        energy_inter = np.empty(len(cities)**2)
        temp = temp_func(i)
        accept = 0
        for j in range(len(cities)**2):
            
            city_pos , energy = annealing_change(city_pos,temp,energy)
            
            energy_inter[j] = energy
            if (j > 0) and (energy != energy_inter[j-1]):
                accept += 1

        acceptance[i] = accept/(len(cities)**2)
        Energy_avg[i] = np.mean(energy_inter)
        Energy_var[i] = np.var(energy_inter)
        Heat_cap[i]  = np.var(energy_inter)/(temp**2)
    
    return city_pos ,Energy_avg , Energy_var , Heat_cap