import matplotlib.pyplot as plt
import os 
import random as rd
import numpy as np
import numba as nb
from time import perf_counter
import modules.plotForReport as pr
from warnings import deprecated
from tqdm import tqdm
import time 





@nb.njit(cache=True)
def benchmark_temperature_function(
        sweep,
        state,
        length,
):
    return np.float64(1000.0)



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
    return np.array(city_pos, dtype=np.float64)

@nb.njit(inline='always', cache=True)
def calculate_total_distance_with_D(D, tour_ids):
    total_lenght = 0
    N_cities = len(tour_ids)
    for i in range(N_cities):
        total_lenght += D[tour_ids[i],tour_ids[(i+1)%N_cities]]
    return total_lenght

def create_tour_ids(tour):
    return np.arange(0,len(tour), dtype=np.int32)
@nb.njit(cache=True)
def create_tour_id_matrix(tours_ids,population_size):
    N_cities = len(tours_ids)
    
    Tour_ID_Matrix = np.empty(
        (population_size, N_cities),
        dtype=np.int32
    )

    for p in range(population_size):
        Tour_ID_Matrix[p] = tours_ids.copy()
    return Tour_ID_Matrix

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
@nb.njit(cache=True)
def subseq_dist_2(pop ,min_sq,max_sq):
    pop_size = pop.shape[0]
    corr1 = np.empty((pop_size,pop_size), dtype=np.int64)
    maximial_sequence = max_sq
    minimal_sequence = min_sq
    pop = pop + 1
    
    pop1 = pop.copy()
    pop2 = pop.copy()
    matched = np.int64(0)
    for i in range(pop_size):
        p1 = pop1[i]
        for j in range(i,pop_size):
            p2 = pop2[j]
            matched = np.int64(0)
            
            for a in range(pop.shape[1]):
                shifted = p1 - np.roll(p2,a)
                shifted_reversed = p1[::-1] -np.roll(p2,a)
                
                seq = 0
                seq_r = 0
                zeros = np.argwhere(shifted==0)
                zeros_r = np.argwhere(shifted_reversed==0)
              
                if len(zeros) < minimal_sequence:
                    continue
                else:
                    for n, zero in enumerate(zeros):
                        
                        if np.abs(zeros[(n+1)%len(zeros)]- zeros[n]) == 1:
                           
                            seq += 1
                        else:
                            seq = 0
                        if minimal_sequence <= seq <= maximial_sequence:
                            
                            matched += 1
                if len(zeros_r) < minimal_sequence:
                    continue
                else:
                    for k, zero in enumerate(zeros_r):
                        
                        if np.abs(zeros_r[(k+1)%len(zeros_r)]- zeros_r[k]) == 1:
                            
                            seq_r += 1
                        else:
                            seq_r = 0
                        if minimal_sequence <= seq_r <= maximial_sequence:
                            matched += 1

            corr1[i,j] = matched
            corr1[j,i] = matched

    
    norm = pop.shape[0]*pop.shape[0]*(max_sq-min_sq)
    likelyness = np.sum(corr1)/norm
    rel_to_disorder = 1.0 - likelyness
    return corr1, rel_to_disorder, likelyness
@nb.njit(inline="always")
def count_circular_runs(flags, min_sequence, max_sequence):
    N = len(flags)

    # Find a false value so a sequence crossing the array
    # boundary is counted as one circular sequence.
    start = -1

    for i in range(N):
        if flags[i] == 0:
            start = i
            break

    # Every edge matches.
    if start == -1:
        if N < min_sequence:
            return np.int64(0)

        return np.int64(
            min(N, max_sequence) - min_sequence + 1
        )

    matched = np.int64(0)
    sequence = 0

    for offset in range(1, N + 1):
        i = (start + offset) % N

        if flags[i] == 1:
            sequence += 1
        else:
            if sequence >= min_sequence:
                matched += (
                    min(sequence, max_sequence)
                    - min_sequence
                    + 1
                )

            sequence = 0

    return matched


@nb.njit(cache=True)
def subseq_dist_2_fast(pop, min_sq, max_sq):
    population_size = pop.shape[0]
    N_cities = pop.shape[1]

    corr = np.empty(
        (population_size, population_size),
        dtype=np.int64
    )

    # For each city, store its next and previous city.
    successor = np.empty(
        (population_size, N_cities),
        dtype=np.int32
    )
    predecessor = np.empty(
        (population_size, N_cities),
        dtype=np.int32
    )

    for p in range(population_size):
        for i in range(N_cities):
            city = pop[p, i]

            successor[p, city] = pop[
                p, (i + 1) % N_cities
            ]
            predecessor[p, city] = pop[
                p, (i - 1) % N_cities
            ]

    forward_matches = np.empty(
        N_cities,
        dtype=np.uint8
    )
    reversed_matches = np.empty(
        N_cities,
        dtype=np.uint8
    )

    # The result is symmetric, so only calculate half.
    for i in range(population_size):
        p1 = pop[i]

        for j in range(i, population_size):
            for k in range(N_cities):
                city = p1[k]
                next_city = p1[(k + 1) % N_cities]

                forward_matches[k] = (
                    successor[j, city] == next_city
                )

                reversed_matches[k] = (
                    predecessor[j, city] == next_city
                )

            matched = (
                count_circular_runs(
                    forward_matches,
                    min_sq,
                    max_sq
                )
                + count_circular_runs(
                    reversed_matches,
                    min_sq,
                    max_sq
                )
            )

            corr[i, j] = matched
            corr[j, i] = matched

    # Kept from your original definition.
    norm = (
        population_size
        * population_size
        * (max_sq - min_sq)
    )

    likelyness = np.sum(corr) / norm
    rel_to_disorder = 1.0 - likelyness

    return corr, rel_to_disorder, likelyness

@nb.njit(cache=True)
def mean_edge_diversity(pop):
    population_size = pop.shape[0]
    N_cities = pop.shape[1]

    if population_size < 2:
        return 0.0

    # Store both neighbours of every city in every tour.
    neighbours = np.empty(
        (population_size, N_cities, 2),
        dtype=np.int32
    )

    for p in range(population_size):
        for i in range(N_cities):
            city = pop[p, i]

            neighbours[p, city, 0] = pop[
                p, (i - 1) % N_cities
            ]
            neighbours[p, city, 1] = pop[
                p, (i + 1) % N_cities
            ]

    total_diversity = 0.0
    number_pairs = 0

    for i in range(population_size):
        for j in range(i + 1, population_size):
            shared_edges = 0

            for k in range(N_cities):
                city_1 = pop[i, k]
                city_2 = pop[i, (k + 1) % N_cities]

                if (
                    neighbours[j, city_1, 0] == city_2
                    or neighbours[j, city_1, 1] == city_2
                ):
                    shared_edges += 1

            pair_diversity = (
                1.0 - shared_edges / N_cities
            )

            total_diversity += pair_diversity
            number_pairs += 1

    return total_diversity / number_pairs

@nb.njit(cache=True)
def mean_edge_diversity_matrix(pop):
    population_size = pop.shape[0]
    N_cities = pop.shape[1]
    edge_matrix = np.empty((population_size,population_size), dtype=np.float64)
    if population_size < 2:
        return edge_matrix

    # Store both neighbours of every city in every tour.
    neighbours = np.empty(
        (population_size, N_cities, 2),
        dtype=np.int32
    )

    for p in range(population_size):
        for i in range(N_cities):
            city = pop[p, i]

            neighbours[p, city, 0] = pop[
                p, (i - 1) % N_cities
            ]
            neighbours[p, city, 1] = pop[
                p, (i + 1) % N_cities
            ]

    number_pairs = population_size * population_size

    for i in range(population_size):
        for j in range(i + 1, population_size):
            shared_edges = 0

            for k in range(N_cities):
                city_1 = pop[i, k]
                city_2 = pop[i, (k + 1) % N_cities]

                if (
                    neighbours[j, city_1, 0] == city_2
                    or neighbours[j, city_1, 1] == city_2
                ):
                    shared_edges += 1

            pair_diversity = (
                1.0 - shared_edges / N_cities
            )
            edge_matrix[i,j] = pair_diversity
            edge_matrix[j,i] = pair_diversity
            
            

    return edge_matrix / number_pairs

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


@nb.njit(inline='always', cache=True)
def reverse_subsequence(city_ids, a , b):

    while a < b:
        temp = city_ids[a]
        city_ids[a] = city_ids[b]
        city_ids[b] = temp
        a+=1
        b-=1
    return city_ids

@nb.njit
def create_diversity_ids(Tour_ID_Matrix,specific_population_members):
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


@nb.njit(inline='always', cache=True)
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

@nb.njit(inline='always', cache=True)
def annealing_step_Dmatrix2(tour_ids,D,temperature,N_cities,Length):
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

@nb.njit(cache=True)
def annealing_D_detailed(
        tour_ids,
        D,
        temperature_function,
        Length,
        n_sweeps,
        const_temp,
        warm_up,
        detailed 
):
    N_cities = len(tour_ids)
    N_cities_sq = N_cities**2

    Lengths = np.empty(n_sweeps + 1, dtype=np.float32)
    Lengths[0] = Length

    temperatures = np.empty(n_sweeps + 1, dtype=np.float32)

    # Raw Markov-chain samples at every proposed move.
    length_samples = np.empty(
        (n_sweeps, N_cities_sq),
        dtype=np.float64
    )

    # Quantities that can be compared with the exactly solved system.
    mean_L = np.empty(n_sweeps, dtype=np.float64)
    mean_L_squared = np.empty(n_sweeps, dtype=np.float64)
    variance_L = np.empty(n_sweeps, dtype=np.float64)
    heat_capacity = np.empty(n_sweeps, dtype=np.float64)

    # Additional useful diagnostic.
    acceptance_rate = np.empty(n_sweeps, dtype=np.float64)

    for n in range(n_sweeps):
        if const_temp >= 0.0:
            if warm_up == True:
                temperature = temperature_function(
                    n,
                    const_temp,
                    Lengths[n],
                    )
            else:
                temperature = const_temp
        else:
            temperature = temperature_function(
            n,
            const_temp,
            Lengths[n],
            )

        current_L = Lengths[n]
        accepted_moves = 0

        sum_L = 0.0
        sum_L_squared = 0.0

        for k in range(N_cities_sq):
            r1 = np.random.randint(0, N_cities)
            r2 = np.random.randint(0, N_cities)

            while r2 == r1:
                r2 = np.random.randint(0, N_cities)

            a = min(r1, r2)
            b = max(r1, r2)

            if a == 0 and b == N_cities - 1:
                if detailed:
                    length_samples[n, k] = current_L
                    sum_L += current_L
                    sum_L_squared += current_L * current_L
                continue

            id_a_minus_one = tour_ids[(a - 1) % N_cities]
            id_a = tour_ids[a]
            id_b_plus_one = tour_ids[(b + 1) % N_cities]
            id_b = tour_ids[b]

            dL = (
                D[id_a_minus_one, id_b]
                - D[id_a_minus_one, id_a]
                + D[id_b_plus_one, id_a]
                - D[id_b_plus_one, id_b]
            )

            U = np.random.random()

            accept = dL <= 0.0

            if not accept and temperature > 0.0:
                accept = U < np.exp(-dL / temperature)

            if accept:
                current_L = current_L + dL
                tour_ids = reverse_subsequence(tour_ids, a, b)
                accepted_moves += 1

            if detailed:
                length_samples[n, k] = current_L

            sum_L += current_L
            sum_L_squared += current_L * current_L
        if detailed:
            mean_L[n] = sum_L / N_cities_sq
            mean_L_squared[n] = sum_L_squared / N_cities_sq 
            variance = (
                mean_L_squared[n]
                - mean_L[n] * mean_L[n]
            )   
        
            if variance < 0.0 and variance > -1e-10:
                variance = 0.0  
            variance_L[n] = variance    
            if temperature > 0.0:
                heat_capacity[n] = variance / (temperature * temperature)
            else:
                heat_capacity[n] = np.nan

        acceptance_rate[n] = accepted_moves / N_cities_sq

        Lengths[n + 1] = current_L
        temperatures[n] = temperature

    if n_sweeps != 0:
        temperatures[-1] = temperatures[-2]

    return (
        tour_ids,
        Lengths,
        temperatures,
        length_samples,
        mean_L,
        mean_L_squared,
        variance_L,
        heat_capacity,
        acceptance_rate,
    )

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

   

@nb.njit(inline='always', cache=True)
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



@nb.njit(inline='always', cache=True)
def mate_ids2(survivors, N_cities,  D_matrix , surviving_ids, dead_ids  ):
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
    #unique_ids[dead_ids] = ""
    N_survivors = len(survivors)
    mating_ids = np.empty((N_survivors,3))
    offspring = np.empty((N_survivors*2, N_cities), dtype=np.int32)
    for n ,surv_id in enumerate(surviving_ids):
        offspring[surv_id] = survivors[n]
    
    indices = np.arange(N_survivors, dtype=np.int32)
    np.random.shuffle(indices)  
    pairs = np.zeros((N_survivors, 2), dtype=np.int32)

    for i in range(0, N_survivors - 1, 2):
        pairs[i] = (indices[i], indices[i + 1])
        pairs[i + 1] = (indices[i + 1], indices[i])

    for n, (i , j) in enumerate(pairs):
        mating_ids[n][0] = surviving_ids[i]
        mating_ids[n][1] = surviving_ids[j]
        mating_ids[n][2] = dead_ids[n]
        dominant_genom = i
        r1 = np.random.randint(0, N_cities)
        r2 = np.random.randint(0, N_cities)
        while r2 == r1:
            r2 = np.random.randint(0, N_cities)
        a = min(r1, r2)
        b = max(r1, r2)
        if a == 0 and b == N_cities - 1:
            a = a + 1 
            b = b - 1
        if abs(a-b) < N_cities//2:
            dominant_genom = j

        
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
                offspring[dead_ids[n],k] = sub_path_i.pop(0)
                
            else:
                offspring[dead_ids[n],k] = remaining_path_j.pop(0)
            #unique_ids[dead_ids[n]]

    offspring_lengths = np.empty(N_survivors*2,dtype=np.float64)
    
    for kid in range(N_survivors*2):
        offspring_lengths[kid] = calculate_total_distance_with_D(D_matrix,offspring[kid])

    return offspring , offspring_lengths , mating_ids

@nb.njit(cache=True)
def choose_survivors_ids2(old_generation,lenghts,population_size, N_cities):
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
    #unique_ids = unique_ids[indeces]
    #old_generation = old_generation[indeces]
    #lenghts = lenghts[indeces]
    survivor_lengths = np.empty((mid,N_cities), dtype=np.float64)
    survivors = np.empty((mid,N_cities),dtype=np.int32)
    pairs = np.zeros((mid, 2), dtype=np.int32)
    surviving_ids = np.empty(mid, dtype=np.int32)
    dead_ids = np.empty(mid, dtype=np.int32)
    
    for i in range(0, population_size-1, 2):
            pairs[i//2] = (indeces[i], indeces[i + 1])
    
    for n, (i , j) in enumerate(pairs):
        if lenghts[i] < lenghts[j]:
            survivors[n] = old_generation[i]
            survivor_lengths[n] = lenghts[i]
            surviving_ids[n] = i    
            dead_ids[n] = j     
        else:
            survivors[n] = old_generation[j]
            survivor_lengths[n] = lenghts[j]
            surviving_ids[n] = j
            dead_ids[n] = i
    
            
   
    return survivors , survivor_lengths , surviving_ids, dead_ids

@nb.njit(inline='always', cache=True)
def mate_ids3(survivors, N_cities,  D_matrix , surviving_ids, dead_ids  ):
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
    #unique_ids[dead_ids] = ""
    N_survivors = len(survivors)
    mating_ids = np.empty((N_survivors,3))
    offspring = np.empty((N_survivors*2, N_cities), dtype=np.int32)
    for n ,surv_id in enumerate(surviving_ids):
        offspring[surv_id] = survivors[n]
    edge_matrix = mean_edge_diversity_matrix(survivors)

    idx = np.argsort(edge_matrix.ravel())[::-1]

    n_cols = edge_matrix.shape[1]

    i_div = idx // n_cols
    j_div = idx % n_cols
    #t = 0
    #i_uniq = np.empty(N_survivors)
    #j_uniq = np.empty(N_survivors)
    #for l , k in zip(i_div,j_div):

    pairs = np.zeros((N_survivors, 2), dtype=np.int32)

    for i in range(0, N_survivors):
        pairs[i] = (i_div[i], j_div[i])
        

    for n, (i , j) in enumerate(pairs):
        mating_ids[n][0] = surviving_ids[i]
        mating_ids[n][1] = surviving_ids[j]
        mating_ids[n][2] = dead_ids[n]
        dominant_genom = i
        r1 = np.random.randint(0, N_cities)
        r2 = np.random.randint(0, N_cities)
        while r2 == r1:
            r2 = np.random.randint(0, N_cities)
        a = min(r1, r2)
        b = max(r1, r2)
        if a == 0 and b == N_cities - 1:
            a = a + 1 
            b = b - 1
        if abs(a-b) < N_cities//2:
            dominant_genom = j

        
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
                offspring[dead_ids[n],k] = sub_path_i.pop(0)
                
            else:
                offspring[dead_ids[n],k] = remaining_path_j.pop(0)
            #unique_ids[dead_ids[n]]

    offspring_lengths = np.empty(N_survivors*2,dtype=np.float64)
    
    for kid in range(N_survivors*2):
        offspring_lengths[kid] = calculate_total_distance_with_D(D_matrix,offspring[kid])

    return offspring , offspring_lengths , mating_ids
@nb.njit(inline='always', cache=True)
def mate_ids(survivors, N_cities,  D_matrix ):
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
        a = np.random.randint(0, N_cities -1)
        b = np.random.randint(a, N_cities)
       
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
        offspring_lengths[kid] = calculate_total_distance_with_D(D_matrix,offspring[kid])

    return offspring , offspring_lengths


@nb.njit(parallel=True, cache=True)
def mixed_annealing_D(
        Tour_ID_Matrix,
        D,
        population_size,
        temperature_function,
        interval_mutation,
        n_sweeps,
        save_dead_IDs,
        save_mating_IDs,
        seed
):
    N_cities = Tour_ID_Matrix.shape[1]
    N_cities_sq = N_cities**2
    N_cities_tripple = N_cities*3

    dead_ids_sweep = np.empty((n_sweeps,population_size//2),dtype=np.int32)
    mating_ids_sweep = np.empty((n_sweeps,population_size//2,3),dtype=np.int32)
    
    Lengths = np.empty((n_sweeps+1,population_size),dtype=np.float64)

    #for i in range(population_size):
    #    Lengths[0][i] = Lengths_0[i]
    
    Tour_ID_Matrix = create_diversity_ids(Tour_ID_Matrix, [-1])
    Tour_ID_Matrix_sweep = np.empty((n_sweeps+1,population_size,N_cities), np.int32)
    Tour_ID_Matrix_sweep[0] = Tour_ID_Matrix

    for i in range(population_size):
        Lengths[0][i] = calculate_total_distance_with_D(D,Tour_ID_Matrix[i])

    temperatures = np.empty(n_sweeps+1,dtype=np.float32)
    
    for n in range(n_sweeps):
        temperature = temperature_function(
            n,
            Tour_ID_Matrix,
            Lengths[n,0]
            )
        
        current_lengths = Lengths[n].copy()
        
        if n != 0:
            np.random.seed(seed + n)
            survivors , survivor_lengths , surviving_ids, dead_ids = choose_survivors_ids2(
                        Tour_ID_Matrix,
                        Lengths[n].copy(),
                        population_size,
                        N_cities
                        )   
            Tour_ID_Matrix, current_lengths, mating_ids = mate_ids3(
                        survivors,
                        N_cities,
                        D,
                        surviving_ids,
                        dead_ids
                    )
            if save_dead_IDs == True:
                dead_ids_sweep[n] = dead_ids
            if save_mating_IDs == True:
                mating_ids_sweep[n] = mating_ids
        #current_lengths = new_generation_length
        if n % interval_mutation == 0:
            for p in nb.prange(population_size):
                np.random.seed(
                    seed
                    + 1_000_000
                    + n * population_size
                    + p
                )
                current_L = current_lengths[p]
                for _ in range(N_cities_sq):
                
                    Tour_ID_Matrix[p], current_L =annealing_step_Dmatrix(
                        Tour_ID_Matrix[p],
                        D,
                        temperature,
                        N_cities,
                        current_L
                        )
                current_lengths[p] = current_L
        
        Lengths[n+1] = current_lengths
        temperatures[n] = temperature
        Tour_ID_Matrix_sweep[n+1] = Tour_ID_Matrix
    if n_sweeps != 0:
        temperatures[-1] = temperatures[-2]
    
    return Tour_ID_Matrix,Lengths, temperatures, Tour_ID_Matrix_sweep, dead_ids_sweep, mating_ids_sweep


@nb.njit(parallel=True, cache=True)
def mixed_annealing_D_const_T(
        Tour_ID_Matrix,
        D,
        population_size,
        temperature_function,
        Lengths_0,
        n_sweeps,
        mutations_per_sweep,
        const_temp,
        warm_up,
        detailed,
):
    N_cities = Tour_ID_Matrix.shape[1]

    Lengths = np.empty(
        (n_sweeps + 1, population_size),
        dtype=np.float32
    )
    Lengths[0] = Lengths_0

    temperatures = np.empty(n_sweeps + 1, dtype=np.float32)
    acceptance_rate = np.empty((n_sweeps,population_size), dtype=np.float64)
    accepted_population = np.empty(population_size, dtype=np.int64)

    for n in range(n_sweeps):
        if const_temp >= 0.0:
            if warm_up:
                temperature = temperature_function(
                    n,
                    const_temp,
                    Lengths[n]
                )
            else:
                temperature = const_temp
        else:
            temperature = temperature_function(
                n,
                const_temp,
                Lengths[n]
            )

        #survivors, survivor_lengths = choose_survivors_ids(
        #    Tour_ID_Matrix,
        #    Lengths[n].copy(),
        #    population_size,
        #    N_cities
        #)
        #Tour_ID_Matrix, current_lengths = mate_ids(
                #    survivors,
                #    N_cities,
                #    D
                #)
        survivors , survivor_lengths , surviving_ids, dead_ids = choose_survivors_ids2(
            Tour_ID_Matrix,
            Lengths[n].copy(),
            population_size,
            N_cities
            )
        
        
        Tour_ID_Matrix, current_lengths = mate_ids2(
                    survivors,
                    N_cities,
                    D,
                    surviving_ids,
                    dead_ids
                )
        for p in nb.prange(population_size):
            current_L = current_lengths[p]
            accepted_moves = 0

            for _ in range(mutations_per_sweep):
                r1 = np.random.randint(0, N_cities)
                r2 = np.random.randint(0, N_cities)

                while r2 == r1:
                    r2 = np.random.randint(0, N_cities)

                a = min(r1, r2)
                b = max(r1, r2)

                if a == 0 and b == N_cities - 1:
                    continue

                id_a_minus_one = Tour_ID_Matrix[p, (a - 1) % N_cities]
                id_a = Tour_ID_Matrix[p, a]
                id_b_plus_one = Tour_ID_Matrix[p, (b + 1) % N_cities]
                id_b = Tour_ID_Matrix[p, b]

                dL = (
                    D[id_a_minus_one, id_b]
                    - D[id_a_minus_one, id_a]
                    + D[id_b_plus_one, id_a]
                    - D[id_b_plus_one, id_b]
                )

                accept = dL <= 0.0
                if not accept and temperature > 0.0:
                    accept = (
                        np.random.random()
                        < np.exp(-dL / temperature)
                    )

                if accept:
                    current_L += dL
                    Tour_ID_Matrix[p] = reverse_subsequence(
                        Tour_ID_Matrix[p],
                        a,
                        b
                    )
                    accepted_moves += 1

            current_lengths[p] = current_L
            accepted_population[p] = accepted_moves
            acceptance_rate[n,p] = (
                accepted_population[p]
                / (mutations_per_sweep)
            )

        Lengths[n + 1] = current_lengths
        temperatures[n] = temperature

    if n_sweeps != 0:
        temperatures[-1] = temperatures[-2]

    return (
        Tour_ID_Matrix,
        Lengths,
        temperatures,
        acceptance_rate,
    )

@nb.njit(parallel=True, cache=True)
def mixed_annealing_D_const_T_2(
        Tour_ID_Matrix,
        D,
        population_size,
        temperature_function,
        Lengths_0,
        n_sweeps,
        mutations_per_sweep,
        mutations_interval,
        const_temp,
        warm_up,
        detailed,
        seed
):
    N_cities = Tour_ID_Matrix.shape[1]

    Lengths = np.empty(
        (n_sweeps + 1, population_size),
        dtype=np.float32
    )
    Lengths[0] = Lengths_0

    temperatures = np.empty(n_sweeps + 1, dtype=np.float32)
    acceptance_rate = np.empty((n_sweeps,population_size), dtype=np.float64)
    accepted_population = np.empty(population_size, dtype=np.int64)

    for n in range(n_sweeps):
        if const_temp >= 0.0:
            if warm_up:
                temperature = temperature_function(
                    n,
                    const_temp,
                    Lengths[n]
                )
            else:
                temperature = const_temp
        else:
            temperature = temperature_function(
                n,
                const_temp,
                Lengths[n]
            )

        #survivors, survivor_lengths = choose_survivors_ids(
        #    Tour_ID_Matrix,
        #    Lengths[n].copy(),
        #    population_size,
        #    N_cities
        #)
        #Tour_ID_Matrix, current_lengths = mate_ids(
                #    survivors,
                #    N_cities,
                #    D
                #)
        np.random.seed(seed + n)
        survivors , survivor_lengths , surviving_ids, dead_ids = choose_survivors_ids2(
            Tour_ID_Matrix,
            Lengths[n].copy(),
            population_size,
            N_cities
            )
        
        
        Tour_ID_Matrix, current_lengths, _ = mate_ids2(
                    survivors,
                    N_cities,
                    D,
                    surviving_ids,
                    dead_ids
                )
        if n % mutations_interval == 0:
            for p in nb.prange(population_size):
                np.random.seed(
                                    seed
                                    + 1_000_000
                                    + n * population_size
                                    + p
                                )
                current_L = current_lengths[p]
                accepted_moves = 0

                for _ in range(mutations_per_sweep):
                    r1 = np.random.randint(0, N_cities)
                    r2 = np.random.randint(0, N_cities)

                    while r2 == r1:
                        r2 = np.random.randint(0, N_cities)

                    a = min(r1, r2)
                    b = max(r1, r2)

                    if a == 0 and b == N_cities - 1:
                        continue

                    id_a_minus_one = Tour_ID_Matrix[p, (a - 1) % N_cities]
                    id_a = Tour_ID_Matrix[p, a]
                    id_b_plus_one = Tour_ID_Matrix[p, (b + 1) % N_cities]
                    id_b = Tour_ID_Matrix[p, b]

                    dL = (
                        D[id_a_minus_one, id_b]
                        - D[id_a_minus_one, id_a]
                        + D[id_b_plus_one, id_a]
                        - D[id_b_plus_one, id_b]
                    )

                    accept = dL <= 0.0
                    if not accept and temperature > 0.0:
                        accept = (
                            np.random.random()
                            < np.exp(-dL / temperature)
                        )

                    if accept:
                        current_L += dL
                        Tour_ID_Matrix[p] = reverse_subsequence(
                            Tour_ID_Matrix[p],
                            a,
                            b
                        )
                        accepted_moves += 1

                current_lengths[p] = current_L
                accepted_population[p] = accepted_moves
                acceptance_rate[n,p] = (
                    accepted_population[p]
                    / (mutations_per_sweep)
                )

        Lengths[n + 1] = current_lengths
        temperatures[n] = temperature

    if n_sweeps != 0:
        temperatures[-1] = temperatures[-2]

    return (
        Tour_ID_Matrix,
        Lengths,
        temperatures,
        acceptance_rate,
    )
    
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

@nb.njit(cache=True)
def construct_D_matrix_dtype(tour , distance_metric,dtype = np.float64):
    N_cities = len(tour)
    D_matrix = np.empty((N_cities,N_cities),dtype=dtype)
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
    Tour_ID_Matrix = np.empty((POPULATION_SIZE,N_cities),dtype=np.int32)
    for n_p in range(POPULATION_SIZE):
        Tour_ID_Matrix[n_p] = tour_ids
        Lengths[n_p] = starting_length
    print("Input dtype:", Tour_ID_Matrix.dtype)
    if WARM_UP_NUMBA:
        seed_numba(BASE_RANDOM_SEED)
        mixed_annealing_D(
            Tour_ID_Matrix,
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
            Tour_ID_Matrix,
            distance_matrix,
            POPULATION_SIZE,
            temp_func,
            Lengths,
            NUMBER_OF_SWEEPS,
        )
        
        Tour_ID_Matrix = final_population.copy()
        print("Input dtype:", Tour_ID_Matrix.dtype)
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


@nb.njit(parallel=True)
def mixed_annealing_D_after(
        Tour_ID_Matrix,
        D,
        population_size,
        temperature_function,
        Lengths_0,
        n_sweeps
):
    N_cities = Tour_ID_Matrix.shape[1]
    N_cities_sq = N_cities**2
    

    
    
    Lengths = np.empty((n_sweeps+1,population_size),dtype=np.float32)

    for i in range(population_size):
        Lengths[0][i] = Lengths_0[i]

    temperatures = np.empty(n_sweeps+1,dtype=np.float32)
    
    for n in range(n_sweeps):
        temperature = temperature_function(
            n,
            Tour_ID_Matrix,
            Lengths[n,0]
            )
        
        current_lengths = Lengths[n].copy()
        for p in nb.prange(population_size):
            current_L = current_lengths[p]
            for _ in range(N_cities_sq):
            
                Tour_ID_Matrix[p], current_L =annealing_step_Dmatrix(
                    Tour_ID_Matrix[p],
                    D,
                    temperature,
                    N_cities,
                    current_L
                    )
            current_lengths[p] = current_L
        survivors, survivor_lengths = choose_survivors_ids(
            Tour_ID_Matrix,
            current_lengths,
            population_size,
            N_cities
            )
        Tour_ID_Matrix, new_generation_length = mate_ids(
            survivors,
            N_cities,
            D
            )
        Lengths[n+1] = new_generation_length
        temperatures[n] = temperature
    if n_sweeps != 0:
        temperatures[-1] = temperatures[-2]
    
    return Tour_ID_Matrix,Lengths, temperatures


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