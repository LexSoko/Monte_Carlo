import matplotlib.pyplot as plt
import pandas as pd
import os 
import random as rd
import numpy as np
from scipy.linalg import norm
import numba as nb
import time


def calculate_lenght(arrays):
    """
    calculates total lenght of a poplulation of city arrays

    Args:
        arrays (np.ndarray): cities array

    Returns:
        np.ndarray: lenghts for each city array
    """
    lenghts = np.empty(len(arrays))
    for i, array in enumerate(arrays):
        shifted_array = np.roll(array, -1, axis=0)
        delta = shifted_array - array
        lenght = np.sum(np.sqrt(np.sum(delta**2, axis=1)))  # Vectorized computation
        lenghts[i] = lenght
    return lenghts

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

@nb.njit
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
        a = np.random.randint(0,len(cities_pos_pop[0]))
        b = np.random.randint(0,len(cities_pos_pop[0]))
        if a > b:
            tempindex = a
            a = b
            b = tempindex  
        
        if a != b:
            if ((a,b) != (0,len(city_indv)-1)):
                e_Kprime = np.sqrt(np.sum((city_indv[(a-1)%len(city_indv)] -city_indv[(b)%len(city_indv)])**2)) + np.sqrt(np.sum((city_indv[(a)%len(city_indv)] -city_indv[(b+1)%len(city_indv)])**2)) 
                e_K = np.sqrt(np.sum(((city_indv[(a-1)%len(city_indv)]- city_indv[(a)%len(city_indv)])**2)) + np.sqrt(np.sum((city_indv[(b)%len(city_indv)]- city_indv[(b+1)%len(city_indv)]))**2))
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


@nb.njit
def choose_survivors(old_generation,lenghts):
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
    mid = len(old_generation)//2
    indeces = np.arange(0,len(old_generation))
    np.random.shuffle(indeces)
    old_generation = old_generation[indeces]
    lenghts = lenghts[indeces]
    
    survivors = np.empty((mid,len(old_generation[0]),len(old_generation[0][0])))
    for i in range(mid):
        if lenghts[i] < lenghts[i + mid]:
            survivors[i] = old_generation[i]
        else:
            survivors[i] = old_generation[i +mid]
   
    return survivors

@nb.njit
def mate(survivors:np.ndarray):
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
    offspring = np.empty((int(len(survivors)*2), len(survivors[0]), len(survivors[0][0])))
    offspring[0:len(survivors),:,:] = survivors
    
    indices = np.arange(len(survivors), dtype=np.int32)  
    np.random.shuffle(indices)  
    pairs = np.zeros((len(survivors), 2), dtype=np.int32)

    for i in range(0, len(survivors) - 1, 2):
        pairs[i] = (indices[i], indices[i + 1])
        pairs[i + 1] = (indices[i + 1], indices[i])

    for n, (i , j) in enumerate(pairs):
        a = np.random.randint(0,len(survivors[i]) - 1)
        b = np.random.randint(a,len(survivors[i]))
       
        sub_path_i = list(survivors[i][a:b])
        remaining_path_j = np.empty((len(survivors[i]) - len(sub_path_i), survivors[j].shape[1]))
        
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
        
        for k in range(0, len(survivors[i])):
            if a <= k < b:
                offspring[n +len(survivors),k,:] = sub_path_i.pop(0)
                
            else:
                offspring[n+len(survivors),k,:] = remaining_path_j.pop(0)

    return offspring

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
        all_cities_lenghts = calculate_lenght(all_cities_specimen)
    time2 = time.time()
    print(f"calculation took : {time2 - time1 } s")
    return all_cities_specimen, all_cities_lenghts

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
    all_cities_lenghts = calculate_lenght(all_cities_specimen)

    return all_cities_specimen, all_cities_lenghts

def generate_plot(all_cities_specimen,all_cities_lenghts, save = True,filename = "citypos"):
    fig2, ax2  = plt.subplots(len(all_cities_lenghts)//2,2, figsize = (12,8))
    ax2 = ax2.flatten()
    for i in range(len(all_cities_specimen)):
        ax2[i].plot(all_cities_specimen[i].T[0],all_cities_specimen[i].T[1], label = f"lenght = {all_cities_lenghts[i]}")
        ax2[i].scatter(all_cities_specimen[i].T[0],all_cities_specimen[i].T[1],marker="+", c="r")
        ax2[i].legend()
    print(all_cities_lenghts)
    if save == True:
        fig2.savefig(f"{filename}{np.random.randint(0,100)}.pdf")
    plt.show()




