import matplotlib.pyplot as plt
import pandas as pd
import os 
import random as rd
import numpy as np
from scipy.linalg import norm
import numba as nb
import time


def calculate_lenght(arrays):
    lenghts = np.empty(len(arrays))
    for i, array in enumerate(arrays):
        shifted_array = np.roll(array, -1, axis=0)
        delta = shifted_array - array
        lenght = np.sum(np.sqrt(np.sum(delta**2, axis=1)))  # Vectorized computation
        lenghts[i] = lenght
    return lenghts



@nb.njit
def mutation(
    cities_pos_pop: np.ndarray,
    temperature: np.float64,
    start_lenghts: np.ndarray
    ):
    
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
def choose_survivors(old_generation,lenghts):
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
def mate(survivors_a:np.ndarray):
    offspring = np.empty((int(len(survivors_a)*2), len(survivors_a[0]), len(survivors_a[0][0])))
    offspring[0:len(survivors_a),:,:] = survivors_a
    
    indices = np.arange(len(survivors_a), dtype=np.int32)  
    np.random.shuffle(indices)  
    pairs = np.zeros((len(survivors_a), 2), dtype=np.int32)

    for i in range(0, len(survivors_a) - 1, 2):
        pairs[i] = (indices[i], indices[i + 1])
        pairs[i + 1] = (indices[i + 1], indices[i])

    for n, (i , j) in enumerate(pairs):
        a = np.random.randint(0,len(survivors_a[i]) - 1)
        b = np.random.randint(a,len(survivors_a[i]))
       
        sub_path_i = list(survivors_a[i][a:b])
        remaining_path_j = np.empty((len(survivors_a[i]) - len(sub_path_i), survivors_a[j].shape[1]))
        
        count = 0
        
        for item in survivors_a[j]:
            found = False
            for sub_item in sub_path_i:
                if np.all(item == sub_item):
                    
                    found = True
                    break
            if not found:
                remaining_path_j[count] = item
                count += 1
            
        remaining_path_j = list(remaining_path_j)
        
        for k in range(0, len(survivors_a[i])):
            if a <= k < b:
                offspring[n +len(survivors_a),k,:] = sub_path_i.pop(0)
                
            else:
                offspring[n+len(survivors_a),k,:] = remaining_path_j.pop(0)

    return offspring

def create_diversity(cities,n = 2):
    population = []
    for i in range(n):
        shuffled_arr = cities.copy()
        np.random.shuffle(shuffled_arr)
        population.append(list(shuffled_arr))
    return np.array(population)
    

def run_mixed(N,temp_func):
    time1 = time.time()
    for k in range(N):
        temp = temp_func(k)
        survivors = choose_survivors(all_cities_specimen,all_cities_lenghts)
        all_cities_specimen = mate(survivors)
        for n in range(all_cities_specimen.shape[1]):        
            all_cities_specimen , all_cities_lenghts = mutation(all_cities_specimen,temp,all_cities_lenghts)
        all_cities_lenghts = calculate_lenght(all_cities_specimen)
    time2 = time.time()
    print(f"calculation took : {time2 - time1 } s")
    return all_cities_specimen, all_cities_lenghts


def generate_plot(all_cities_specimen,all_cities_lenghts):
    fig2, ax2  = plt.subplots(4,2, figsize = (16,12))
    ax2 = ax2.flatten()
    for i in range(len(all_cities_specimen)):
        ax2[i].plot(all_cities_specimen[i].T[0],all_cities_specimen[i].T[1], label = f"lenght = {all_cities_lenghts[i]}")
        ax2[i].scatter(all_cities_specimen[i].T[0],all_cities_specimen[i].T[1],marker="+", c="r")
        ax2[i].legend()
    print(all_cities_lenghts)

    fig2.savefig("genetic_annealing_combined.pdf")
    plt.show()




pathdata = os.getcwd() + "\\data\\"
pathgraphics = os.getcwd() + "\\graphics\\"
N = 100000
cities = pd.read_csv(pathdata+"bier127.csv", delimiter=";")


cities1 = np.array([[x,y] for x,y in zip(cities["x"],cities["y"]) ])
#cities1 = cities1[0:10]

#all_cities_specimen = np.array([cities1 for n in range(8)])
all_cities_specimen = create_diversity(cities1,8)
all_cities_lenghts = calculate_lenght(all_cities_specimen)

Tstart = 10
q = 0.1
time1 = time.time()
for k in range(N):
    #print(f"################## k = {k} ######################")
    temp = Tstart*((k+1)**(-q))
    survivors = choose_survivors(all_cities_specimen,all_cities_lenghts)
    all_cities_specimen = mate(survivors)
    for n in range(all_cities_specimen.shape[1]):        
        all_cities_specimen , all_cities_lenghts = mutation(all_cities_specimen,temp,all_cities_lenghts)

    all_cities_lenghts = calculate_lenght(all_cities_specimen)
time2 = time.time()
print(f"calculation took : {time2 - time1 } s")


temp = lambda k,Tstart,q: Tstart*((k+1)**(-q))

generate_plot(all_cities_specimen,all_cities_lenghts)