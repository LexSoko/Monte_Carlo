import matplotlib.pyplot as plt
import pandas as pd
import os 
import random as rd
import numpy as np
from scipy.linalg import norm
import matplotlib.style as mplstyle
import matplotlib as mpl
from tqdm import tqdm
import time
__author__ = "Aleksey Sokolov"
#12004091

pathdata = os.getcwd() + "\\data\\"
pathgraphics = os.getcwd() + "\\graphics\\"
def change_pos(array, index1, index2):
    temparray = array
    if index1 > index2:
        tempindex = index1
        index1 = index2
        index2 = tempindex
    subarray = temparray[index1:index2+1]
    subarray = subarray[::-1]
    temparray[index1:index2+1] = subarray
    return temparray

def calculate_energy1(array):
    shifted_array = np.roll(array,-1)
    delta = shifted_array - array
    energy = 0
    for d in delta:
        energy += norm(d)
    return energy

def calculate_energy(array):
    
    
    shifted_array = np.roll(array, -1, axis=0)
    delta = shifted_array - array
    lenght = np.sum(np.sqrt(np.sum(delta**2, axis=1)))  # Vectorized computation
    
    return lenght
def create_cities(N):
    city_pos = []
    city_dist = np.zeros((N,N))
    for i in range(N):
        x = rd.random()
        y = rd.random()
        city_pos.append(np.array([x,y]))
    return city_pos
def make_loop(new_path):
    new_path = np.concatenate([new_path, [new_path[0]]],axis=0)
    
    x = new_path.T[0]
    y = new_path.T[1]
    return x,y
def plot_GUI(ax_ts,current_array,all_energies_k, all_energies_avg, all_energies_var):
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 1.0
    ax_ts[0,0].set_aspect("equal")
    ax_ts[0,0].plot(current_array.T[0],current_array.T[1], "b-")
    ax_ts[0,0].plot([current_array.T[0][-1],current_array.T[0][0]],[current_array.T[1][-1],current_array.T[1][0]], "b-")
    ax_ts[0,0].plot(current_array.T[0],current_array.T[1], "ro")


    ax_ts[0,1].plot(all_energies_k[1:], label = f"Best Energy = {all_energies_k[-1]:.3f}")

    ax_ts[1,0].plot(all_energies_avg[1:], label = "$\\bar{E}_k = \\frac{1}{L} \sum_j E_{j,k}$  (L = n_{steps})\n $\\bar{E}_k$ = " + f"{all_energies_avg[-1]:.3f} $\pm$ {np.sqrt(all_energies_var[-1]):.3f}")
    ax_ts[1,0].fill_between(np.arange(0,len(all_energies_avg[1:]),1),all_energies_avg[1:] - np.sqrt(all_energies_var[1:]),all_energies_avg[1:] + np.sqrt(all_energies_var[1:]),alpha = 0.2)

    ax_ts[1,1].plot(all_energies_var[1:], label = "$(\Delta E)^2$ = " + f"{all_energies_var[-1]:.6f}")

    ax_ts[0,1].set_xlabel("$k$")
    ax_ts[0,1].set_ylabel("Energy / arb.U.")

    ax_ts[1,0].set_xlabel("$k$")
    ax_ts[1,0].set_ylabel("Avg Energy / arb.U.")

    ax_ts[1,1].set_ylabel("Variance / arb.U ")
    ax_ts[1,1].set_xlabel("$k$")

    ax_ts[1,0].legend()
    ax_ts[0,1].legend(loc = "upper right")
    ax_ts[1,1].legend()

def find_best_route(array,Tstart,q,startEnergy,nsteps, figname = "lastfirst", plot = True):
    all_energies = [startEnergy]
    all_energies_k = []
    all_energies_avg = []
    all_energies_var = []
    cities_temp = [array]
    current_E = startEnergy
    current_array = np.copy(array)
    current_T = Tstart
    same = 0
    k = 1
    n = 1
    
    fig_ts , ax_ts = plt.subplots(2,2,figsize=(16,16))
    fig_ts.suptitle(f" $q$ = {q} ," + "$T_{start}$ = " +f"{Tstart} ," + "$n_{steps}$ = " +f"{nsteps}")
    mplstyle.use('fast')
    notconverged = 0
    while notconverged<3:
        a = np.random.randint(0,len(array))
        b = np.random.randint(0,len(array))
        if a > b:
            tempindex = a
            a = b
            b = tempindex
        if a != b:
            if ((a,b) != (0,len(array)-1)):
                e_Kprime = norm(current_array[(a-1)%len(current_array)] -current_array[(b)%len(array)]) + norm(current_array[(a)%len(array)] -current_array[(b+1)%len(array)]) 
                e_K = (norm(current_array[(a-1)%len(current_array)]- current_array[(a)%len(array)]) + norm(current_array[(b)%len(array)]- current_array[(b+1)%len(array)]))
                dE = e_Kprime - e_K   
                
            else:
                dE = 0.0 
        else:
            dE = 0.0

        if np.exp(-dE/current_T) > np.random.random():
            if dE != 0.0:
                current_array = change_pos(current_array,a,b)
            current_E += dE
            all_energies.append(current_E)
            
            
        else:
            all_energies.append(current_E)
            same += 1
        if n%nsteps == 0:
            minimum_Energy = np.min(all_energies[-len(array)**2:-1])
            average_Energy = np.mean(all_energies[-len(array)**2:-1])
            variance_Energy = np.var(all_energies[-len(array)**2:-1])
           
            all_energies_k.append(minimum_Energy)
            all_energies_avg.append(average_Energy)
            all_energies_var.append(variance_Energy)
            cities_temp.append(np.copy(current_array))
    
            if k > 1:
                if len(current_array) -np.sum(np.all(cities_temp[-1]== cities_temp[-2], axis=1)) <= 1:
                    notconverged +=1
                if n>20*len(current_array)*nsteps:
                    notconverged +=1
                
            if plot:
                plot_GUI(ax_ts,current_array,all_energies_k, all_energies_avg, all_energies_var)
                ax_ts[1,1].text(0.9,0.9,"$T_k = T_{start} k^{-q}$ = "+ f"{current_T:.3f}")
                plt.show(block=False)
                plt.pause(0.001)
                if notconverged < 3:
                    ax_ts[0, 0].cla()
                    ax_ts[0, 1].cla()
                    ax_ts[1, 0].cla()
                    ax_ts[1, 1].cla()
            k += 1
            current_T = Tstart*(k**(-q))
       
        n += 1
    if plot != True:
        plot_GUI(ax_ts,current_array,all_energies_k, all_energies_avg, all_energies_var)
    
    fig_ts.savefig(f"{figname}.pdf")
    fig_ts.clf()
        

    return current_array, all_energies

def find_best_route_self_stop(array,Tstart,q,startEnergy,nsteps):
    start = time.time()
    all_energies = [startEnergy]
    all_energies_k = []
    all_energies_avg = []
    all_energies_var = []
    cities_temp = [array]
    current_E = startEnergy
    current_array = np.copy(array)
    current_T = Tstart
    same = 0
    k = 1
    n = 1

    notconverged = 0
    while notconverged<3:
        a = np.random.randint(0,len(array))
        b = np.random.randint(0,len(array))
        if a > b:
            tempindex = a
            a = b
            b = tempindex
        if a != b:
            if ((a,b) != (0,len(array)-1)):
                e_Kprime = norm(current_array[(a-1)%len(current_array)] -current_array[(b)%len(array)]) + norm(current_array[(a)%len(array)] -current_array[(b+1)%len(array)]) 
                e_K = (norm(current_array[(a-1)%len(current_array)]- current_array[(a)%len(array)]) + norm(current_array[(b)%len(array)]- current_array[(b+1)%len(array)]))
                dE = e_Kprime - e_K   
                
            else:
                dE = 0.0 
        else:
            dE = 0.0

        if np.exp(-dE/current_T) > np.random.random():
            if dE != 0.0:
                current_array = change_pos(current_array,a,b)
            current_E += dE
            all_energies.append(current_E)
            
            
        else:
            all_energies.append(current_E)
            same += 1
        if n%nsteps == 0:
            minimum_Energy = np.min(all_energies[-len(array)**2:-1])
            average_Energy = np.mean(all_energies[-len(array)**2:-1])
            variance_Energy = np.var(all_energies[-len(array)**2:-1])
           
            all_energies_k.append(minimum_Energy)
            all_energies_avg.append(average_Energy)
            all_energies_var.append(variance_Energy)
            cities_temp.append(np.copy(current_array))
    
            if k > 1:
                if len(current_array) -np.sum(np.all(cities_temp[-1]== cities_temp[-2], axis=1)) <= 1:
                    notconverged +=1
                if n>20*len(current_array)*nsteps:
                    notconverged +=1

            k += 1
            current_T = Tstart*(k**(-q))
       
        n += 1
    
    end = time.time()
    print(f"elapsed time = {end-start} ")
        

    return current_array, all_energies

if __name__ == '__main__':
    print(pathdata+"ch150.csv")
    cities = np.array(create_cities(50))
    cities = pd.read_csv(pathdata+"ch150.csv", delimiter=";")
    #cities = pd.read_csv("bestpath7800.csv", delimiter=";")
    fig1, ax1  = plt.subplots(1,1)
    ax1.plot(   cities["x"],cities["y"])
    ax1.scatter(cities["x"],cities["y"], marker="+", c="r")
    plt.show()
    Energy = calculate_energy(np.array(cities))
    print(Energy)
    new_path, en = find_best_route(
        cities,
        10,
        0.1,
        Energy,
        len(cities)**2,
        plot=True
        )
    print("************energy+++++++++  =",en[-1])
   
    x,y = make_loop(new_path)

    fig, ax  = plt.subplots(1,1)

    ax.plot(x, y)
    ax.scatter(x,y, marker="+", c="r")
    plt.show()
    pass