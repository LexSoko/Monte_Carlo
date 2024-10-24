import matplotlib.pyplot as plt
import pandas as pd
import os 
import random as rd
import numpy as np
from scipy.linalg import norm
from numba import njit
import time
import modules.uselful as us
__author__ = "Aleksey Sokolov"
#12004091

pathdata = os.getcwd() + "\\data\\"
pathgraphics = os.getcwd() + "\\graphics\\"
def change_pos(array, index1, index2):
    temparray = np.copy(array)
    if index1 > index2:
        tempindex = index1
        index1 = index2
        index2 = tempindex
    subarray = temparray[index1:index2+1]

    subarray = subarray[::-1]
   
    temparray[index1:index2+1] = subarray
    
    return temparray
a = [1,2,3,4,5,6,7,8,9]

print(change_pos(a,1,5),change_pos(a,1,5))
def calculate_energy(array):
    shifted_array = np.roll(array,-1)
    delta = shifted_array - array
    energy = 0
    for d in delta:
        energy += norm(d)
    return energy

def create_cities(N):
    city_pos = []
    for i in range(N):
        x = rd.random()
        y = rd.random()
        city_pos.append(np.array([x,y]))
    return city_pos

def find_best_route_manual(
        cities_pos1,
        t_func = lambda x: x,   
        M_sweeps = 1,
        N_sweeps = 10,
        ):
    
    start = time.time()
    cities_pos = np.array(cities_pos1)
    all_energies = np.zeros((M_sweeps,N_sweeps))
    current_E = calculate_energy(cities_pos)
    all_energies[0,0] = np.copy(current_E)

    all_energies_avg = np.zeros(M_sweeps)
    all_energies_var = np.zeros(M_sweeps)
    best_path = []
    t= 0
    for m_s in range(0,M_sweeps):
        current_temp = t_func(m_s)
        for n_s in range(0,N_sweeps):
            a = np.random.randint(0,len(cities_pos))
            b = np.random.randint(0,len(cities_pos))

            if a > b:
                tempindex = a
                a = b
                b = tempindex      

            if a != b:
                if ((a,b) != (0,len(cities_pos)-1)):
                    e_Kprime = norm(cities_pos[(a-1)%len(cities_pos)] -cities_pos[(b)%len(cities_pos)]) + norm(cities_pos[(a)%len(cities_pos)] -cities_pos[(b+1)%len(cities_pos)]) 
                    e_K = (norm(cities_pos[(a-1)%len(cities_pos)]- cities_pos[(a)%len(cities_pos)]) + norm(cities_pos[(b)%len(cities_pos)]- cities_pos[(b+1)%len(cities_pos)]))
                    dE = e_Kprime - e_K  
                else:
                    dE = 0.0 
            else:
                dE = 0.0

            if np.exp(-dE/current_temp) > np.random.random():
                if dE != 0.0:
                    cities_pos = change_pos(cities_pos,a,b)
                current_E += dE
                all_energies[m_s,n_s] =current_E
                if m_s > 1 and n_s > 1:
                    if current_E < np.min(all_energies[0:m_s][0:n_s]):
                        best_path = np.copy(cities_pos)
            else:
                all_energies[m_s,n_s] =current_E
            
        all_energies_avg[m_s] = np.mean(all_energies[m_s])
        all_energies_var[m_s] = np.var(all_energies[m_s])
        
        
        print(f"{100*m_s/M_sweeps} %")
    end = time.time()        
    print(f"elapsed time = {end-start} ")
    return [cities_pos,best_path], all_energies, all_energies_avg, all_energies_var

def make_loop(new_path):
    new_path = np.concatenate([new_path, [new_path[0]]],axis=0)
    
    x = new_path.T[0]
    y = new_path.T[1]
    return x,y
def exp_temp(x,A,k,w,C):
    return A*np.exp(-k*x)*np.abs(np.cos(w*x)+C) 
def polynomial(m,T,q):
    return T*((m+1)**(-q))
def normal_dist(x,A,x0,sig):
    return A*np.exp(-((x-x0)**2)/sig)


if __name__ == '__main__':
   
    cities = pd.read_csv(pathdata+"ch150.csv", delimiter=";")
    #cities = np.loadtxt(pathdata + "U4 city-positions.txt", delimiter=";")
    temperature_func_1 = lambda x: exp_temp(x,10,0.5,np.pi/6,2) + normal_dist(x,10,60,0.1) + normal_dist(x,2,90,0.1)
    paths, all_energies , all_energies_avg, all_energies_var = find_best_route_manual(
        cities_pos1=cities,
        t_func= temperature_func_1,
        M_sweeps=150,
        N_sweeps=len(cities)**2    
    )
    new_path = paths[0]
    best_path = paths[1]
    best_path_energy = calculate_energy(best_path)
    print(f"Last Energy reached = {all_energies_avg[-1]} +- {all_energies_var[-1]} ")
    print(f" Best Path Energy = {best_path_energy}")
    all = np.concatenate(all_energies)
    x,y = make_loop(new_path)
    x1, y1 = make_loop(best_path)
    fig, ax  = plt.subplots(2,2,figsize =( 15,15))
    #plt.title("ch150 Dataset")
    ax[0,0].plot(x, y, label = "Last Path")
    ax[0,0].scatter(x,y, marker="+", c="r")
    ax[0,0].legend()
    ax[0,1].plot(x1, y1, label = "Minimum Path")
    ax[0,1].scatter(x1,y1, marker="+", c="g")
    ax[0,1].legend()
    us.un_plot(fig,ax[1,0],np.arange(0,len(all_energies_avg)), us.un_merger(all_energies_avg,np.sqrt(all_energies_var)),
               x_label="m_s",y_label="lenght",
               format="b-",
               show_error="fill_between")
    ax[1,0].plot(len(all_energies_avg),all_energies_avg[-1], "r+", label = f" Last Energy = {all_energies_avg[-1]:.2f} +- {all_energies_var[-1]:.2f}")
    ax[1,0].plot(np.argmin(all_energies_avg),all_energies_avg[np.argmin(all_energies_avg)], "g+", label = f" Minimum Energy = {all_energies_avg[np.argmin(all_energies_avg)]:.2f} +- {all_energies_var[-1]:.2f}")
    ax[1,0].legend()
    m_s = np.arange(0,len(all_energies_avg))
    ax[1,1].plot(m_s,temperature_func_1(m_s), "g-", label = "Temperature Function")
    ax[1,1].legend()
    fig.savefig("ch150_City_Pos_two_heating2.pdf")
    #ax[1].plot(all_energies_var)
    plt.show()
    pass

























#def main(a = False, b =False, c = False, d=False):
#    
#    if a:
#    
#        data = np.loadtxt(pathdata + "U4 city-positions.txt", delimiter=";")
#        
#        enegy = calculate_energy(data)
#
#        new_path,en = find_best_route(data,1,1,enegy,len(data)**2, figname=pathgraphics+"Tstart_1_q_1_U4")
#        print(f"best Energy reauched: {en[-1]}" )
#        new_new_path , ne_en = find_best_route(np.copy(new_path),0.5,1,en[-1],len(new_path)**2,figname=pathgraphics+"secondrun_Tstart_05_q_1")
#        print(f"best Energy reauched: {ne_en[-1]}" )
#    if b:
#        data2 = np.array(pd.read_csv(pathdata+ "albert3.csv", delimiter=","))
#        random.shuffle(data2)
#        data = data2
#
#        enegy = calculate_energy(data)
#
#        new_path,en = find_best_route(data,1,1,enegy,len(data)**2, figname=pathgraphics+"albert_Tstart_1_q_1_U4")
#        print(f"best Energy reached first run: {en[-1]}" )
#        new_new_path , ne_en = find_best_route(np.copy(new_path),0.8,1,en[-1],len(new_path)**2,figname=pathgraphics+"albert_secondrun_Tstart_08_q_1")
#        print(f"best Energy reached second run: {ne_en[-1]}" )
#        new_new_new_path , new_en = find_best_route(np.copy(new_new_path),0.3,2,ne_en[-1],len(new_path)**2,figname=pathgraphics+"albert_thirdrun_Tstart_03_q_2")
#        print(f"best Energy reached third run: {new_en[-1]}" )
#
#    if c: 
#        
#        data = np.loadtxt(pathdata + "U4 city-positions.txt", delimiter=";")
#        data2 = data
#        enegy = calculate_energy(data2)
#       
#        nsteps = [len(data), len(data)**2, 2*len(data)**2]
#        qsteps = [0.3,0.5,1,3]
#        for n in tqdm(nsteps, desc= "nsteps loop"):
#            for q in  tqdm(qsteps,desc= "q loop"):
#                new_new_path ,en = find_best_route(data,1,q,enegy,n, figname=pathgraphics+f"Different_configs_nsteps_{n}_q_{q}",plot=False)