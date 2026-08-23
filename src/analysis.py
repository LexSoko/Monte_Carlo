import numpy as np
import src.modules.TSP as tsp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numba as nb
datach150 = np.loadtxt("data/ch150.csv", delimiter=";",skiprows=1)


def plot_fromITSP():
    data = tsp.make_loop(np.loadtxt("results/mixed_path_1.csv", delimiter=";")).T
    data2 = tsp.make_loop(np.loadtxt("results/mixed_path_2.csv", delimiter=";")).T
    tsp.generate_plot(np.array([data,data2]),tsp.calculate_lenghts([data.T,data2.T]),save=True)

@nb.njit
def temp_func(x):
    return 10*np.exp(-0.0005*x)*np.cos(0.001*x)**2

#Energy_avg = []
#Energy_var = []
#Heat_cap = []
#energy = np.array([tsp.calculate_lenght(np.array([datach150.T]))])
#print(energy)
#print("fuck")
#city_pos = np.array([datach150])
#
#for i in tqdm(range(10000)):
#    energy_inter = []
#    temp = temp_func(i,0.5,10)
#    #print(datach150.shape[0])
#    #print(city_pos)
#    for j in range(datach150.shape[0]**2):
#        
#        city_pos , energy = tsp.mutation(city_pos,temp,energy)
#        energy_inter.append(energy[0])
#    Energy_avg.append(np.mean(energy_inter))
#    Energy_var.append(np.var(energy_inter))
#    Heat_cap.append(np.var(energy_inter)/(temp**2))

energy = tsp.calculate_lenght(datach150)
city_pos ,Energy_avg , Energy_var , Heat_cap = tsp.run_annealing_fixedN(10000,datach150,energy, temp_func)
plt.plot(temp_func(np.arange(0,10000)))
plt.show()
plt.scatter(city_pos.T[0],city_pos.T[1])
plt.plot(city_pos.T[0],city_pos.T[1])
plt.show()
plt.plot(Energy_avg)
plt.show()
plt.plot(Energy_var)
plt.show()
plt.plot(Heat_cap)
plt.show()