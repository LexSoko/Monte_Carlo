import numpy as np 
import matplotlib as plt
import os 

mixed_path = os.path.join(
    "..",
    "results",
    "tsp_benchmark",
    "ch150_ns600_pop16_report_annealing_ch150",
    "mixed"
)

lenght_history = np.load(
    os.path.join(
        mixed_path,
        "run_000",
        "length_history.npy"
    )

)

annealing_path = os.path.join(
    "..",
    "results",
    "tsp_benchmark",
    "ch150_ns600_pop16_report_annealing_ch150",
    "annealing"
)
annealing_data = np.empty((16,lenght_history.shape[0]))

for i in range(16):
    annealing_data[i] = np.load(
        os.path.join(
            annealing_path,
            f"run_0{i:02d}",
            "length_history.npy"
        )
    )



l_T = lenght_history.T
print(f"min per population member")
mins = np.empty(lenght_history.shape[1])
min_indecces_mixed = np.empty(lenght_history.shape[1])

for p, m in enumerate(l_T):
    minimal_idx = np.argmin(m)
    min_indecces_mixed[p] = minimal_idx
    mins[p] = m[minimal_idx]
    print(f"pop {p}: {m[minimal_idx]} reached at sweep: {minimal_idx}")

sweep_index_minimal_l = int(min_indecces_mixed[np.argmin(mins)])
print(sweep_index_minimal_l)
minimum_mixed = mins[np.argmin(mins)]
population_members_at_min = lenght_history[[sweep_index_minimal_l-1,sweep_index_minimal_l,sweep_index_minimal_l+1],:]
population_members_at_min = population_members_at_min.T

for p, m in enumerate(population_members_at_min):
    print(f"pop {p} lenght at {sweep_index_minimal_l-1}= {m[0]}, {sweep_index_minimal_l}= {m[1]} {sweep_index_minimal_l+1}= {m[2]}")
print("best window -----------------------------")
for p, m in enumerate(population_members_at_min):
    if (p==1) or (p==4) or (p==14):
        print(f"pop {p} lenght at {sweep_index_minimal_l-1}= {m[0]}, {sweep_index_minimal_l}= {m[1]} {sweep_index_minimal_l+1}= {m[2]}")
print(f"best members at {sweep_index_minimal_l-1}")
pop_mebers_before_min = np.argsort(population_members_at_min[:,0])
#print(pop_mebers_before_min)
for i in pop_mebers_before_min:
    print(f"pop {i} length {population_members_at_min[i,0]}")
print(min_indecces_mixed.shape)
print(f"mean of mins: {np.mean(mins)} +- {np.std(mins)}")

print(f"reach minimum after approx: {np.mean(min_indecces_mixed)} +- {np.std(min_indecces_mixed)}")

print(lenght_history.shape)



print("annealing---------------------")
#print(f"mean per population member")
mean_a = np.mean(annealing_data, axis=1)
#for p, m in enumerate(mean):
#    print(f"pop {p}: {m}")

l_T = lenght_history.T
min_indecces_an = np.empty(annealing_data.shape[0])
print(f"min per run")
mins = np.empty(annealing_data.shape[0])
for p, m in enumerate(annealing_data):
    minimal_idx = np.argmin(m)
    mins[p] = m[minimal_idx]
    min_indecces_an[p] = minimal_idx
    print(f"run {p}: {m[minimal_idx]} reached at sweep: {minimal_idx}")
minimum_an = mins[np.argmin(mins)]
print(f"mean of mins: {np.mean(mins)} +- {np.std(mins)}")
print(f"reach minimum after approx: {np.mean(min_indecces_an)} +- {np.std(min_indecces_an)}")
print(annealing_data.shape)

print(f"precent lower {1- minimum_an/minimum_mixed}")
print(f"gap precent {(minimum_mixed-minimum_an)/minimum_an}")