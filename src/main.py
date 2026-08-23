#!/usr/bin/env python3
"""Run and compare the annealing and mixed TSP solvers.

Set SOLVER to "annealing", "mixed", or "both" in the configuration below.
Every completed run is saved immediately as .npy files.
"""

from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

import modules.plotForReport as pr
from modules.TSP import annealing_D, mixed_annealing_D, construct_D_matrix, euclidian_dist, tsplib_dist , calculate_lenght
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOLVER = "both"  # "annealing", "mixed", or "both"
show_temp_func =True
DATA_FILE = Path("data/dsj1000.csv")
Set_name = str(DATA_FILE).split("\\")[-1].split(".")[0]
print(Set_name)
OUTPUT_DIRECTORY = Path(f"../results/tsp_benchmark/{Set_name}")

CSV_DELIMITER = ";"
COORDINATE_COLUMNS = (0, 1)

NUMBER_OF_RUNS = 8
NUMBER_OF_SWEEPS = 200
POPULATION_SIZE = 10

INITIAL_TEMPERATURE = 10000
COOLING_FACTOR = 0.99
COOLING_FACTOR_exp = 0.03
BASE_RANDOM_SEED = 28041999
PERIOD = 5
FREQUENCY = np.pi/PERIOD
KNOWN_OPTIMUM = 18659688
MINIMAL_CYCLE_TEMP = 800
MAKE_PLOTS_FOR_EACH_RUN = True
WARM_UP_NUMBA = True


# ---------------------------------------------------------------------------
# Functions required by the solvers
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def temperature_function(sweep, current_tour, current_length):
    """Simple exponential cooling schedule.

    current_tour and current_length are accepted because this is the callback
    signature expected by the current solver implementation.
    """
    return INITIAL_TEMPERATURE * COOLING_FACTOR**sweep
@nb.njit(cache=True)
def temperature_function_periodic(sweep,current_tour,current_lenght):
    return INITIAL_TEMPERATURE*np.exp(-COOLING_FACTOR_exp*sweep)*(np.cos(FREQUENCY*sweep)**2+ MINIMAL_CYCLE_TEMP*COOLING_FACTOR**sweep)
@nb.njit(cache=True)
def temperature_function_periodic2(sweep,current_tour,current_lenght):
    return INITIAL_TEMPERATURE*np.exp(-COOLING_FACTOR_exp*sweep)*(np.cos(FREQUENCY*sweep)**2+ 1)
@nb.njit(cache=True)
def temperature_function_periodic3(sweep,current_tour,current_lenght):
    return (INITIAL_TEMPERATURE*np.exp(-COOLING_FACTOR_exp*sweep)+MINIMAL_CYCLE_TEMP)*(np.cos(FREQUENCY*sweep)**2 + 1)
@nb.njit(cache=True)
def length_function(ordered_coordinates):
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


# ---------------------------------------------------------------------------
# Input and small conversion helpers
# ---------------------------------------------------------------------------


def load_coordinates(filename):
    """Load the selected x and y columns from a CSV file."""
    data = np.loadtxt(filename, delimiter=CSV_DELIMITER)

    if data.ndim != 2:
        raise ValueError("The input CSV must contain a two-dimensional table")

    coordinates = data[:, COORDINATE_COLUMNS]
    return np.array(coordinates, dtype=np.float64)


def one_temperature_history(temperatures):
    """Convert the current solver temperature output to one curve."""
    temperatures = np.asarray(temperatures)

    if temperatures.ndim == 1:
        return temperatures

    # The current mixed solver stores the same temperature per population.
    # Once that solver returns a 1D temperature array, this branch is unused.
    return temperatures[:, -1]


def plot_returned_tour(coordinates, returned_tour, title, save_path):
    """Plot either city IDs or the reordered coordinates returned currently."""
    returned_tour = np.asarray(returned_tour)

    if returned_tour.ndim == 1:
        return pr.plot_tour(
            coordinates,
            returned_tour,
            title=title,
            save_path=save_path,
        )

    if returned_tour.ndim == 2 and returned_tour.shape[1] == 2:
        city_ids = np.arange(returned_tour.shape[0])
        return pr.plot_tour(
            returned_tour,
            city_ids,
            title=title,
            save_path=save_path,
        )

    raise ValueError("The returned tour has an unsupported shape")


def save_aggregate_results(directory, histories, final_lengths, runtimes):
    """Save all completed runs after every new result."""
    np.save(directory / "all_length_histories.npy", np.stack(histories))
    np.save(directory / "all_final_lengths.npy", np.asarray(final_lengths))
    np.save(directory / "all_runtimes_seconds.npy", np.asarray(runtimes))


# ---------------------------------------------------------------------------
# Solver runners
# ---------------------------------------------------------------------------


def run_annealing(coordinates, distance_matrix, temp_func):
    """Run repeated independent annealing experiments."""
    output_directory = OUTPUT_DIRECTORY / "annealing"
    output_directory.mkdir(parents=True, exist_ok=True)

    histories = []
    final_lengths = []
    runtimes = []
    tour = coordinates.copy()
    starting_length = length_function(tour)#
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
    starting_length = length_function(tour)#
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
        #INITIAL_TEMPERATURE = INITIAL_TEMPERATURE/(run_index+1)
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
                temperature_history=one_temperature_history(temperatures),
                optimum=KNOWN_OPTIMUM,
                title=f"Annealing run {run_index}",
                save_path=run_directory / "diagnostics.png",
            )
            plt.close(figure)

            figure, _ = plot_returned_tour(
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


def run_mixed(coordinates, distance_matrix, temp_func):
    """Run repeated independent mixed-population experiments."""
    output_directory = OUTPUT_DIRECTORY / "mixed"
    output_directory.mkdir(parents=True, exist_ok=True)

    histories = []
    final_lengths = []
    runtimes = []
    tour = coordinates.copy()
    starting_length = length_function(tour)#
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
        #INITIAL_TEMPERATURE = INITIAL_TEMPERATURE/(run_index+1)
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
                temperature_history=one_temperature_history(temperatures),
                optimum=KNOWN_OPTIMUM,
                title=f"Mixed solver run {run_index}",
                save_path=run_directory / "diagnostics.png",
            )
            plt.close(figure)

            best_tour = final_population[best_population_index]
            figure, _ = plot_returned_tour(
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


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------


def main():
    
    if show_temp_func:
        sweep = np.arange(NUMBER_OF_SWEEPS)
        plt.plot(sweep, temperature_function_periodic3(sweep, 0 ,0 ))
        plt.show()
        plt.cla()
        plt.plot(sweep, temperature_function_periodic2(sweep, 0 ,0 ))
        plt.show()
        plt.cla()
        plt.plot(sweep, temperature_function_periodic(sweep, 0 ,0 ))
        plt.show()
        plt.cla()
        plt.plot(sweep,temperature_function(sweep,0,0))
        plt.show()
        print("satisfied with the temp funcs? [y/n]\n")
        x = input()
        if x == "n":
            exit()
    if SOLVER not in {"annealing", "mixed", "both"}:
        raise ValueError('SOLVER must be "annealing", "mixed", or "both"')

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    coordinates = load_coordinates(DATA_FILE)
    print("Coordinate shape:", coordinates.shape)
    distance_matrix = construct_D_matrix(coordinates,tsplib_dist)
    print(f"Distance Matrix {distance_matrix}")
    print(f"Loaded {coordinates.shape[0]} cities")
    print(f"Selected solver: {SOLVER}")

    annealing_results = None
    mixed_results = None
    
    
    if SOLVER == "annealing" or SOLVER == "both":
        annealing_results = run_annealing(coordinates, distance_matrix,temperature_function_periodic3)

    if SOLVER == "mixed" or SOLVER == "both":
        mixed_results = run_mixed(coordinates, distance_matrix,temperature_function_periodic3)

    if annealing_results is not None:
        print("\nAnnealing summary:")
        print(
            pr.benchmark_summary(
                annealing_results["final_lengths"],
                annealing_results["runtimes"],
                optimum=KNOWN_OPTIMUM,
            )
        )

    if mixed_results is not None:
        print("\nMixed summary:")
        print(
            pr.benchmark_summary(
                mixed_results["final_lengths"],
                mixed_results["runtimes"],
                optimum=KNOWN_OPTIMUM,
            )
        )

    if annealing_results is not None and mixed_results is not None:
        n_cities = coordinates.shape[0]

        annealing_x = (
            np.arange(1, annealing_results["histories"].shape[1] + 1)
            * n_cities**2
        )
        mixed_x = (
            np.arange(1, mixed_results["histories"].shape[1] + 1)
            * n_cities**2
            * POPULATION_SIZE
        )
        #annealing_x, mixed_x = np.log10(annealing_x), np.log10(mixed_x)
        figure, _ = pr.plot_comparison(
            annealing_results["histories"],
            mixed_results["histories"],
            labels=("Annealing", "Mixed"),
            x_a=annealing_x,
            x_b=mixed_x,
            optimum=KNOWN_OPTIMUM,
            x_label="sweeps / N",
            title="Annealing versus mixed solver",
            save_path=OUTPUT_DIRECTORY / "solver_comparison.png",
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
