"""Compare the exactly solved small TSP with mixed annealing.

This script deliberately does *not* generate permutations or solve the exact
system.  It loads the CSV produced by ``solve_system.py`` and performs the same
temperature scan with a mixed genetic/annealing population.

Before running this file, copy ``mixed_annealing_D_detailed`` and its private
helpers from the accompanying TSP file into ``modules/TSP.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from tqdm import tqdm

import modules.TSP as tsp
import modules.plotForReport as pr


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

RESULTS_PATH = Path("..") / "results" / "solve_system_N11" / "gpt_output"
PLOTS_PATH = RESULTS_PATH / "plots"
DETAIL_PATH = RESULTS_PATH / "mixed_population_details"

N_CITIES = 11
CITY_LOW = -40
CITY_HIGH = 40
RANDOM_SEED = 30121999

T_LIMS = (0.5, 50.0, 0.01)
TOTAL_SWEEPS = 800
WARMUP_SWEEPS = 200
POPULATION_SIZE = 8

# False: no optional population metadata is returned or saved.
# True: lightweight final metadata is retained for every temperature.
DETAILED = True

# When detailed, population-member histories are downsampled to every N sweeps.
DETAIL_RECORD_EVERY = 25

# Full downsampled histories are written only for every Nth temperature (and
# the final temperature).  They are written immediately, never accumulated in
# RAM across all temperatures.
DETAIL_TEMPERATURE_EVERY = 250

SHOW_PLOTS = False
REUSE_EXISTING_MIXED_DATA = True


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

@nb.njit(inline="always")
def calculate_total_distance_with_D(D, tour_ids):
    total_length = 0.0
    n_cities = len(tour_ids)
    for i in range(n_cities):
        total_length += D[
            tour_ids[i], tour_ids[(i + 1) % n_cities]
        ]
    return total_length


@nb.njit
def temp_func_const(n, ids, length):
    """Fallback schedule; const_temp normally supplies the temperature."""
    return 1.0


@nb.njit
def temp_func_warmup(n, target_temperature, length):
    """The same warm-up schedule used in solve_system.py."""
    return 1200.0 / ((n + 1.0)*10) + target_temperature


def _population_lengths(population, D):
    lengths = np.empty(population.shape[0], dtype=np.float64)
    for p in range(population.shape[0]):
        lengths[p] = calculate_total_distance_with_D(D, population[p])
    return lengths


def _initial_population(tour_ids, population_size, seed):
    """Keep one reference tour and shuffle the remaining members."""
    if population_size < 2 or population_size % 2 != 0:
        raise ValueError("population_size must be an even integer >= 2")

    rng = np.random.default_rng(seed)
    population = np.empty(
        (population_size, tour_ids.size), dtype=np.int32
    )
    population[0] = tour_ids
    for p in range(1, population_size):
        population[p] = rng.permutation(tour_ids)
    return population


def _tag(t_lims):
    return f"{t_lims[0]}_{t_lims[1]}_{t_lims[2]}"


def _save_semicolon_csv(path, data, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.asarray(data),
        delimiter=";",
        header=";".join(columns),
    )


# ---------------------------------------------------------------------------
# Exact-solution input: loading only, no permutations in this script
# ---------------------------------------------------------------------------

def load_solved_system_data(
        results_path,
        t_lims,
        add_info="gen",
):
    """Load the exact CSV previously written by solve_system.py.

    The returned shape is (6, n_temperatures):
    T, Z(T), <L>, Var(L), C(T), <L^2>.
    """
    path = Path(results_path) / (
        f"solved_system_data_{_tag(t_lims)}_{add_info}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Exact-solution file not found: {path}\n"
            "Run solve_system.py once to create it. This mixed script does "
            "not calculate permutations."
        )

    data = np.loadtxt(path, delimiter=";", skiprows=1)
    data = np.atleast_2d(data)
    if data.shape[1] != 6:
        raise ValueError(
            f"Expected 6 columns in {path}, found {data.shape[1]}"
        )
    return data.T


# ---------------------------------------------------------------------------
# Mixed temperature scan
# ---------------------------------------------------------------------------

def solve_system_with_TSP_mixed(
        tour_ids,
        D,
        temperatures,
        nsweeps,
        population_size=8,
        warmup=1500,
        detailed=False,
        record_every=1,
        detail_temperature_every=1,
        detail_output_directory=None,
        seed=30121999,
):
    """Mixed-population counterpart of solve_system_with_TSP.

    ``nsweeps`` has the same meaning as in the original function: it includes
    the warm-up sweeps.  Each temperature is first warmed up and is then
    sampled for ``nsweeps - warmup`` sweeps.

    The return structure keeps the original two-list style, but avoids storing
    arrays with a temperature x sweep x population x proposal dimension:

    ``data1``
        [temperatures, mean_L_per_temperature, acceptance_per_temperature]
    ``data2``
        [mean_L, variance_L, heat_capacity, mean_L_squared]

    If ``detailed=True``, ``data2`` has a fifth item containing lightweight
    population summary arrays and paths to incrementally saved detailed NPZ
    files.  The NPZ files contain the downsampled length of every population
    member, member acceptance rates, diversity, uniqueness, and genetic versus
    annealing length changes.  With ``detailed=False`` no population metadata
    item is returned and no detailed files are written.

    Important: tournament selection and crossover do not satisfy Metropolis
    detailed balance.  The resulting moments are therefore empirical moments
    of the mixed algorithm, not unbiased canonical-ensemble estimates.  The
    exact comparison is useful as an algorithm diagnostic, but it must not be
    interpreted as a thermodynamically exact sampler test.
    """
    temperatures = np.atleast_1d(
        np.asarray(temperatures, dtype=np.float64)
    )
    tour_ids = np.asarray(tour_ids, dtype=np.int32)
    D = np.asarray(D, dtype=np.float64)

    measurement_sweeps = int(nsweeps) - int(warmup)
    if measurement_sweeps <= 0:
        raise ValueError("nsweeps must be larger than warmup")
    if record_every < 1:
        raise ValueError("record_every must be at least 1")
    if detail_temperature_every < 1:
        raise ValueError(
            "detail_temperature_every must be at least 1"
        )

    population = _initial_population(
        tour_ids, population_size, seed
    )
    current_lengths = _population_lengths(population, D)
    tsp.seed_numba(seed)

    n_temperatures = temperatures.size
    mean_L_T = np.empty(n_temperatures, dtype=np.float64)
    mean_L_squared_T = np.empty(n_temperatures, dtype=np.float64)
    variance_L_T = np.empty(n_temperatures, dtype=np.float64)
    heat_capacity_T = np.empty(n_temperatures, dtype=np.float64)
    acceptance_rate_T = np.empty(n_temperatures, dtype=np.float64)

    metadata = None
    detailed_files = []
    detail_output_directory = (
        None
        if detail_output_directory is None
        else Path(detail_output_directory)
    )

    if detailed:
        metadata = {
            "temperature": temperatures.copy(),
            "acceptance_mean": np.empty(n_temperatures),
            "final_best_length": np.empty(n_temperatures),
            "final_mean_population_length": np.empty(n_temperatures),
            "final_variance_population_length": np.empty(n_temperatures),
            "final_edge_diversity": np.empty(n_temperatures),
            "final_unique_cycle_fraction": np.empty(n_temperatures),
            "best_seen_length": np.empty(n_temperatures),
            "last_crossover_delta_mean": np.empty(n_temperatures),
            "last_annealing_delta_mean": np.empty(n_temperatures),
            "last_improved_fraction": np.empty(n_temperatures),
            "samples_per_temperature": np.full(
                n_temperatures,
                measurement_sweeps
                * population_size
                * tour_ids.size
                * tour_ids.size,
                dtype=np.int64,
            ),
            "detail_saved": np.zeros(n_temperatures, dtype=np.int64),
        }
        if detail_output_directory is not None:
            detail_output_directory.mkdir(parents=True, exist_ok=True)

    print(
        "total expected temperature iterations = "
        f"{n_temperatures}"
    )

    iterator = tqdm(
        enumerate(temperatures),
        total=n_temperatures,
        desc="solving with mixed TSP",
    )
    for temperature_index, temperature in iterator:
        # Work on a copy because the mixed solver mutates its population.
        warm_output = tsp.mixed_annealing_D_detailed(
            population.copy(),
            D,
            population_size,
            temp_func_warmup,
            current_lengths.copy(),
            warmup,
            const_temp=temperature,
            warm_up=True,
            detailed=False,
        )
        population = warm_output[0]
        current_lengths = warm_output[1][-1].copy()

        save_this_temperature = detailed and (
            temperature_index % detail_temperature_every == 0
            or temperature_index == n_temperatures - 1
        )

        # For temperatures that are not selected for a complete diagnostic
        # history, detailed mode records only the initial and final states.
        current_record_every = (
            record_every
            if save_this_temperature
            else measurement_sweeps
        )

        measurement_output = tsp.mixed_annealing_D_detailed(
            population.copy(),
            D,
            population_size,
            temp_func_const,
            current_lengths.copy(),
            measurement_sweeps,
            const_temp=temperature,
            warm_up=False,
            detailed=detailed,
            record_every=current_record_every,
        )

        (
            population,
            length_history,
            _temperature_history,
            sweep_mean_L,
            sweep_mean_L_squared,
            _sweep_variance,
            _sweep_heat_capacity,
            sweep_acceptance,
        ) = measurement_output[:8]
        current_lengths = length_history[-1].copy()

        # All sweeps have the same number of proposals, so averaging their
        # online moments gives the moment over every sampled proposal state.
        mean_L = float(np.mean(sweep_mean_L))
        mean_L_squared = float(np.mean(sweep_mean_L_squared))
        variance_L = mean_L_squared - mean_L * mean_L
        if variance_L < 0.0 and variance_L > -1e-10:
            variance_L = 0.0

        mean_L_T[temperature_index] = mean_L
        mean_L_squared_T[temperature_index] = mean_L_squared
        variance_L_T[temperature_index] = variance_L
        heat_capacity_T[temperature_index] = (
            variance_L / (temperature * temperature)
            if temperature > 0.0
            else np.nan
        )
        acceptance_rate_T[temperature_index] = np.mean(
            sweep_acceptance
        )

        if detailed:
            run_metadata = measurement_output[8]
            final_record = -1
            metadata["acceptance_mean"][temperature_index] = (
                acceptance_rate_T[temperature_index]
            )
            metadata["final_best_length"][temperature_index] = (
                run_metadata["best_length"][final_record]
            )
            metadata["final_mean_population_length"][temperature_index] = (
                run_metadata["mean_population_length"][final_record]
            )
            metadata[
                "final_variance_population_length"
            ][temperature_index] = run_metadata[
                "variance_population_length"
            ][final_record]
            metadata["final_edge_diversity"][temperature_index] = (
                run_metadata["edge_diversity"][final_record]
            )
            metadata[
                "final_unique_cycle_fraction"
            ][temperature_index] = run_metadata[
                "unique_cycle_fraction"
            ][final_record]
            metadata["best_seen_length"][temperature_index] = (
                run_metadata["best_so_far"][final_record]
            )
            metadata[
                "last_crossover_delta_mean"
            ][temperature_index] = run_metadata[
                "crossover_delta_mean"
            ][final_record]
            metadata[
                "last_annealing_delta_mean"
            ][temperature_index] = run_metadata[
                "annealing_delta_mean"
            ][final_record]
            metadata["last_improved_fraction"][temperature_index] = (
                run_metadata["improved_fraction"][final_record]
            )

            if save_this_temperature and detail_output_directory is not None:
                detail_file = detail_output_directory / (
                    f"temperature_{temperature_index:05d}.npz"
                )
                np.savez_compressed(
                    detail_file,
                    temperature=np.array(temperature),
                    temperature_index=np.array(temperature_index),
                    final_population=population,
                    final_population_lengths=current_lengths,
                    sample_mean_L=np.array(mean_L),
                    sample_variance_L=np.array(variance_L),
                    **run_metadata,
                )
                metadata["detail_saved"][temperature_index] = 1
                detailed_files.append(str(detail_file))

    data1 = [
        temperatures,
        mean_L_T,
        acceptance_rate_T,
    ]
    data2 = [
        mean_L_T,
        variance_L_T,
        heat_capacity_T,
        mean_L_squared_T,
    ]
    if detailed:
        metadata["detail_files"] = detailed_files
        data2.append(metadata)

    return data1, data2


# ---------------------------------------------------------------------------
# Saving and plotting
# ---------------------------------------------------------------------------

def mixed_summary_array(data1, data2):
    temperatures = data1[0]
    return np.vstack(
        (
            temperatures,
            np.full(temperatures.shape, np.nan),
            data2[0],
            data2[1],
            data2[2],
            data2[3],
        )
    )


def save_mixed_results(
        mixed_data,
        results_path,
        t_lims,
        metadata=None,
):
    summary_path = Path(results_path) / (
        f"solved_system_data_{_tag(t_lims)}_mixed_TSP.csv"
    )
    _save_semicolon_csv(
        summary_path,
        mixed_data.T,
        ["T", "Z(T)", "L_mean", "var", "C(T)", "L_sq"],
    )

    metadata_path = None
    if metadata is not None:
        metadata_columns = [
            "temperature",
            "acceptance_mean",
            "final_best_length",
            "final_mean_population_length",
            "final_variance_population_length",
            "final_edge_diversity",
            "final_unique_cycle_fraction",
            "best_seen_length",
            "last_crossover_delta_mean",
            "last_annealing_delta_mean",
            "last_improved_fraction",
            "samples_per_temperature",
            "detail_saved",
        ]
        metadata_array = np.column_stack(
            [metadata[name] for name in metadata_columns]
        )
        metadata_path = Path(results_path) / (
            f"solved_system_data_{_tag(t_lims)}_mixed_TSP_metadata.csv"
        )
        _save_semicolon_csv(
            metadata_path, metadata_array, metadata_columns
        )

    return summary_path, metadata_path


def plot_exact_vs_mixed(
        exact_data,
        mixed_data,
        plots_path,
        t_lims,
):
    """Plot the same exact quantities with the mixed estimates overlaid."""
    plots_path = Path(plots_path)
    plots_path.mkdir(parents=True, exist_ok=True)
    temperature = exact_data[0]

    panel_data = [
        (1, r"$Z(T)$", False),
        (2, r"$\langle L \rangle_T$", True),
        (3, r"$\langle L^2 \rangle_T-\langle L \rangle_T^2$", True),
        (4, r"$C(T)$", True),
        (5, r"$\langle L^2 \rangle_T$", True),
    ]

    fig, axes = plt.subplots(len(panel_data), 1, figsize=(8, 11), sharex=True)
    for ax, (row, ylabel, plot_mixed) in zip(axes, panel_data):
        ax.plot(
            temperature,
            exact_data[row],
            label="exact",
            color="black",
            linewidth=1.5,
        )
        if plot_mixed:
            ax.plot(
                mixed_data[0],
                mixed_data[row],
                label="mixed annealing",
                color="tab:orange",
                linewidth=1.2,
            )
        if row == 2:
            standard_deviation = np.sqrt(
                np.maximum(mixed_data[3], 0.0)
            )
            ax.fill_between(
                mixed_data[0],
                mixed_data[2] - standard_deviation,
                mixed_data[2] + standard_deviation,
                color="tab:orange",
                alpha=0.15,
                label=r"mixed $\pm 1\sigma$",
            )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel(r"$T$")
    fig.tight_layout()
    output = plots_path / (
        f"solved_system_{_tag(t_lims)}_mixed_comparison.pdf"
    )
    fig.savefig(output, bbox_inches="tight")
    return fig, axes


def plot_temperature_metadata(metadata, plots_path, t_lims):
    plots_path = Path(plots_path)
    temperature = metadata["temperature"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    axes[0, 0].plot(temperature, metadata["acceptance_mean"])
    axes[0, 0].set_ylabel("acceptance rate")
    axes[0, 0].set_ylim(0.0, 1.0)

    axes[0, 1].plot(temperature, metadata["final_edge_diversity"])
    axes[0, 1].set_ylabel("edge diversity")
    axes[0, 1].set_ylim(0.0, 1.0)

    axes[1, 0].plot(
        temperature, metadata["final_unique_cycle_fraction"]
    )
    axes[1, 0].set_ylabel("unique-cycle fraction")
    axes[1, 0].set_ylim(0.0, 1.0)

    axes[1, 1].plot(
        temperature,
        metadata["final_best_length"],
        label="final best",
    )
    axes[1, 1].plot(
        temperature,
        metadata["final_mean_population_length"],
        label="final mean",
    )
    axes[1, 1].set_ylabel("population length")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xlabel(r"$T$")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    output = plots_path / (
        f"solved_system_{_tag(t_lims)}_mixed_metadata.pdf"
    )
    fig.savefig(output, bbox_inches="tight")
    return fig, axes


def plot_saved_population_details(detail_files, plots_path):
    """Read and plot one detailed file at a time to keep RAM bounded."""
    plots_path = Path(plots_path) / "mixed_population_details"
    plots_path.mkdir(parents=True, exist_ok=True)

    for detail_file in detail_files:
        with np.load(detail_file) as data:
            temperature = float(data["temperature"])
            temperature_index = int(data["temperature_index"])
            recorded_sweeps = data["recorded_sweeps"]
            population_lengths = data["population_lengths"]
            member_acceptance = data["member_acceptance_rate"]

            stem = f"temperature_{temperature_index:05d}"
            figure, _ = pr.plot_population(
                population_lengths,
                x=recorded_sweeps,
                title=f"Mixed population at T={temperature:g}",
                x_label="sweep",
                save_path=plots_path / f"{stem}_population.pdf",
            )
            plt.close(figure)

            acceptance = np.nanmean(member_acceptance, axis=1)
            figure, _ = pr.plot_diagnostics(
                best_history=data["best_so_far"],
                mean_history=data["mean_population_length"],
                variance_history=data["variance_population_length"],
                temperature_history=np.full(
                    recorded_sweeps.shape, temperature
                ),
                acceptance_history=acceptance,
                diversity_history=data["edge_diversity"],
                title=f"Mixed diagnostics at T={temperature:g}",
                save_path=plots_path / f"{stem}_diagnostics.pdf",
            )
            plt.close(figure)


def main():
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    exact_data = load_solved_system_data(
        RESULTS_PATH, T_LIMS, add_info="gen"
    )
    temperatures = exact_data[0]

    tour = tsp.create_cities(
        N_CITIES,
        low=CITY_LOW,
        high=CITY_HIGH,
        seed=RANDOM_SEED,
    )
    tour_ids = tsp.create_tour_ids(tour)
    D = tsp.construct_D_matrix(tour, tsp.euclidian_dist)

    mixed_csv = RESULTS_PATH / (
        f"solved_system_data_{_tag(T_LIMS)}_mixed_TSP.csv"
    )
    metadata_csv = RESULTS_PATH / (
        f"solved_system_data_{_tag(T_LIMS)}_mixed_TSP_metadata.csv"
    )

    metadata = None
    if REUSE_EXISTING_MIXED_DATA and mixed_csv.exists():
        print(f"loading mixed data from {mixed_csv}")
        mixed_data = np.loadtxt(
            mixed_csv, delimiter=";", skiprows=1
        ).T
        if DETAILED and metadata_csv.exists():
            metadata_values = np.loadtxt(
                metadata_csv, delimiter=";", skiprows=1
            )
            metadata_values = np.atleast_2d(metadata_values)
            metadata_names = [
                "temperature",
                "acceptance_mean",
                "final_best_length",
                "final_mean_population_length",
                "final_variance_population_length",
                "final_edge_diversity",
                "final_unique_cycle_fraction",
                "best_seen_length",
                "last_crossover_delta_mean",
                "last_annealing_delta_mean",
                "last_improved_fraction",
                "samples_per_temperature",
                "detail_saved",
            ]
            metadata = {
                name: metadata_values[:, i]
                for i, name in enumerate(metadata_names)
            }
            metadata["detail_files"] = []
            for temperature_index in np.flatnonzero(
                metadata["detail_saved"]
            ):
                path = DETAIL_PATH / (
                    f"temperature_{temperature_index:05d}.npz"
                )
                if path.exists():
                    metadata["detail_files"].append(str(path))
    else:
        data1, data2 = solve_system_with_TSP_mixed(
            tour_ids,
            D,
            temperatures,
            TOTAL_SWEEPS,
            population_size=POPULATION_SIZE,
            warmup=WARMUP_SWEEPS,
            detailed=DETAILED,
            record_every=DETAIL_RECORD_EVERY,
            detail_temperature_every=DETAIL_TEMPERATURE_EVERY,
            detail_output_directory=DETAIL_PATH,
            seed=RANDOM_SEED,
        )
        mixed_data = mixed_summary_array(data1, data2)
        if DETAILED:
            metadata = data2[4]
        save_mixed_results(
            mixed_data,
            RESULTS_PATH,
            T_LIMS,
            metadata=metadata,
        )

    if mixed_data.shape != exact_data.shape:
        raise ValueError(
            "Exact and mixed data do not have the same shape: "
            f"{exact_data.shape} versus {mixed_data.shape}"
        )
    if not np.allclose(mixed_data[0], exact_data[0]):
        raise ValueError(
            "Temperature grid in mixed data does not match exact data"
        )

    comparison_figure, _ = plot_exact_vs_mixed(
        exact_data, mixed_data, PLOTS_PATH, T_LIMS
    )

    if metadata is not None:
        metadata_figure, _ = plot_temperature_metadata(
            metadata, PLOTS_PATH, T_LIMS
        )
        plot_saved_population_details(
            metadata.get("detail_files", []), PLOTS_PATH
        )
    else:
        metadata_figure = None

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(comparison_figure)
        if metadata_figure is not None:
            plt.close(metadata_figure)


if __name__ == "__main__":
    main()
