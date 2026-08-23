"""Simple runtime metrics and plots for TSP solvers.

Expected array shapes
---------------------
tour:               (n_cities,)
tour_population:    (population_size, n_cities)
population_history: (n_records, population_size)
repeated histories: (n_runs, n_records)
"""

import matplotlib.pyplot as plt
import numba as nb
import numpy as np


# Functions that can be called inside @nb.njit solver functions.


@nb.njit(cache=True)
def tour_length(tour, distance_matrix):
    """Length of a closed tour."""
    length = 0.0
    n_cities = tour.size

    for i in range(n_cities):
        city_a = tour[i]
        city_b = tour[(i + 1) % n_cities]
        length += distance_matrix[city_a, city_b]

    return length


@nb.njit(cache=True)
def population_length_statistics(lengths):
    """Return best, mean, variance and the index of the best tour."""
    population_size = lengths.size
    best = float(lengths[0])
    best_index = 0
    mean = 0.0

    for p in range(population_size):
        value = float(lengths[p])
        mean += value

        if value < best:
            best = value
            best_index = p

    mean /= population_size

    variance = 0.0
    for p in range(population_size):
        difference = float(lengths[p]) - mean
        variance += difference * difference

    variance /= population_size
    return best, mean, variance, best_index


@nb.njit(cache=True)
def edge_diversity(tour_population, best_index):
    """Mean edge difference between the population and its best tour.

    Zero means that all tours have the same undirected edges. A value near
    one means that they share few edges. City IDs must be 0, ..., N - 1.
    """
    population_size, n_cities = tour_population.shape

    if population_size <= 1:
        return 0.0

    # Store the two neighbours of each city in the best tour.
    neighbours = np.empty((n_cities, 2), dtype=np.int64)
    best_tour = tour_population[best_index]

    for i in range(n_cities):
        city = best_tour[i]
        neighbours[city, 0] = best_tour[(i - 1) % n_cities]
        neighbours[city, 1] = best_tour[(i + 1) % n_cities]

    diversity_sum = 0.0

    for p in range(population_size):
        if p == best_index:
            continue

        shared_edges = 0
        candidate = tour_population[p]

        for i in range(n_cities):
            city_a = candidate[i]
            city_b = candidate[(i + 1) % n_cities]

            if (
                city_b == neighbours[city_a, 0]
                or city_b == neighbours[city_a, 1]
            ):
                shared_edges += 1

        diversity_sum += 1.0 - shared_edges / n_cities

    return diversity_sum / (population_size - 1)


@nb.njit(cache=True)
def population_snapshot(tour_population, current_lengths):
    """Return best length, mean, variance, diversity and best index."""
    best, mean, variance, best_index = population_length_statistics(
        current_lengths
    )
    diversity = edge_diversity(tour_population, best_index)
    return best, mean, variance, diversity, best_index


# Ordinary Python functions for analysis and plotting.


def benchmark_summary(final_lengths, runtimes=None, optimum=None):
    """Return report statistics over independent runs."""
    values = np.asarray(final_lengths, dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("final_lengths must be a non-empty 1D array")

    summary = {
        "number_of_runs": int(values.size),
        "best": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "variance": float(np.var(values)),
        "standard_deviation": float(np.std(values)),
        "worst": float(np.max(values)),
    }

    if optimum is not None:
        gaps = 100.0 * (values - optimum) / optimum
        summary["best_gap_percent"] = float(np.min(gaps))
        summary["mean_gap_percent"] = float(np.mean(gaps))
        summary["runs_reaching_optimum"] = int(
            np.count_nonzero(values == optimum)
        )

    if runtimes is not None:
        runtimes = np.asarray(runtimes, dtype=np.float64)
        if runtimes.shape != values.shape:
            raise ValueError(
                "runtimes and final_lengths must have the same shape"
            )

        summary["mean_runtime_seconds"] = float(np.mean(runtimes))
        summary["runtime_std_seconds"] = float(np.std(runtimes))

    return summary


def plot_tour(coordinates, tour, title="TSP tour", save_path=None):
    """Plot one closed two-dimensional tour."""
    coordinates = np.asarray(coordinates)
    tour = np.asarray(tour)

    ordered = coordinates[tour]
    closed = np.vstack((ordered, ordered[0]))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(closed[:, 0], closed[:, 1], linewidth=0.7, color="tab:blue")
    ax.scatter(ordered[:, 0], ordered[:, 1], s=4, color="black", zorder=2)
    ax.set(title=title, xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    _save(fig, save_path)
    return fig, ax


def plot_population(
    length_history,
    x=None,
    optimum=None,
    title="Population convergence",
    x_label="recorded step",
    save_path=None,
):
    """Plot individual population members, their mean and their best.

    length_history must have shape (n_records, population_size).
    """
    history = np.asarray(length_history, dtype=np.float64)
    if history.ndim == 1:
        history = history[:, None]
    if history.ndim != 2:
        raise ValueError("length_history must be a 1D or 2D array")

    n_records = history.shape[0]
    x = np.arange(n_records) if x is None else np.asarray(x)
    if x.size != n_records:
        raise ValueError("x must have one value per recorded step")

    mean = np.mean(history, axis=1)
    std = np.std(history, axis=1)
    best = np.min(history, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for p in range(history.shape[1]):
        label = "individual tours" if p == 0 else None
        ax.plot(
            x,
            history[:, p],
            color="0.75",
            linewidth=0.6,
            alpha=0.45,
            label=label,
        )

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color="tab:blue",
        alpha=0.18,
        label="mean ± standard deviation",
    )
    ax.plot(x, mean, color="tab:blue", linewidth=1.8, label="mean")
    ax.plot(x, best, color="tab:red", linewidth=1.8, label="best")

    if optimum is not None:
        ax.axhline(
            optimum,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="optimum",
        )

    ax.set(title=title, xlabel=x_label, ylabel="tour length")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    _save(fig, save_path)
    return fig, ax


def plot_diagnostics(
    best_history,
    mean_history=None,
    variance_history=None,
    temperature_history=None,
    acceptance_history=None,
    diversity_history=None,
    optimum=None,
    title="Solver diagnostics",
    save_path=None,
):
    """Plot convergence, temperature, acceptance rate and diversity."""
    best = np.asarray(best_history, dtype=np.float64)
    x = np.arange(best.size)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    length_ax, temperature_ax, acceptance_ax, diversity_ax = axes.ravel()

    length_ax.plot(x, best, color="tab:red", label="best")

    if mean_history is not None:
        mean = _history(mean_history, best.size, "mean_history")
        length_ax.plot(x, mean, color="tab:blue", label="mean")

        if variance_history is not None:
            variance = _history(
                variance_history, best.size, "variance_history"
            )
            std = np.sqrt(np.maximum(variance, 0.0))
            length_ax.fill_between(
                x, mean - std, mean + std, color="tab:blue", alpha=0.18
            )

    if optimum is not None:
        length_ax.axhline(
            optimum, color="black", linestyle="--", label="optimum"
        )

    length_ax.set(title="Convergence", ylabel="tour length")
    length_ax.legend()

    _optional_plot(
        temperature_ax,
        temperature_history,
        best.size,
        "Temperature",
        "temperature",
        "tab:orange",
    )
    _optional_plot(
        acceptance_ax,
        acceptance_history,
        best.size,
        "Acceptance rate",
        "accepted / proposed",
        "tab:green",
    )
    _optional_plot(
        diversity_ax,
        diversity_history,
        best.size,
        "Edge diversity",
        "mean edge difference",
        "tab:purple",
    )

    acceptance_ax.set_ylim(0.0, 1.0)
    diversity_ax.set_ylim(0.0, 1.0)

    for ax in axes.ravel():
        ax.set_xlabel("recorded step")
        ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()

    _save(fig, save_path)
    return fig, axes


def plot_comparison(
    histories_a,
    histories_b,
    labels=("Annealing", "Mixed"),
    x_a=None,
    x_b=None,
    optimum=None,
    x_label="evaluated moves or elapsed time",
    title="Solver comparison",
    save_path=None,
):
    """Compare convergence and final results over independent runs.

    Each history has shape (n_runs, n_records). A 1D input is treated as
    one run. The curves show best-so-far length, not current length.
    """
    histories_a = _run_matrix(histories_a)
    histories_b = _run_matrix(histories_b)

    best_a = np.minimum.accumulate(histories_a, axis=1)
    best_b = np.minimum.accumulate(histories_b, axis=1)

    x_a = np.arange(best_a.shape[1]) if x_a is None else np.asarray(x_a)
    x_b = np.arange(best_b.shape[1]) if x_b is None else np.asarray(x_b)

    if x_a.size != best_a.shape[1] or x_b.size != best_b.shape[1]:
        raise ValueError("each x array must match its recorded history")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    convergence_ax, final_ax = axes

    _mean_std_curve(
        convergence_ax, x_a, best_a, labels[0], "tab:blue"
    )
    _mean_std_curve(
        convergence_ax, x_b, best_b, labels[1], "tab:orange"
    )

    if optimum is not None:
        convergence_ax.axhline(
            optimum,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="optimum",
        )
        final_ax.axhline(
            optimum, color="black", linestyle="--", linewidth=1.2
        )

    convergence_ax.set(
        title="Best-so-far convergence",
        xlabel=x_label,
        ylabel="tour length",
    )
    convergence_ax.grid(alpha=0.25)
    convergence_ax.legend()

    final_ax.boxplot(
        [best_a[:, -1], best_b[:, -1]],
        tick_labels=labels,
        showmeans=True,
    )
    final_ax.set(title="Final best lengths", ylabel="tour length")
    final_ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()

    _save(fig, save_path)
    return fig, axes


# Small internal plotting helpers.


def _history(values, expected_size, name):
    values = np.asarray(values, dtype=np.float64)
    if values.size != expected_size:
        raise ValueError(f"{name} has the wrong length")
    return values


def _optional_plot(ax, values, expected_size, title, ylabel, color):
    ax.set(title=title, ylabel=ylabel)
    if values is None:
        ax.text(
            0.5,
            0.5,
            "not recorded",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="0.45",
        )
        return

    values = _history(values, expected_size, title)
    ax.plot(np.arange(expected_size), values, color=color)


def _run_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values[None, :]
    if values.ndim != 2:
        raise ValueError("histories must have shape (n_runs, n_records)")
    return values


def _mean_std_curve(ax, x, histories, label, color):
    mean = np.mean(histories, axis=0)
    std = np.std(histories, axis=0)
    ax.plot(x, mean, color=color, linewidth=1.8, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


def _save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
