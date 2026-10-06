from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.core import (
    Package,
    State,
    Truck,
    calculate_objective,
    generate_initial_state,
)
from src.simulated_annealing import (
    OBJECTIVE_VALUE,
    OBJECTIVE_ETA_WEIGHTED,
    calculate_sa_objective,
    simulated_annealing,
)
from src.visualization import save_sa_visualizations


OBJECTIVE_MODE = OBJECTIVE_ETA_WEIGHTED

OUTPUT_DIR = Path(
    f"output/simulated_annealing/{OBJECTIVE_MODE}"
)

SA_CONFIG = {
    "initial_temperature": 100.0,
    "cooling_rate": 0.99,
    "min_temperature": 1e-3,
    "max_iterations": 1000,
    "stuck_threshold": 50,
    "local_optimum_probe_size": 50,
}

RUN_SEEDS = [42, 123, 999]


def make_experiment_problem():
    truck = Truck(
        width=8,
        length=8,
        height=5,
        max_capacity=40,
    )

    packages = (
        Package("A", 4, 4, 2, 140, 10, False, 3),
        Package("B", 4, 3, 2, 120, 9, False, 2),
        Package("C", 3, 3, 2, 95, 7, False, 5),
        Package("D", 5, 3, 2, 160, 12, False, 1),
        Package("E", 2, 2, 2, 70, 5, True, 4),
        Package("F", 3, 2, 2, 85, 6, False, 6),
        Package("G", 4, 2, 1, 75, 5, False, 2),
        Package("H", 2, 2, 1, 55, 3, True, 3),
        Package("I", 3, 3, 1, 90, 6, False, 1),
        Package("J", 5, 2, 2, 130, 10, False, 5),
        Package("K", 2, 3, 2, 80, 6, False, 4),
        Package("L", 3, 2, 1, 65, 4, False, 7),
    )

    return truck, packages



def state_to_dict(
    state: State,
    objective_mode: str,
) -> Dict[str, Any]:

    packages = {
        package.id: package
        for package in state.packages
    }

    inside = []

    for package_id, placement in state.placements.items():
        package = packages[package_id]

        inside.append({
            "id": package.id,
            "dimensions": {
                "width": package.width,
                "length": package.length,
                "height": package.height,
            },
            "value": package.value,
            "weight": package.weight,
            "is_fragile": package.is_fragile,
            "eta": package.eta,
            "position": {
                "x": placement.x,
                "y": placement.y,
                "z": placement.z,
            },
            "orientation": placement.orientation,
        })

    outside = [
        package.id
        for package in state.packages
        if package.id not in state.placements
    ]

    return {
        "objective_mode": objective_mode,
        "objective_score": calculate_sa_objective(
            state,
            objective_mode,
        ),
        "total_value": calculate_objective(state),
        "inside_packages": inside,
        "outside_packages": outside,
    }



def write_history_csv(
    run_dir: Path,
    result,
) -> None:

    history_path = run_dir / "history.csv"

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "iteration",
            "current_objective",
            "best_objective",
            "neighbor_objective",
            "temperature",
            "delta_e",
            "acceptance_probability",
            "accepted",
        ])

        for i in range(result.iterations):
            writer.writerow([
                i + 1,
                result.objective_history[i + 1],
                result.best_objective_history[i + 1],
                result.neighbor_objective_history[i],
                result.temperature_history[i],
                result.delta_e_history[i],
                result.acceptance_probability_history[i],
                result.accepted_history[i],
            ])


def save_run(
    run_number: int,
    seed: int,
    result,
) -> Dict[str, Any]:

    run_dir = OUTPUT_DIR / f"run_{run_number}"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    objective_mode = result.metadata["objective_mode"]

    states = {
        "initial_state": state_to_dict(
            result.initial_state,
            objective_mode,
        ),
        "final_state": state_to_dict(
            result.final_state,
            objective_mode,
        ),
        "best_state": state_to_dict(
            result.best_state,
            objective_mode,
        ),
    }

    with (run_dir / "states.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            states,
            file,
            indent=2,
        )

    write_history_csv(
        run_dir,
        result,
    )


    save_sa_visualizations(
        result,
        run_dir,
        show=False,
    )


    summary = {
        "run": run_number,
        "seed": seed,

        "objective_mode":
            result.metadata["objective_mode"],

        "objective_formula":
            result.metadata["objective_formula"],

        "initial_objective":
            result.metadata["initial_objective"],

        "final_objective":
            result.metadata["final_objective"],

        "best_objective":
            result.metadata["best_objective"],

        "initial_total_value":
            result.metadata["initial_total_value"],

        "final_total_value":
            result.metadata["final_total_value"],

        "best_total_value":
            result.metadata["best_total_value"],

        "iterations":
            result.iterations,

        "duration_seconds":
            result.duration,

        "accepted_moves":
            result.metadata["accepted_moves"],

        "accepted_better_or_equal_moves":
            result.metadata[
                "accepted_better_or_equal_moves"
            ],

        "accepted_worse_moves":
            result.metadata[
                "accepted_worse_moves"
            ],

        "rejected_moves":
            result.metadata["rejected_moves"],

        "stuck_count":
            result.metadata["stuck_count"],

        "stuck_iterations":
            result.metadata["stuck_iterations"],

        "longest_no_improvement_streak":
            result.metadata[
                "longest_no_improvement_streak"
            ],

        "stop_reason":
            result.metadata["stop_reason"],
    }

    with (run_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return summary

def print_summary(
    summaries: List[Dict[str, Any]],
) -> None:

    print(
        "\nSIMULATED ANNEALING EXPERIMENT SUMMARY"
    )

    print(
        "Objective mode:",
        OBJECTIVE_MODE,
    )

    print()

    for summary in summaries:
        print(
            f"Run {summary['run']} | "
            f"seed={summary['seed']} | "
            f"initial="
            f"{summary['initial_objective']:.2f} | "
            f"final="
            f"{summary['final_objective']:.2f} | "
            f"best="
            f"{summary['best_objective']:.2f} | "
            f"best total value="
            f"{summary['best_total_value']} | "
            f"iterations="
            f"{summary['iterations']} | "
            f"stuck="
            f"{summary['stuck_count']} | "
            f"duration="
            f"{summary['duration_seconds']:.4f}s"
        )

    best_scores = [
        summary["best_objective"]
        for summary in summaries
    ]

    best_total_values = [
        summary["best_total_value"]
        for summary in summaries
    ]

    durations = [
        summary["duration_seconds"]
        for summary in summaries
    ]

    print()

    print(
        "Average best objective:",
        sum(best_scores) / len(best_scores),
    )

    print(
        "Average best total value:",
        sum(best_total_values)
        / len(best_total_values),
    )

    print(
        "Average duration:",
        sum(durations) / len(durations),
    )

    print(
        "Highest best objective:",
        max(best_scores),
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    truck, packages = make_experiment_problem()

    summaries = []

    for run_number, seed in enumerate(
        RUN_SEEDS,
        start=1,
    ):

        print(
            f"Running experiment "
            f"{run_number}/"
            f"{len(RUN_SEEDS)}..."
        )

        initial_state = generate_initial_state(
            truck,
            packages,
            seed=seed,
        )

        result = simulated_annealing(
            initial_state,
            seed=seed,
            objective_mode=OBJECTIVE_MODE,
            **SA_CONFIG,
        )

        summary = save_run(
            run_number,
            seed,
            result,
        )

        summaries.append(summary)

    print_summary(summaries)

    print(
        "\nExperiment files saved to:",
        OUTPUT_DIR,
    )

if __name__ == "__main__":
    main()