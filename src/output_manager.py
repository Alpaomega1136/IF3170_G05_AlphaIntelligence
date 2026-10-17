from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict

from src.core import State, calculate_objective, get_truck_state
from src.simulated_annealing import calculate_sa_objective
from src.visualization import plot_objective_history, save_sa_visualizations, visualize_state

OutputSaver = Callable[[object, Dict[str, Any], str | Path], Path]
OUTPUT_SAVERS: Dict[str, OutputSaver] = {}


def register_output_saver(algorithm_name: str):
    def decorator(func: OutputSaver) -> OutputSaver:
        OUTPUT_SAVERS[algorithm_name] = func
        return func

    return decorator


def state_to_dict(state: State, objective_score: float) -> Dict[str, Any]:
    packages = {package.id: package for package in state.packages}
    if state.additional_trucks:
        trucks = []
        for index in range(len(state.trucks)):
            local = get_truck_state(state, index)
            data = state_to_dict(local, calculate_objective(local))
            data.pop("outside_packages")
            data.pop("objective_score")
            data["truck_index"] = index
            trucks.append(data)
        return {"trucks": trucks, "objective_score": objective_score,
            "total_value": calculate_objective(state),
            "total_weight": sum(packages[package_id].weight for package_id in state.placements),
            "outside_packages": [p.id for p in state.packages if p.id not in state.placements]}

    inside = []

    for package_id, placement in sorted(state.placements.items()):
        package = packages[package_id]

        inside.append({"id": package.id, "width": package.width, "length": package.length,
            "height": package.height, "value": package.value, "weight": package.weight,
            "is_fragile": package.is_fragile, "eta": package.eta, "position": {"x": placement.x,
                "y": placement.y, "z": placement.z}, "orientation": placement.orientation})

    outside = [package.id for package in state.packages if package.id not in state.placements]

    total_weight = sum(packages[package_id].weight for package_id in state.placements)

    return {"truck": {"width": state.truck.width, "length": state.truck.length,
            "height": state.truck.height, "max_capacity": state.truck.max_capacity},
        "objective_score": objective_score, "total_value": calculate_objective(state),
        "total_weight": total_weight, "inside_packages": inside, "outside_packages": outside}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _write_sa_history(run_dir: Path, result) -> None:
    history_path = run_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["iteration", "current_objective", "best_objective", "neighbor_objective",
            "temperature", "delta_e", "acceptance_probability", "accepted"])

        for i in range(result.iterations):
            writer.writerow([i + 1, result.objective_history[i + 1],
                result.best_objective_history[i + 1], result.neighbor_objective_history[i],
                result.temperature_history[i], result.delta_e_history[i],
                result.acceptance_probability_history[i], result.accepted_history[i]])


def _remove_obsolete_sa_outputs(run_dir: Path) -> None:
    obsolete_files = ("temperature_history.png", "best_state.png", "best_objective_history.png")

    for filename in obsolete_files:
        path = run_dir / filename
        if path.exists():
            path.unlink()


@register_output_saver("simulated_annealing")
def save_sa_outputs(result, parameters: Dict[str, Any], input_file: str | Path) -> Path:
    objective_mode = result.metadata["objective_mode"]

    run_name = Path(input_file).stem

    root = Path("output/multi_truck") if result.final_state.additional_trucks else Path("output")
    run_dir = root / "simulated_annealing" / objective_mode / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _remove_obsolete_sa_outputs(run_dir)

    states = {"initial_state": state_to_dict(result.initial_state, calculate_sa_objective(
                result.initial_state, objective_mode)), "final_state": state_to_dict(
            result.final_state, calculate_sa_objective(result.final_state, objective_mode))}

    _write_json(run_dir / "states.json", states)

    summary = {"algorithm": "simulated_annealing", "objective_mode": objective_mode,
        "seed": parameters.get("seed", result.metadata.get("seed")), "configuration": {
            "initial_temperature": result.metadata["initial_temperature"],
            "cooling_rate": result.metadata["cooling_rate"], "min_temperature": result.metadata[
                "min_temperature"], "max_iterations": result.metadata["max_iterations"],
            "stuck_threshold": result.metadata["stuck_threshold"],
            "local_optimum_probe_size": result.metadata["local_optimum_probe_size"]}, "result": {
            "initial_objective": result.metadata["initial_objective"],
            "final_objective": result.metadata["final_objective"],
            "best_objective": result.metadata["best_objective"],
            "initial_total_value": result.metadata["initial_total_value"],
            "final_total_value": result.metadata["final_total_value"],
            "best_total_value": result.metadata["best_total_value"],
            "iterations": result.iterations, "duration_seconds": result.duration,
            "final_temperature": result.metadata["final_temperature"],
            "stop_reason": result.metadata["stop_reason"]}, "sa_statistics": {
            "accepted_moves": result.metadata["accepted_moves"],
            "accepted_worse_moves": result.metadata["accepted_worse_moves"],
            "rejected_moves": result.metadata["rejected_moves"], "stuck_count": result.metadata[
                "stuck_count"], "stuck_iterations": result.metadata["stuck_iterations"]}}

    _write_json(run_dir / "summary.json", summary)

    _write_sa_history(run_dir, result)

    save_sa_visualizations(result, run_dir, show=False)

    return run_dir


@register_output_saver("genetic_algorithm")
def save_ga_outputs(result, parameters: Dict[str, Any], input_file: str | Path) -> Path:
    population_size = result.metadata["population_size"]
    max_iterations = result.metadata["max_iterations"]
    root = Path("output/multi_truck") if result.final_state.additional_trucks else Path("output")
    run_dir = (root / "genetic_algorithm" /
               f"population_{population_size}_iterations_{max_iterations}" / Path(input_file).stem)
    run_dir.mkdir(parents=True, exist_ok=True)

    states = {"initial_state": state_to_dict(result.initial_state, calculate_objective(result.initial_state)),
              "final_state": state_to_dict(result.final_state, calculate_objective(result.final_state))}
    _write_json(run_dir / "states.json", states)
    summary = {"algorithm": "genetic_algorithm", "seed": result.metadata["seed"],
        "configuration": {"population_size": population_size, "max_iterations": max_iterations},
        "operators": {"selection": "tournament", "tournament_size": min(3, population_size),
            "crossover": "uniform", "parent_a_probability": 0.5, "mutation_rate": 0.1,
            "elite_count": 1, "repair": "greedy_bottom_up"},
        "result": {"initial_objective": result.objective_history[0],
            "final_objective": result.objective_history[-1],
            "initial_mean_objective": result.mean_objective_history[0],
            "final_mean_objective": result.mean_objective_history[-1],
            "iterations": result.iterations, "duration_seconds": result.duration,
            "is_valid_final_state": result.metadata["is_valid_final_state"]},
        "duration_scope": "population initialization and evolution; excludes IO and visualization"}
    _write_json(run_dir / "summary.json", summary)

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["generation", "max_objective", "mean_objective"])
        for generation in range(result.iterations + 1):
            writer.writerow([generation, result.objective_history[generation],
                             result.mean_objective_history[generation]])

    visualize_state(result.initial_state, title="Initial State (Best Individual)",
                    output_path=run_dir / "initial_state.png")
    visualize_state(result.final_state, title="Final State (Best Individual)",
                    output_path=run_dir / "final_state.png")
    plot_objective_history(result, output_path=run_dir / "objective_history.png")
    return run_dir


def save_algorithm_outputs(algorithm_name: str, result, parameters: Dict[str, Any],
    input_file: str | Path) -> Path | None:
    saver = OUTPUT_SAVERS.get(algorithm_name)

    if saver is None:
        if result.final_state.additional_trucks and algorithm_name in {
            "steepest_ascent", "stochastic_hill_climbing", "hill_climbing_sideways", "random_restart_hill_climbing"}:
            return save_fleet_hc_outputs(algorithm_name, result, parameters, input_file)
        return None

    return saver(result, parameters, input_file)


def save_fleet_hc_outputs(algorithm_name: str, result, parameters: Dict[str, Any],
    input_file: str | Path) -> Path:
    run_dir = Path("output/multi_truck") / algorithm_name / Path(input_file).stem
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "states.json", {
        "initial_state": state_to_dict(result.initial_state, calculate_objective(result.initial_state)),
        "final_state": state_to_dict(result.final_state, calculate_objective(result.final_state))})
    _write_json(run_dir / "summary.json", {"algorithm": algorithm_name, "parameters": parameters,
        "truck_count": len(result.final_state.trucks), "metadata": result.metadata,
        "result": {"initial_objective": calculate_objective(result.initial_state),
            "final_objective": calculate_objective(result.final_state),
            "iterations": result.iterations, "duration_seconds": result.duration}})
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "objective"])
        writer.writerows(enumerate(result.objective_history))
    visualize_state(result.initial_state, title="Initial Fleet", output_path=run_dir / "initial_state.png")
    visualize_state(result.final_state, title="Final Fleet", output_path=run_dir / "final_state.png")
    plot_objective_history(result, output_path=run_dir / "objective_history.png")
    return run_dir
