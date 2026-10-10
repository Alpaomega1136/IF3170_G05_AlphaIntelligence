from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable

from src.core import (Package, Truck, calculate_objective, generate_initial_state, validate_state)
from src.genetic_algorithm import genetic_algorithm
from src.hill_climbing import (hill_climbing_sideways, random_restart_hill_climbing,
    steepest_ascent, stochastic_hill_climbing)
from src.output_manager import save_algorithm_outputs
from src.parser import load_input
from src.simulated_annealing import (calculate_sa_objective, simulated_annealing)


ALGORITHM_ALIASES = {"steepest": "steepest_ascent", "steepest_ascent": "steepest_ascent",
    "stochastic": "stochastic_hill_climbing",
    "stochastic_hill_climbing": "stochastic_hill_climbing", "sideways": "hill_climbing_sideways",
    "hill_climbing_sideways": "hill_climbing_sideways",
    "random_restart": "random_restart_hill_climbing",
    "random-restart": "random_restart_hill_climbing",
    "random_restart_hill_climbing": "random_restart_hill_climbing", "sa": "simulated_annealing",
    "simulated_annealing": "simulated_annealing", "ga": "genetic_algorithm",
    "genetic_algorithm": "genetic_algorithm"}

ALLOWED_PARAMETERS = {"steepest_ascent": {"seed", "max_iterations", "neighbor_limit"},
    "stochastic_hill_climbing": {"seed", "max_iterations", "neighbor_limit"},
    "hill_climbing_sideways": {"seed", "max_iterations", "neighbor_limit", "max_sideways_moves"},
    "random_restart_hill_climbing": {"seed", "max_restarts", "max_iterations", "neighbor_limit"},
    "simulated_annealing": {"seed", "initial_temperature", "cooling_rate", "min_temperature",
        "max_iterations", "stuck_threshold", "local_optimum_probe_size", "objective_mode"},
    "genetic_algorithm": {"seed", "population_size", "max_iterations"}}


def _normalize_algorithm_name(name: str) -> str:
    normalized = ALGORITHM_ALIASES.get(name)
    if normalized is None:
        supported = ", ".join(sorted(ALLOWED_PARAMETERS))
        raise ValueError(f"Unknown algorithm '{name}'. Supported algorithms: {supported}")
    return normalized


def _validate_parameters(algorithm_name: str, parameters: Dict[str, Any]) -> None:
    allowed = ALLOWED_PARAMETERS[algorithm_name]
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown parameter(s) for {algorithm_name}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}")


def _initial_seed(parameters: Dict[str, Any]) -> int | None:
    seed = parameters.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("algorithm.parameters.seed must be an integer or null")
    return seed


def run_algorithm(truck: Truck, packages: Iterable[Package], algorithm_name: str,
    parameters: Dict[str, Any]):
    algorithm_name = _normalize_algorithm_name(algorithm_name)
    _validate_parameters(algorithm_name, parameters)

    params = dict(parameters)
    seed = _initial_seed(params)
    package_tuple = tuple(packages)

    if algorithm_name == "random_restart_hill_climbing":
        return algorithm_name, random_restart_hill_climbing(truck, package_tuple, **params)
    if algorithm_name == "genetic_algorithm":
        return algorithm_name, genetic_algorithm(truck, package_tuple, **params)

    initial_state = generate_initial_state(truck, package_tuple, seed=seed)

    if algorithm_name == "steepest_ascent":
        result = steepest_ascent(initial_state, **params)
    elif algorithm_name == "stochastic_hill_climbing":
        result = stochastic_hill_climbing(initial_state, **params)
    elif algorithm_name == "hill_climbing_sideways":
        result = hill_climbing_sideways(initial_state, **params)
    else:
        result = simulated_annealing(initial_state, **params)

    return algorithm_name, result


def print_result(algorithm_name: str, result) -> None:
    valid, reasons = validate_state(result.final_state)

    print(f"Algorithm: {algorithm_name}")

    if algorithm_name == "simulated_annealing":
        objective_mode = result.metadata["objective_mode"]
        print(f"Objective mode: {objective_mode}")
        print("Initial objective: "
            f"{calculate_sa_objective(result.initial_state, objective_mode):.2f}")
        print(f"Initial total value: {calculate_objective(result.initial_state)}")
        print("Final objective: "
            f"{calculate_sa_objective(result.final_state, objective_mode):.2f}")
        print(f"Final total value: {calculate_objective(result.final_state)}")
        print("Best objective: " f"{calculate_sa_objective(result.best_state, objective_mode):.2f}")
        print(f"Best total value: {calculate_objective(result.best_state)}")
        print(f"Stuck count: {result.metadata['stuck_count']}")
        print(f"Stop reason: {result.metadata['stop_reason']}")
    else:
        print(f"Initial objective: {calculate_objective(result.initial_state)}")
        print(f"Final objective: {calculate_objective(result.final_state)}")

        if algorithm_name == "hill_climbing_sideways":
            print(f"Sideways moves: {result.metadata['sideways_moves']}")
            print(f"Maximum sideways moves: {result.metadata['max_sideways_moves']}")
        elif algorithm_name == "random_restart_hill_climbing":
            print(f"Restart count: {result.metadata['restart_count']}")
            print("Iterations per restart: " f"{result.metadata['iterations_per_restart']}")
        elif algorithm_name == "genetic_algorithm":
            print(f"Population size: {result.metadata['population_size']}")
            print(f"Initial population mean: {result.mean_objective_history[0]:.2f}")
            print(f"Final population mean: {result.mean_objective_history[-1]:.2f}")

    print(f"Iterations: {result.iterations}")
    print(f"Duration: {result.duration:.6f}s")
    if reasons:
        print(f"Invalid reasons: {', '.join(reasons)}")
    print(f"Placed packages: {sorted(result.final_state.placements)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local-search algorithms from a JSON input file.")
    parser.add_argument("input_file", nargs="?", help="Path to input JSON")
    args = parser.parse_args()

    try:
        program_input = load_input(args.input_file)
        algorithm_name, result = run_algorithm(program_input.truck, program_input.packages,
            program_input.algorithm_name, program_input.parameters)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    print_result(algorithm_name, result)

    output_dir = save_algorithm_outputs(algorithm_name, result, program_input.parameters,
        args.input_file)

    if output_dir is not None:
        print(f"Result files saved to: {output_dir}")


if __name__ == "__main__":
    main()
