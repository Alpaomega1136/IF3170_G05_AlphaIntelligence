from __future__ import annotations

import argparse

from core import Package, Truck, calculate_objective, generate_initial_state, validate_state
from hill_climbing import (
    hill_climbing_sideways,
    random_restart_hill_climbing,
    steepest_ascent,
    stochastic_hill_climbing,
)


def sample_problem() -> tuple[Truck, list[Package]]:
    truck = Truck(width=6, length=6, height=5, max_capacity=40)
    packages = [
        Package("A", 2, 2, 1, value=20, weight=5, is_fragile=False, eta=1),
        Package("B", 3, 2, 1, value=30, weight=8, is_fragile=False, eta=2),
        Package("C", 2, 2, 2, value=25, weight=7, is_fragile=True, eta=3),
        Package("D", 1, 3, 1, value=15, weight=4, is_fragile=False, eta=2),
        Package("E", 2, 1, 3, value=18, weight=6, is_fragile=False, eta=4),
    ]
    return truck, packages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal IF3170 hill-climbing demo.")
    parser.add_argument(
        "--algorithm",
        choices=("steepest", "stochastic", "sideways", "random-restart"),
        default="steepest",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()

    truck, packages = sample_problem()
    initial_state = generate_initial_state(truck, packages, seed=args.seed)

    if args.algorithm == "steepest":
        result = steepest_ascent(initial_state, max_iterations=args.max_iterations, seed=args.seed)
    elif args.algorithm == "stochastic":
        result = stochastic_hill_climbing(initial_state, max_iterations=args.max_iterations, seed=args.seed)
    elif args.algorithm == "sideways":
        result = hill_climbing_sideways(initial_state, max_iterations=args.max_iterations, seed=args.seed)
    else:
        result = random_restart_hill_climbing(
            truck,
            packages,
            max_restarts=5,
            max_iterations=args.max_iterations,
            seed=args.seed,
        )

    valid, reasons = validate_state(result.final_state)
    print(f"Algorithm: {result.metadata['algorithm']}")
    print(f"Initial objective: {calculate_objective(result.initial_state)}")
    print(f"Final objective: {calculate_objective(result.final_state)}")
    print(f"Iterations: {result.iterations}")
    print(f"Duration: {result.duration:.6f}s")
    print(f"Valid final state: {valid}")
    if reasons:
        print(f"Invalid reasons: {', '.join(reasons)}")
    print(f"Placed packages: {sorted(result.final_state.placements)}")


if __name__ == "__main__":
    main()
