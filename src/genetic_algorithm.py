from __future__ import annotations

import random
from typing import Iterable, List, Tuple

from src.core import (
    Package,
    State,
    Truck,
    calculate_objective,
    generate_initial_state,
    validate_state,
)


def initialize_population(
    truck: Truck,
    packages: Iterable[Package],
    population_size: int,
    rng: random.Random,
) -> List[State]:
    if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < 2:
        raise ValueError("population_size must be an integer >= 2")

    package_tuple = tuple(packages)
    return [
        generate_initial_state(truck, package_tuple, seed=rng.randrange(1_000_000_000))
        for _ in range(population_size)
    ]


def evaluate_population(population: Iterable[State]) -> Tuple[List[int], State, float]:
    individuals = list(population)
    if not individuals:
        raise ValueError("population must not be empty")

    fitness_scores = []
    for index, individual in enumerate(individuals):
        valid, reasons = validate_state(individual)
        if not valid:
            raise ValueError(f"population[{index}] is invalid: {', '.join(reasons)}")
        fitness_scores.append(calculate_objective(individual))

    best_index = fitness_scores.index(max(fitness_scores))
    return (
        fitness_scores,
        individuals[best_index].copy(),
        sum(fitness_scores) / len(fitness_scores),
    )