from __future__ import annotations

import random
from typing import Iterable, List, Tuple

from src.core import (
    ORIENTATIONS,
    Package,
    State,
    Truck,
    calculate_objective,
    generate_initial_state,
    generate_neighbor,
    is_valid_state,
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
    return [generate_initial_state(truck, package_tuple, seed=rng.randrange(1_000_000_000)) for _ in range(population_size)]


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
    return (fitness_scores, individuals[best_index].copy(), sum(fitness_scores) / len(fitness_scores))


def tournament_selection(population: List[State], fitness_scores: List[int], rng: random.Random) -> State:
    if not population or len(population) != len(fitness_scores):
        raise ValueError("population and fitness_scores must have the same non-zero length")
    participants = rng.sample(range(len(population)), min(3, len(population)))
    return population[max(participants, key=fitness_scores.__getitem__)]


def repair_state(state: State) -> State:
    child = State(state.truck, state.packages, {})
    selected = [(package.id, state.placements[package.id]) for package in state.packages
                if package.id in state.placements]
    for package_id, placement in sorted(selected, key=lambda item: item[1].z):
        if placement.orientation not in ORIENTATIONS:
            continue
        candidate = child.copy()
        candidate.placements[package_id] = placement
        if is_valid_state(candidate):
            child = candidate
    return child


def uniform_crossover(parent_a: State, parent_b: State, rng: random.Random) -> State:
    if parent_a.truck != parent_b.truck or parent_a.packages != parent_b.packages:
        raise ValueError("parents must use the same truck and ordered packages")
    child = State(parent_a.truck, parent_a.packages, {})
    for package in child.packages:
        parent = parent_a if rng.random() < 0.5 else parent_b
        placement = parent.placements.get(package.id)
        if placement is not None:
            child.placements[package.id] = placement
    return repair_state(child)


def mutate(state: State, rng: random.Random) -> State:
    if rng.random() < 0.1:
        neighbor = generate_neighbor(state, seed=rng.randrange(1_000_000_000))
        if neighbor is not None:
            return neighbor
    return state.copy()


def next_generation(population: List[State], rng: random.Random) -> List[State]:
    fitness_scores, elite, _ = evaluate_population(population)
    children = [elite]
    while len(children) < len(population):
        parent_a = tournament_selection(population, fitness_scores, rng)
        parent_b = tournament_selection(population, fitness_scores, rng)
        child = uniform_crossover(parent_a, parent_b, rng)
        children.append(mutate(child, rng))
    return children