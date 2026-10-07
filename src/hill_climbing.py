from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Dict, Iterable, List, Optional

from src.core import (
    Package,
    State,
    Truck,
    calculate_objective,
    generate_initial_state,
    generate_neighbors,
    is_valid_state,
)


@dataclass
class SearchResult:
    initial_state: State
    final_state: State
    objective_history: List[int]
    iterations: int
    duration: float
    metadata: Dict[str, object]


def steepest_ascent(
    initial_state: State,
    max_iterations: int = 100,
    neighbor_limit: int = 200,
    seed: Optional[int] = None,
) -> SearchResult:
    start = time.perf_counter()
    rng = random.Random(seed)
    current = initial_state.copy()
    history = [calculate_objective(current)]

    for _ in range(max_iterations):
        neighbor_seed = rng.randrange(1_000_000_000)
        neighbors = generate_neighbors(current, limit=neighbor_limit, seed=neighbor_seed)
        if not neighbors:
            break

        best_neighbor = None
        best_score = history[-1]

        for neighbor in neighbors:
            neighbor_score = calculate_objective(neighbor)
            if neighbor_score > best_score:
                best_neighbor = neighbor
                best_score = neighbor_score

        if best_neighbor is None:
            break

        current = best_neighbor
        history.append(best_score)

    return _result(
        "steepest_ascent",
        initial_state,
        current,
        history,
        iteration_count=len(history) - 1,
        start=start,
        seed=seed,
    )


def stochastic_hill_climbing(
    initial_state: State,
    max_iterations: int = 100,
    neighbor_limit: int = 200,
    seed: Optional[int] = None,
) -> SearchResult:
    start = time.perf_counter()
    rng = random.Random(seed)
    current = initial_state.copy()
    history = [calculate_objective(current)]

    for _ in range(max_iterations):
        neighbor_seed = rng.randrange(1_000_000_000)
        neighbors = generate_neighbors(current, limit=neighbor_limit, seed=neighbor_seed)
        improving = []

        for neighbor in neighbors:
            if calculate_objective(neighbor) > history[-1]:
                improving.append(neighbor)

        if not improving:
            break

        current = rng.choice(improving)
        history.append(calculate_objective(current))

    return _result(
        "stochastic_hill_climbing",
        initial_state,
        current,
        history,
        iteration_count=len(history) - 1,
        start=start,
        seed=seed,
    )


def hill_climbing_sideways(
    initial_state: State,
    max_iterations: int = 100,
    neighbor_limit: int = 200,
    max_sideways_moves: int = 10,
    seed: Optional[int] = None,
) -> SearchResult:
    start = time.perf_counter()
    rng = random.Random(seed)
    current = initial_state.copy()
    history = [calculate_objective(current)]
    sideways_moves = 0

    for _ in range(max_iterations):
        neighbor_seed = rng.randrange(1_000_000_000)
        neighbors = generate_neighbors(current, limit=neighbor_limit, seed=neighbor_seed)
        if not neighbors:
            break

        best_score = None
        choices = []

        for neighbor in neighbors:
            score = calculate_objective(neighbor)
            if best_score is None or score > best_score:
                best_score = score
                choices = [neighbor]
            elif score == best_score:
                choices.append(neighbor)

        if best_score is None:
            break

        if best_score > history[-1]:
            current = rng.choice(choices)
            history.append(best_score)
            sideways_moves = 0
        elif best_score == history[-1] and sideways_moves < max_sideways_moves:
            current = rng.choice(choices)
            history.append(best_score)
            sideways_moves += 1
        else:
            break

    result = _result(
        "hill_climbing_sideways",
        initial_state,
        current,
        history,
        iteration_count=len(history) - 1,
        start=start,
        seed=seed,
    )
    result.metadata["sideways_moves"] = sideways_moves
    result.metadata["max_sideways_moves"] = max_sideways_moves
    return result


def random_restart_hill_climbing(
    truck: Truck,
    packages: Iterable[Package],
    max_restarts: int = 5,
    max_iterations: int = 100,
    neighbor_limit: int = 200,
    seed: Optional[int] = None,
) -> SearchResult:
    start = time.perf_counter()
    rng = random.Random(seed)
    package_tuple = tuple(packages)
    best_result: Optional[SearchResult] = None
    iterations_per_restart: List[int] = []
    objectives_per_restart: List[int] = []

    for _ in range(max_restarts + 1):
        initial_seed = rng.randrange(1_000_000_000)
        search_seed = rng.randrange(1_000_000_000)
        initial = generate_initial_state(truck, package_tuple, seed=initial_seed)
        result = steepest_ascent(
            initial,
            max_iterations=max_iterations,
            neighbor_limit=neighbor_limit,
            seed=search_seed,
        )
        iterations_per_restart.append(result.iterations)
        objectives_per_restart.append(calculate_objective(result.final_state))

        current_score = calculate_objective(result.final_state)
        if best_result is None:
            best_result = result
            continue

        best_score = calculate_objective(best_result.final_state)
        if current_score > best_score:
            best_result = result

    assert best_result is not None
    history = objectives_per_restart
    final = best_result.final_state
    initial_state = best_result.initial_state
    duration = time.perf_counter() - start
    return SearchResult(
        initial_state=initial_state,
        final_state=final,
        objective_history=history,
        iterations=sum(iterations_per_restart),
        duration=duration,
        metadata={
            "algorithm": "random_restart_hill_climbing",
            "final_objective": calculate_objective(final),
            "is_valid_final_state": is_valid_state(final),
            "restart_count": max_restarts,
            "max_restarts": max_restarts,
            "iterations_per_restart": iterations_per_restart,
            "objectives_per_restart": objectives_per_restart,
            "seed": seed,
        },
    )


def _result(
    algorithm: str,
    initial_state: State,
    final_state: State,
    history: List[int],
    iteration_count: int,
    start: float,
    seed: Optional[int],
) -> SearchResult:
    return SearchResult(
        initial_state=initial_state.copy(),
        final_state=final_state.copy(),
        objective_history=history,
        iterations=iteration_count,
        duration=time.perf_counter() - start,
        metadata={
            "algorithm": algorithm,
            "final_objective": calculate_objective(final_state),
            "is_valid_final_state": is_valid_state(final_state),
            "seed": seed,
        },
    )
