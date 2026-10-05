from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Dict, List, Optional
from src.core import (
    State,
    calculate_objective,
    generate_neighbor,
    is_valid_state,
)


@dataclass
class SimulatedAnnealingResult:
    initial_state: State
    final_state: State
    best_state: State
    objective_history: List[int]
    best_objective_history: List[int]
    temperature_history: List[float]
    acceptance_probability_history: List[float]
    iterations: int
    duration: float
    metadata: Dict[str, object]


def simulated_annealing(
    initial_state: State,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 1e-3,
    max_iterations: int = 1000,
    seed: Optional[int] = None,
) -> SimulatedAnnealingResult:
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be > 0")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1")
    if min_temperature < 0:
        raise ValueError("min_temperature must be >= 0")
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")
    if not is_valid_state(initial_state):
        raise ValueError("initial_state must satisfy all constraints")

    start = time.perf_counter()
    rng = random.Random(seed)

    current = initial_state.copy()
    best = current.copy()

    current_score = calculate_objective(current)
    best_score = current_score
    temperature = initial_temperature

    objective_history: List[int] = [current_score]
    best_objective_history: List[int] = [best_score]
    temperature_history: List[float] = []
    acceptance_probability_history: List[float] = []

    accepted_moves = 0
    accepted_worse_moves = 0
    rejected_moves = 0
    iterations = 0

    for _ in range(max_iterations):
        if temperature <= min_temperature:
            break

        neighbor_seed = rng.randrange(1_000_000_000)
        neighbor = generate_neighbor(current, seed=neighbor_seed)

        if neighbor is None:
            break

        neighbor_score = calculate_objective(neighbor)
        delta_e = neighbor_score - current_score

        if delta_e >= 0:
            acceptance_probability = 1.0
            accept = True
        else:
            acceptance_probability = math.exp(delta_e / temperature)
            accept = rng.random() < acceptance_probability

        temperature_history.append(temperature)
        acceptance_probability_history.append(acceptance_probability)

        if accept:
            current = neighbor
            current_score = neighbor_score
            accepted_moves += 1
            if delta_e < 0:
                accepted_worse_moves += 1
        else:
            rejected_moves += 1

        if current_score > best_score:
            best = current.copy()
            best_score = current_score

        objective_history.append(current_score)
        best_objective_history.append(best_score)

        temperature *= cooling_rate
        iterations += 1

    return SimulatedAnnealingResult(
        initial_state=initial_state.copy(),
        final_state=current.copy(),
        best_state=best.copy(),
        objective_history=objective_history,
        best_objective_history=best_objective_history,
        temperature_history=temperature_history,
        acceptance_probability_history=acceptance_probability_history,
        iterations=iterations,
        duration=time.perf_counter() - start,
        metadata={
            "algorithm": "simulated_annealing",
            "final_objective": calculate_objective(current),
            "best_objective": best_score,
            "is_valid_final_state": is_valid_state(current),
            "is_valid_best_state": is_valid_state(best),
            "accepted_moves": accepted_moves,
            "accepted_worse_moves": accepted_worse_moves,
            "rejected_moves": rejected_moves,
            "initial_temperature": initial_temperature,
            "cooling_rate": cooling_rate,
            "min_temperature": min_temperature,
            "max_iterations": max_iterations,
            "seed": seed,
        },
    )
