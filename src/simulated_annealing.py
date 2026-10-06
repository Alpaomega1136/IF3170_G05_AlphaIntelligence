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
    generate_neighbors,
    is_valid_state,
)

OBJECTIVE_VALUE = "value"
OBJECTIVE_ETA_WEIGHTED = "eta_weighted"

def calculate_eta_weighted_objective(state: State) -> float:
    packages = {
        package.id: package
        for package in state.packages
    }
    total_score = 0.0
    for package_id in state.placements:
        package = packages.get(package_id)
        if package is None: continue
        if package.eta < 0: raise ValueError("Package ETA must be >= 0")
        urgency_multiplier = (1.0 + (1.0 / (package.eta + 1)))
        total_score += (package.value * urgency_multiplier)
        return total_score


def calculate_sa_objective(
        state: State,
        objective_mode: str = OBJECTIVE_VALUE,
) -> float:
    if objective_mode == OBJECTIVE_VALUE:
        return float(calculate_objective(state))
    if objective_mode == OBJECTIVE_ETA_WEIGHTED:
        return calculate_eta_weighted_objective(state)
    raise ValueError("objective_mode must be 'value' or 'eta_weighted'")

@dataclass
class SimulatedAnnealingResult:
    initial_state: State
    final_state: State
    best_state: State

    objective_history: List[float]
    best_objective_history: List[float]
    neighbor_objective_history: List[float]

    temperature_history: List[float]
    delta_e_history: List[float]
    acceptance_probability_history: List[float]
    accepted_history: List[float]

    iterations: int
    duration: float
    metadata: Dict[str, object]

def simulated_annealing(initial_state: State, initial_temperature: float = 100.0, cooling_rate: float = 0.95, min_temperature: float = 1e-3, max_iterations: int = 1000, seed: Optional[int] = None, stuck_threshold: int = 50, local_optimum_probe_size: int = 50, objective_mode: str = OBJECTIVE_VALUE) -> SimulatedAnnealingResult:
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be > 0")
    if not 0<cooling_rate<1:
        raise ValueError("cooling_rate must be between 0 and 1")
    if min_temperature <0:
        raise ValueError("min_temperature must be >= 0")
    if max_iterations <0:
        raise ValueError("max_iterations must be >= 0")
    if stuck_threshold <= 0:
        raise ValueError("stuck_threshold must be > 0")
    if local_optimum_probe_size <= 0:
        raise ValueError("local_optimum_probe_size must be > 0")
    if objective_mode not in (OBJECTIVE_VALUE, OBJECTIVE_ETA_WEIGHTED):
        raise ValueError("objective_mode must be 'value' or 'eta_weighted'")
    if not is_valid_state(initial_state):
        raise ValueError("initial_state must satisfy all constraints")

    start = time.perf_counter()
    rng = random.Random(seed)

    current = initial_state.copy()
    best = current.copy()
    current_score = calculate_sa_objective(current, objective_mode)

    best_score = current_score
    temperature = initial_temperature

    objective_history: List[float] = [current_score]
    best_objective_history: List[float] = [best_score]
    neighbor_objective_history: List[float] = []
    temperature_history: List[float] = []
    delta_e_history: List[float] = []
    acceptance_probability_history: List[float] = []
    accepted_history: List[bool] = []

    accepted_moves = 0
    accepted_better_or_equal_moves = 0
    accepted_worse_moves = 0
    rejected_moves = 0

    no_improvement_streak = 0
    longest_no_improvement_streak = 0
    stuck_count = 0
    stuck_iterations: List[int] = []
    inside_stuck_episode = False

    iterations = 0
    stop_reason = "max_iterations"

    for iteration in range(max_iterations):
        if temperature <= min_temperature:
            stop_reason = "min_temperature"
            break

        neighbor_seed = rng.randrange(1_000_000_000)
        neighbor = generate_neighbor(current, seed=neighbor_seed,)
        if neighbor is None:
            stop_reason = "no_valid_neighbor"
            break

        neighbor_score = calculate_sa_objective(neighbor, objective_mode,)
        delta_e = neighbor_score - current_score
        if delta_e >= 0:
            acceptance_probability = 1.0
            accept = True
        else:
            acceptance_probability = math.exp(delta_e / temperature)
            accept = rng.random() < acceptance_probability

        neighbor_objective_history.append(neighbor_score)
        temperature_history.append(temperature)
        delta_e_history.append(delta_e)
        acceptance_probability_history.append(acceptance_probability)
        accepted_history.append(accept)

        if accept:
            current = neighbor
            current_score = neighbor_score
            accepted_moves += 1

            if delta_e < 0: accepted_worse_moves += 1
            else: accepted_better_or_equal_moves += 1
        else:
            rejected_moves += 1

        if current_score > best_score:
            best = current.copy()
            best_score = current_score
            no_improvement_streak = 0
            inside_stuck_episode = False
        else:
            no_improvement_streak += 1
        longest_no_improvement_streak = max(longest_no_improvement_streak, no_improvement_streak,)

        if (no_improvement_streak >= stuck_threshold and not inside_stuck_episode):
            probe_seed = rng. randrange(1_000_000_000)
            probe_neighbors = generate_neighbors(current, limit=local_optimum_probe_size, seed=probe_seed)
            has_better_neighbor = any(calculate_sa_objective(candidate, objective_mode,) > current_score for candidate in probe_neighbors)

            if (probe_neighbors and not has_better_neighbor):
                stuck_count += 1
                stuck_iterations.append(iteration+1)
                inside_stuck_episode = True

        objective_history.append(current_score)
        best_objective_history.append(best_score)

        temperature *= cooling_rate
        iterations += 1

    duration = time.perf_counter() - start
    initial_total_value = calculate_objective(initial_state)
    final_total_value = calculate_objective(current)
    best_total_value = calculate_objective(best)

    if objective_mode == OBJECTIVE_VALUE:
        objective_formula = "sum(value)"
    else:
        objective_formula = "sum(value * (1 + 1 / (ETA + 1)))"

    return SimulatedAnnealingResult(
        initial_state=initial_state.copy(),
        final_state=current.copy(),
        best_state=best.copy(),
        objective_history=objective_history,
        best_objective_history=(best_objective_history),
        neighbor_objective_history=(neighbor_objective_history),
        temperature_history=(temperature_history),
        delta_e_history=delta_e_history,
        acceptance_probability_history=(acceptance_probability_history),
        accepted_history=accepted_history,
        iterations=iterations,
        duration=duration,

        metadata={
            "algoritma": "simulated_annealing",
            "objective_mode": objective_mode,
            "objective_formula": objective_formula,
            "initial_objective": calculate_sa_objective(initial_state, objective_mode,),
            "final_objective": calculate_sa_objective(current, objective_mode,),
            "best_objective": best_score,
            "initial_total_value": initial_total_value,
            "final_total_value": final_total_value,
            "best_total_value": best_total_value,
            "is_valid_final_state": is_valid_state(current),
            "is_valid_best_state": is_valid_state(best),
            "accepted_moves": accepted_moves,
            "accepted_better_or_equal_moves": accepted_better_or_equal_moves,
            "accepted_worse_moves": accepted_worse_moves,
            "rejected_moves": rejected_moves,
            "stuck_count": stuck_count,
            "stuck_iterations": stuck_iterations,
            "longest_no_improvement_streak": longest_no_improvement_streak,
            "stuck_threshold": stuck_threshold,
            "local_optimum_probe_size": local_optimum_probe_size,
            "initial_temperature": initial_temperature,
            "final_temperature": temperature,
            "cooling_rate": cooling_rate,
            "min_temperature": min_temperature,
            "max_iterations": max_iterations,
            "stop_reason": stop_reason,
            "seed": seed,
        },
    )