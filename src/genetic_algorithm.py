from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

from src.core import (
    ORIENTATIONS,
    Package,
    PlacedPackage,
    State,
    Truck,
    calculate_objective,
    generate_initial_state,
    is_valid_state,
    rotate_orientation,
    validate_state,
)


MUTATION_RATE = 0.1


@dataclass(frozen=True)
class ChromosomeCodec:
    trucks: Tuple[Truck, ...]
    packages: Tuple[Package, ...]

    def __post_init__(self) -> None:
        if not self.trucks:
            raise ValueError("At least one truck is required")

    @property
    def field_limits(self) -> Tuple[int, ...]:
        return (len(self.trucks) + 1, max(t.width for t in self.trucks),
                max(t.length for t in self.trucks), max(t.height for t in self.trucks),
                len(ORIENTATIONS))

    @property
    def field_widths(self) -> Tuple[int, ...]:
        return tuple(len(str(limit - 1)) for limit in self.field_limits)

    @property
    def chromosome_length(self) -> int:
        return len(self.packages) * sum(self.field_widths)

    def encode(self, state: State) -> str:
        """Encode a state; outside packages start with zero-valued latent fields."""
        if state.trucks != self.trucks or state.packages != self.packages:
            raise ValueError("state must use the codec's trucks and ordered packages")
        valid, reasons = validate_state(state)
        if not valid:
            raise ValueError(f"Cannot encode invalid state: {', '.join(reasons)}")
        genes = []
        widths = self.field_widths
        for package in self.packages:
            placement = state.placements.get(package.id)
            values = ((0, 0, 0, 0, 0) if placement is None else
                      (placement.truck_index + 1, placement.x, placement.y, placement.z,
                       ORIENTATIONS.index(placement.orientation)))
            genes.append("".join(f"{value:0{width}d}" for value, width in zip(values, widths)))
        return "".join(genes)

    def decode(self, chromosome: str) -> State:
        """Decode fields; geometric constraints are checked separately before evaluation."""
        if (not isinstance(chromosome, str) or len(chromosome) != self.chromosome_length
                or any(character not in "0123456789" for character in chromosome)):
            raise ValueError("chromosome must be a decimal string of the expected length")
        state = State(self.trucks[0], self.packages, {}, self.trucks[1:])
        offset = 0
        widths, limits = self.field_widths, self.field_limits
        for package in self.packages:
            values = []
            for width, limit in zip(widths, limits):
                value = int(chromosome[offset:offset + width])
                if value >= limit:
                    raise ValueError("chromosome field is outside its domain")
                values.append(value)
                offset += width
            truck_code, x, y, z, orientation = values
            if truck_code:
                state.placements[package.id] = PlacedPackage(
                    x, y, z, ORIENTATIONS[orientation], truck_code - 1)
        return state


@dataclass
class GeneticAlgorithmResult:
    initial_state: State
    final_state: State
    objective_history: List[int]
    mean_objective_history: List[float]
    iterations: int
    duration: float
    metadata: Dict[str, object]


def initialize_population(
    codec: ChromosomeCodec,
    population_size: int,
    rng: random.Random,
) -> List[str]:
    if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < 2:
        raise ValueError("population_size must be an integer >= 2")

    return [codec.encode(generate_initial_state(codec.trucks, codec.packages,
            seed=rng.randrange(1_000_000_000))) for _ in range(population_size)]


def evaluate_population(population: Iterable[str], codec: ChromosomeCodec) -> Tuple[List[int], str, float]:
    individuals = list(population)
    if not individuals:
        raise ValueError("population must not be empty")

    fitness_scores = []
    for index, individual in enumerate(individuals):
        state = codec.decode(individual)
        valid, reasons = validate_state(state)
        if not valid:
            raise ValueError(f"population[{index}] is invalid: {', '.join(reasons)}")
        fitness_scores.append(calculate_objective(state))

    best_index = fitness_scores.index(max(fitness_scores))
    return (fitness_scores, individuals[best_index], sum(fitness_scores) / len(fitness_scores))


def roulette_wheel_selection(population: List[str], fitness_scores: List[int], rng: random.Random) -> str:
    if not population or len(population) != len(fitness_scores):
        raise ValueError("population and fitness_scores must have the same non-zero length")
    if any(score < 0 for score in fitness_scores):
        raise ValueError("fitness scores must be non-negative")
    if not any(fitness_scores):
        return rng.choice(population)
    return rng.choices(population, weights=fitness_scores, k=1)[0]


def repair_state(state: State) -> State:
    child = State(state.truck, state.packages, {}, state.additional_trucks)
    selected = [(package.id, state.placements[package.id]) for package in state.packages
                if package.id in state.placements]
    # shortcut: drops infeasible inherited genes, revisit reinsertion if solution quality stalls.
    for package_id, placement in sorted(selected, key=lambda item: item[1].z):
        if placement.orientation not in ORIENTATIONS:
            continue
        candidate = child.copy()
        candidate.placements[package_id] = placement
        if is_valid_state(candidate):
            child = candidate
    return child


def repair_chromosome(chromosome: str, codec: ChromosomeCodec) -> str:
    repaired = repair_state(codec.decode(chromosome))
    widths = codec.field_widths
    gene_width = sum(widths)
    genes = []
    for index, package in enumerate(codec.packages):
        start = index * gene_width
        gene = chromosome[start:start + gene_width]
        if package.id not in repaired.placements:
            # Preserve candidate coordinates/orientation so later mutations can activate them.
            gene = "0" * widths[0] + gene[widths[0]:]
        genes.append(gene)
    return "".join(genes)


def one_point_crossover(parent_a: str, parent_b: str, codec: ChromosomeCodec, rng: random.Random) -> str:
    if len(parent_a) != codec.chromosome_length or len(parent_b) != codec.chromosome_length:
        raise ValueError("parents must have the expected chromosome length")
    widths = codec.field_widths * len(codec.packages)
    if len(widths) < 2:
        return parent_a
    # Cut between variables so a multi-digit coordinate remains a complete field.
    cut = sum(widths[:rng.randrange(1, len(widths))])
    return parent_a[:cut] + parent_b[cut:]


def mutate(chromosome: str, codec: ChromosomeCodec, rng: random.Random) -> str:
    if not chromosome or rng.random() >= MUTATION_RATE:
        return chromosome
    widths = codec.field_widths * len(codec.packages)
    limits = codec.field_limits * len(codec.packages)
    index = rng.choice([i for i, limit in enumerate(limits) if limit > 1])
    start = sum(widths[:index])
    end = start + widths[index]
    old_value = int(chromosome[start:end])
    if index % len(codec.field_widths) == 4:
        orientation = rotate_orientation(ORIENTATIONS[old_value], rng.choice(("x", "y", "z")))
        new_value = ORIENTATIONS.index(orientation)
    else:
        new_value = rng.randrange(limits[index] - 1)
        if new_value >= old_value:
            new_value += 1
    return chromosome[:start] + f"{new_value:0{widths[index]}d}" + chromosome[end:]


def next_generation(population: List[str], codec: ChromosomeCodec, rng: random.Random) -> List[str]:
    fitness_scores, _, _ = evaluate_population(population, codec)
    children = []
    while len(children) < len(population):
        parent_a = roulette_wheel_selection(population, fitness_scores, rng)
        parent_b = roulette_wheel_selection(population, fitness_scores, rng)
        child = one_point_crossover(parent_a, parent_b, codec, rng)
        child = mutate(child, codec, rng)
        children.append(repair_chromosome(child, codec))
    return children


def genetic_algorithm(
    truck: Truck | Tuple[Truck, ...],
    packages: Iterable[Package],
    population_size: int = 50,
    max_iterations: int = 100,
    seed: Optional[int] = None,
) -> GeneticAlgorithmResult:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be an integer >= 0")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or null")

    start = time.perf_counter()
    rng = random.Random(seed)
    codec = ChromosomeCodec(truck if isinstance(truck, tuple) else (truck,), tuple(packages))
    population = initialize_population(codec, population_size, rng)
    fitness_scores, best, mean = evaluate_population(population, codec)
    initial_chromosome = best
    initial_state = codec.decode(best)
    objective_history = [max(fitness_scores)]
    mean_objective_history = [mean]

    for _ in range(max_iterations):
        population = next_generation(population, codec, rng)
        fitness_scores, best, mean = evaluate_population(population, codec)
        objective_history.append(max(fitness_scores))
        mean_objective_history.append(mean)

    final_state = codec.decode(best)
    return GeneticAlgorithmResult(
        initial_state=initial_state,
        final_state=final_state,
        objective_history=objective_history,
        mean_objective_history=mean_objective_history,
        iterations=max_iterations,
        duration=time.perf_counter() - start,
        metadata={
            "algorithm": "genetic_algorithm",
            "seed": seed,
            "population_size": population_size,
            "max_iterations": max_iterations,
            "representation": "fixed_width_decimal_string",
            "gene_fields": ["truck_code", "x", "y", "z", "orientation"],
            "field_widths": list(codec.field_widths),
            "orientation_order": list(ORIENTATIONS),
            "package_order": [package.id for package in codec.packages],
            "initial_chromosome": initial_chromosome,
            "final_chromosome": best,
            "operators": {"selection": "roulette_wheel", "crossover": "one_point",
                "crossover_boundary": "variable", "mutation": "single_variable",
                "orientation_mutation": "rotate_90_degrees",
                "mutation_rate": MUTATION_RATE, "elite_count": 0, "repair": "greedy_bottom_up"},
            "is_valid_final_state": is_valid_state(final_state),
        },
    )
