from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Dict, Iterable, List, Optional, Tuple


Orientation = str
Position = Tuple[int, int, int]  # (x, y, z)
Dimensions = Tuple[int, int, int]  # (width, length, height)

HORIZONTAL = "Horizontal"
VERTICAL = "Vertical"
ORIENTATIONS: Tuple[Orientation, Orientation] = (HORIZONTAL, VERTICAL)

# Data Models

@dataclass(frozen=True)
class Package:
    id: str
    width: int
    length: int
    height: int
    value: int
    weight: int
    is_fragile: bool
    eta: int


@dataclass(frozen=True)
class Truck:
    width: int
    length: int
    height: int
    max_capacity: int


@dataclass(frozen=True)
class PlacedPackage:
    x: int
    y: int
    z: int
    orientation: Orientation = HORIZONTAL

    @property
    def position(self) -> Position:
        return (self.x, self.y, self.z)


@dataclass
class State:
    """A full solution state.

    Coordinates are integer (x, y, z) left-bottom-back corners:
    x = width axis, y = length/depth axis, z = height axis.
    Packages absent from placements are outside the truck.
    """

    truck: Truck
    packages: Tuple[Package, ...]
    placements: Dict[str, PlacedPackage]

    def copy(self) -> "State":
        return State(self.truck, self.packages, dict(self.placements))


# Geometry Helpers


def package_map(state: State) -> Dict[str, Package]:
    return {package.id: package for package in state.packages}


def get_package(state: State, package_id: str) -> Package:
    return package_map(state)[package_id]


def get_oriented_dimensions(package: Package, orientation: Orientation) -> Dimensions:
    if orientation == HORIZONTAL:
        return (package.width, package.length, package.height)
    if orientation == VERTICAL:
        return (package.width, package.height, package.length)
    raise ValueError(f"Unknown orientation: {orientation}")


def _bounds(package: Package, placement: PlacedPackage) -> Tuple[int, int, int, int, int, int]:
    width, length, height = get_oriented_dimensions(package, placement.orientation)
    return (
        placement.x,
        placement.x + width,
        placement.y,
        placement.y + length,
        placement.z,
        placement.z + height,
    )


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def _xy_overlap(package_a: Package, place_a: PlacedPackage, package_b: Package, place_b: PlacedPackage) -> bool:
    ax0, ax1, ay0, ay1, _, _ = _bounds(package_a, place_a)
    bx0, bx1, by0, by1, _, _ = _bounds(package_b, place_b)
    return _intervals_overlap(ax0, ax1, bx0, bx1) and _intervals_overlap(ay0, ay1, by0, by1)


def is_overlapping(package_a: Package, place_a: PlacedPackage, package_b: Package, place_b: PlacedPackage) -> bool:
    ax0, ax1, ay0, ay1, az0, az1 = _bounds(package_a, place_a)
    bx0, bx1, by0, by1, bz0, bz1 = _bounds(package_b, place_b)
    return (
        _intervals_overlap(ax0, ax1, bx0, bx1)
        and _intervals_overlap(ay0, ay1, by0, by1)
        and _intervals_overlap(az0, az1, bz0, bz1)
    )


# Constraints


def has_support(state: State, package_id: str) -> bool:
    packages = package_map(state)
    if package_id not in packages:
        return False

    placement = state.placements[package_id]
    if placement.z == 0:
        return True

    package = packages[package_id]
    for other_id, other_place in state.placements.items():
        if other_id == package_id:
            continue
        if other_id not in packages:
            continue
        other = packages[other_id]
        _, _, _, _, _, other_top = _bounds(other, other_place)
        if other_top == placement.z and _xy_overlap(package, placement, other, other_place):
            return True
    return False


def validate_state(state: State) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    packages = package_map(state)

    unknown_ids = set(state.placements) - set(packages)
    if unknown_ids:
        reasons.append("UNKNOWN_PACKAGE")

    total_weight = sum(packages[pid].weight for pid in state.placements if pid in packages)
    if total_weight > state.truck.max_capacity:
        reasons.append("WEIGHT_EXCEEDED")

    placed_items = [(pid, packages[pid], place) for pid, place in state.placements.items() if pid in packages]
    for package_id, package, placement in placed_items:
        x0, x1, y0, y1, z0, z1 = _bounds(package, placement)
        if x0 < 0 or y0 < 0 or z0 < 0 or x1 > state.truck.width or y1 > state.truck.length or z1 > state.truck.height:
            reasons.append("OUT_OF_BOUNDS")
        if placement.orientation not in ORIENTATIONS:
            reasons.append("INVALID_ORIENTATION")
        if not has_support(state, package_id):
            reasons.append("FLOATING_PACKAGE")

    for index, (id_a, package_a, place_a) in enumerate(placed_items):
        for id_b, package_b, place_b in placed_items[index + 1 :]:
            if is_overlapping(package_a, place_a, package_b, place_b):
                reasons.append("PACKAGE_OVERLAP")

            _, _, _, _, _, top_a = _bounds(package_a, place_a)
            _, _, _, _, _, top_b = _bounds(package_b, place_b)
            if package_a.is_fragile and top_a == place_b.z and _xy_overlap(package_a, place_a, package_b, place_b):
                reasons.append("FRAGILE_SUPPORT")
            if package_b.is_fragile and top_b == place_a.z and _xy_overlap(package_b, place_b, package_a, place_a):
                reasons.append("FRAGILE_SUPPORT")

    return (not reasons, sorted(set(reasons)))


def is_valid_state(state: State) -> bool:
    valid, _ = validate_state(state)
    return valid


# =========================
# Objective
# =========================


def calculate_objective(state: State) -> int:
    packages = package_map(state)
    return sum(packages[package_id].value for package_id in state.placements if package_id in packages)


# =========================
# State Initialization
# =========================


def generate_initial_state(
    truck: Truck,
    packages: Iterable[Package],
    seed: Optional[int] = None,
    attempts_per_package: int = 100,
) -> State:
    rng = random.Random(seed)
    package_list = tuple(packages)
    state = State(truck, package_list, {})
    shuffled = list(package_list)
    rng.shuffle(shuffled)

    for package in shuffled:
        orientations = list(ORIENTATIONS)
        rng.shuffle(orientations)
        placed = False
        for _ in range(attempts_per_package):
            orientation = rng.choice(orientations)
            width, length, height = get_oriented_dimensions(package, orientation)
            if width > truck.width or length > truck.length or height > truck.height:
                continue

            candidate = state.copy()
            candidate.placements[package.id] = PlacedPackage(
                rng.randint(0, truck.width - width),
                rng.randint(0, truck.length - length),
                rng.randint(0, truck.height - height),
                orientation,
            )
            if is_valid_state(candidate):
                state = candidate
                placed = True
                break

        if not placed:
            floor_candidate = _first_valid_floor_position(state, package, orientations)
            if floor_candidate is not None:
                state = floor_candidate

    return state


def _first_valid_floor_position(state: State, package: Package, orientations: Iterable[Orientation]) -> Optional[State]:
    for orientation in orientations:
        width, length, height = get_oriented_dimensions(package, orientation)
        if width > state.truck.width or length > state.truck.length or height > state.truck.height:
            continue
        for z in (0,):
            for y in range(state.truck.length - length + 1):
                for x in range(state.truck.width - width + 1):
                    candidate = state.copy()
                    candidate.placements[package.id] = PlacedPackage(x, y, z, orientation)
                    if is_valid_state(candidate):
                        return candidate
    return None


# =========================
# Neighbor Operations
# =========================


def swap_packages(state: State, package_id_a: str, package_id_b: str, require_valid: bool = True) -> Optional[State]:
    candidate = state.copy()
    place_a = candidate.placements.get(package_id_a)
    place_b = candidate.placements.get(package_id_b)

    if place_a is None and place_b is None:
        return None
    if place_a is None:
        candidate.placements[package_id_a] = place_b
        del candidate.placements[package_id_b]
    elif place_b is None:
        candidate.placements[package_id_b] = place_a
        del candidate.placements[package_id_a]
    else:
        candidate.placements[package_id_a], candidate.placements[package_id_b] = place_b, place_a

    if require_valid and not is_valid_state(candidate):
        return None
    return candidate


def move_package(
    state: State,
    package_id: str,
    position: Optional[Position],
    require_valid: bool = True,
) -> Optional[State]:
    candidate = state.copy()
    if position is None:
        candidate.placements.pop(package_id, None)
    else:
        old = candidate.placements.get(package_id, PlacedPackage(0, 0, 0, HORIZONTAL))
        candidate.placements[package_id] = replace(old, x=position[0], y=position[1], z=position[2])

    if require_valid and not is_valid_state(candidate):
        return None
    return candidate


def change_orientation(
    state: State,
    package_id: str,
    orientation: Orientation,
    require_valid: bool = True,
) -> Optional[State]:
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unknown orientation: {orientation}")
    if package_id not in state.placements:
        return None

    candidate = state.copy()
    candidate.placements[package_id] = replace(candidate.placements[package_id], orientation=orientation)
    if require_valid and not is_valid_state(candidate):
        return None
    return candidate


def generate_neighbor(state: State, seed: Optional[int] = None, max_attempts: int = 200) -> Optional[State]:
    rng = random.Random(seed)
    return _random_neighbor(state, rng, max_attempts)


def generate_neighbors(state: State, limit: Optional[int] = 200, seed: Optional[int] = None) -> List[State]:
    rng = random.Random(seed)
    neighbors: List[State] = []
    seen = set()
    max_attempts = (limit or 200) * 20

    for _ in range(max_attempts):
        neighbor = _random_neighbor(state, rng, 1)
        if neighbor is None:
            continue
        key = _state_key(neighbor)
        if key in seen:
            continue
        seen.add(key)
        neighbors.append(neighbor)
        if limit is not None and len(neighbors) >= limit:
            break

    return neighbors


def _random_neighbor(state: State, rng: random.Random, max_attempts: int) -> Optional[State]:
    package_ids = [package.id for package in state.packages]
    if not package_ids:
        return None

    for _ in range(max_attempts):
        move_type = rng.choice(("swap", "move", "orient"))

        if move_type == "swap" and len(package_ids) >= 2:
            package_a, package_b = rng.sample(package_ids, 2)
            neighbor = swap_packages(state, package_a, package_b)
        elif move_type == "move":
            package_id = rng.choice(package_ids)
            neighbor = _random_move(state, package_id, rng)
        else:
            placed_ids = list(state.placements)
            if not placed_ids:
                continue
            package_id = rng.choice(placed_ids)
            current = state.placements[package_id].orientation
            orientation = VERTICAL if current == HORIZONTAL else HORIZONTAL
            neighbor = change_orientation(state, package_id, orientation)

        if neighbor is not None:
            return neighbor
    return None


def _random_move(state: State, package_id: str, rng: random.Random) -> Optional[State]:
    if rng.random() < 0.2:
        return move_package(state, package_id, None)

    package = get_package(state, package_id)
    orientation = state.placements.get(package_id, PlacedPackage(0, 0, 0)).orientation
    width, length, height = get_oriented_dimensions(package, orientation)
    if width > state.truck.width or length > state.truck.length or height > state.truck.height:
        return None
    return move_package(
        state,
        package_id,
        (
            rng.randint(0, state.truck.width - width),
            rng.randint(0, state.truck.length - length),
            rng.randint(0, state.truck.height - height),
        ),
    )


def _state_key(state: State) -> Tuple[Tuple[str, Tuple[int, int, int, Orientation]], ...]:
    return tuple(
        sorted((package_id, (place.x, place.y, place.z, place.orientation)) for package_id, place in state.placements.items())
    )
