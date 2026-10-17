from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Dict, Iterable, List, Optional, Tuple


Orientation = str
Position = Tuple[int, int, int]  # (x, y, z)
Dimensions = Tuple[int, int, int]  # (width, length, height)

ORIENTATION_WLH = "w:x,l:y,h:z"
ORIENTATION_WHL = "w:x,l:z,h:y"
ORIENTATION_LWH = "w:y,l:x,h:z"
ORIENTATION_LHW = "w:z,l:x,h:y"
ORIENTATION_HWL = "w:y,l:z,h:x"
ORIENTATION_HLW = "w:z,l:y,h:x"

DEFAULT_ORIENTATION = ORIENTATION_WLH
ORIENTATIONS: Tuple[Orientation, ...] = (
    ORIENTATION_WLH,
    ORIENTATION_WHL,
    ORIENTATION_LWH,
    ORIENTATION_LHW,
    ORIENTATION_HWL,
    ORIENTATION_HLW,
)
ORIENTATION_AXES = {
    ORIENTATION_WLH: ("w", "l", "h"),
    ORIENTATION_WHL: ("w", "h", "l"),
    ORIENTATION_LWH: ("l", "w", "h"),
    ORIENTATION_LHW: ("l", "h", "w"),
    ORIENTATION_HWL: ("h", "w", "l"),
    ORIENTATION_HLW: ("h", "l", "w"),
}
AXES_ORIENTATION = {axes: orientation for orientation, axes in ORIENTATION_AXES.items()}

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
    orientation: Orientation = DEFAULT_ORIENTATION
    truck_index: int = 0

    @property
    def position(self) -> Position:
        return (self.x, self.y, self.z)


@dataclass
class State:
    truck: Truck
    packages: Tuple[Package, ...]
    placements: Dict[str, PlacedPackage]
    additional_trucks: Tuple[Truck, ...] = ()

    @property
    def trucks(self) -> Tuple[Truck, ...]:
        return (self.truck,) + self.additional_trucks

    def copy(self) -> "State":
        return State(self.truck, self.packages, dict(self.placements), self.additional_trucks)


def get_truck_state(state: State, truck_index: int) -> State:
    placements = {package_id: replace(place, truck_index=0)
                  for package_id, place in state.placements.items() if place.truck_index == truck_index}
    packages = tuple(package for package in state.packages if package.id in placements)
    return State(state.trucks[truck_index], packages, placements)


# Geometry Helpers

def package_map(state: State) -> Dict[str, Package]:
    return {package.id: package for package in state.packages}


def get_package(state: State, package_id: str) -> Package:
    return package_map(state)[package_id]


def get_oriented_dimensions(package: Package, orientation: Orientation) -> Dimensions:
    if orientation not in ORIENTATION_AXES:
        raise ValueError(f"Unknown orientation: {orientation}")

    dimensions = []
    for side in ORIENTATION_AXES[orientation]:
        if side == "w":
            dimensions.append(package.width)
        elif side == "l":
            dimensions.append(package.length)
        else:
            dimensions.append(package.height)

    return (dimensions[0], dimensions[1], dimensions[2])


def rotate_orientation(orientation: Orientation, axis: str) -> Orientation:
    """Rotate a package 90 degrees around x, y, or z.

    For rectangular boxes, +90 and -90 have the same resulting dimensions,
    so each axis simply swaps the two other axes.
    """
    if orientation not in ORIENTATION_AXES:
        raise ValueError(f"Unknown orientation: {orientation}")

    x_side, y_side, z_side = ORIENTATION_AXES[orientation]
    if axis == "x":
        rotated_axes = (x_side, z_side, y_side)
    elif axis == "y":
        rotated_axes = (z_side, y_side, x_side)
    elif axis == "z":
        rotated_axes = (y_side, x_side, z_side)
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")

    return AXES_ORIENTATION[rotated_axes]


def _bounds(package: Package, placement: PlacedPackage) -> Tuple[int, int, int, int, int, int]:
    width, length, height = get_oriented_dimensions(package, placement.orientation)
    x0 = placement.x
    x1 = placement.x + width
    y0 = placement.y
    y1 = placement.y + length
    z0 = placement.z
    z1 = placement.z + height
    return (x0, x1, y0, y1, z0, z1)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def _xy_overlap(package_a: Package, place_a: PlacedPackage,
                package_b: Package, place_b: PlacedPackage) -> bool:
    if place_a.truck_index != place_b.truck_index:
        return False
    ax0, ax1, ay0, ay1, _, _ = _bounds(package_a, place_a)
    bx0, bx1, by0, by1, _, _ = _bounds(package_b, place_b)
    x_overlaps = _intervals_overlap(ax0, ax1, bx0, bx1)
    y_overlaps = _intervals_overlap(ay0, ay1, by0, by1)
    return x_overlaps and y_overlaps


def is_overlapping(package_a: Package, place_a: PlacedPackage,
                   package_b: Package, place_b: PlacedPackage) -> bool:
    if place_a.truck_index != place_b.truck_index:
        return False
    ax0, ax1, ay0, ay1, az0, az1 = _bounds(package_a, place_a)
    bx0, bx1, by0, by1, bz0, bz1 = _bounds(package_b, place_b)
    x_overlaps = _intervals_overlap(ax0, ax1, bx0, bx1)
    y_overlaps = _intervals_overlap(ay0, ay1, by0, by1)
    z_overlaps = _intervals_overlap(az0, az1, bz0, bz1)
    return x_overlaps and y_overlaps and z_overlaps


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

    for placement in state.placements.values():
        if (isinstance(placement.truck_index, bool) or not isinstance(placement.truck_index, int)
                or not 0 <= placement.truck_index < len(state.trucks)):
            return False, sorted(set(reasons + ["INVALID_TRUCK_INDEX"]))
        if placement.orientation not in ORIENTATIONS:
            return False, sorted(set(reasons + ["INVALID_ORIENTATION"]))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in placement.position):
            return False, sorted(set(reasons + ["INVALID_COORDINATE"]))

    if state.additional_trucks:
        for truck_index in range(len(state.trucks)):
            _, truck_reasons = validate_state(get_truck_state(state, truck_index))
            reasons.extend(f"TRUCK_{truck_index + 1}_{reason}" for reason in truck_reasons)
        return not reasons, sorted(set(reasons))

    total_weight = 0
    for package_id in state.placements:
        if package_id in packages:
            total_weight += packages[package_id].weight

    if total_weight > state.truck.max_capacity:
        reasons.append("WEIGHT_EXCEEDED")

    placed_items = []
    for package_id, placement in state.placements.items():
        if package_id in packages:
            placed_items.append((package_id, packages[package_id], placement))

    for package_id, package, placement in placed_items:
        x0, x1, y0, y1, z0, z1 = _bounds(package, placement)
        out_of_bounds = (
            x0 < 0 or y0 < 0 or z0 < 0
            or x1 > state.truck.width
            or y1 > state.truck.length
            or z1 > state.truck.height
        )
        if out_of_bounds:
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
            package_a_supports_b = top_a == place_b.z and _xy_overlap(package_a, place_a, package_b, place_b)
            package_b_supports_a = top_b == place_a.z and _xy_overlap(package_b, place_b, package_a, place_a)

            if package_a.is_fragile and package_a_supports_b:
                reasons.append("FRAGILE_SUPPORT")
            if package_b.is_fragile and package_b_supports_a:
                reasons.append("FRAGILE_SUPPORT")

    return (not reasons, sorted(set(reasons)))


def is_valid_state(state: State) -> bool:
    valid, _ = validate_state(state)
    return valid


# Objective


def calculate_objective(state: State) -> int:
    packages = package_map(state)
    total_value = 0
    for package_id in state.placements:
        if package_id in packages:
            total_value += packages[package_id].value
    return total_value


# State Initialization

def generate_initial_state(
    truck: Truck | Tuple[Truck, ...],
    packages: Iterable[Package],
    seed: Optional[int] = None,
    attempts_per_package: int = 100,
) -> State:
    rng = random.Random(seed)
    package_list = tuple(packages)
    if isinstance(truck, tuple):
        if not truck:
            raise ValueError("At least one truck is required")
        if len(truck) == 1:
            return generate_initial_state(truck[0], package_list, seed, attempts_per_package)
        state = State(truck[0], package_list, {}, truck[1:])
        truck_indices = list(range(len(truck)))
        rng.shuffle(truck_indices)
        # shortcut: random sequential packing, consider joint packing if initialization quality stalls.
        for truck_index in truck_indices:
            remaining = tuple(package for package in package_list if package.id not in state.placements)
            local = generate_initial_state(truck[truck_index], remaining,
                seed=rng.randrange(1_000_000_000), attempts_per_package=attempts_per_package)
            state.placements.update({package_id: replace(place, truck_index=truck_index)
                                     for package_id, place in local.placements.items()})
        return state
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


# Neighbor Operations

def swap_packages(state: State, package_id_a: str, package_id_b: str,
                  require_valid: bool = True) -> Optional[State]:
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
    *,
    truck_index: Optional[int] = None,
) -> Optional[State]:
    candidate = state.copy()
    if position is None:
        candidate.placements.pop(package_id, None)
    else:
        old = candidate.placements.get(package_id, PlacedPackage(0, 0, 0, DEFAULT_ORIENTATION))
        candidate.placements[package_id] = replace(old, x=position[0], y=position[1], z=position[2],
            truck_index=old.truck_index if truck_index is None else truck_index)

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


def rotate_package(
    state: State,
    package_id: str,
    axis: str,
    require_valid: bool = True,
) -> Optional[State]:
    if package_id not in state.placements:
        return None

    current = state.placements[package_id].orientation
    orientation = rotate_orientation(current, axis)
    return change_orientation(state, package_id, orientation, require_valid)


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
    package_ids = []
    for package in state.packages:
        package_ids.append(package.id)

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
            axis = rng.choice(("x", "y", "z"))
            neighbor = rotate_package(state, package_id, axis)

        if neighbor is not None:
            return neighbor
    return None


def _random_move(state: State, package_id: str, rng: random.Random) -> Optional[State]:
    if rng.random() < 0.2:
        return move_package(state, package_id, None)

    package = get_package(state, package_id)
    orientation = state.placements.get(package_id, PlacedPackage(0, 0, 0)).orientation
    width, length, height = get_oriented_dimensions(package, orientation)
    truck_index = rng.randrange(len(state.trucks)) if state.additional_trucks else 0
    truck = state.trucks[truck_index]
    if width > truck.width or length > truck.length or height > truck.height:
        return None
    return move_package(
        state,
        package_id,
        (
            rng.randint(0, truck.width - width),
            rng.randint(0, truck.length - length),
            rng.randint(0, truck.height - height),
        ),
        truck_index=truck_index,
    )


def _state_key(state: State) -> Tuple[Tuple[str, Tuple[int, int, int, Orientation, int]], ...]:
    key_items = []
    for package_id, place in state.placements.items():
        placement_data = (place.x, place.y, place.z, place.orientation, place.truck_index)
        key_items.append((package_id, placement_data))
    return tuple(sorted(key_items))
