import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import (  # noqa: E402
    HORIZONTAL,
    VERTICAL,
    Package,
    PlacedPackage,
    State,
    Truck,
    calculate_objective,
    change_orientation,
    generate_initial_state,
    get_oriented_dimensions,
    is_valid_state,
    move_package,
    swap_packages,
    validate_state,
)


def pkg(package_id, width=2, length=2, height=1, value=10, weight=1, fragile=False):
    return Package(package_id, width, length, height, value, weight, fragile, eta=1)


def state_with(placements, packages=None, truck=None):
    packages = tuple(packages or [pkg(package_id) for package_id in placements])
    truck = truck or Truck(5, 5, 5, 20)
    return State(truck, packages, placements)


def test_boundary_valid_and_invalid():
    package = pkg("A")
    truck = Truck(4, 4, 2, 10)

    assert is_valid_state(State(truck, (package,), {"A": PlacedPackage(0, 0, 0)}))

    valid, reasons = validate_state(State(truck, (package,), {"A": PlacedPackage(3, 0, 0)}))
    assert not valid
    assert "OUT_OF_BOUNDS" in reasons

    valid, reasons = validate_state(State(truck, (package,), {"A": PlacedPackage(0, 0, 2)}))
    assert not valid
    assert "OUT_OF_BOUNDS" in reasons


def test_collision_touching_surface_is_valid_overlap_volume_is_invalid():
    packages = (pkg("A"), pkg("B"))

    touching = state_with({"A": PlacedPackage(0, 0, 0), "B": PlacedPackage(2, 0, 0)}, packages)
    assert is_valid_state(touching)

    overlapping = state_with({"A": PlacedPackage(0, 0, 0), "B": PlacedPackage(1, 0, 0)}, packages)
    valid, reasons = validate_state(overlapping)
    assert not valid
    assert "PACKAGE_OVERLAP" in reasons


def test_support_floor_supported_and_floating():
    bottom = pkg("A", width=3, length=3, height=1)
    top = pkg("B", width=1, length=1, height=1)

    floor = state_with({"A": PlacedPackage(0, 0, 0)}, (bottom, top))
    assert is_valid_state(floor)

    supported = state_with({"A": PlacedPackage(0, 0, 0), "B": PlacedPackage(1, 1, 1)}, (bottom, top))
    assert is_valid_state(supported)

    floating = state_with({"A": PlacedPackage(0, 0, 0), "B": PlacedPackage(4, 4, 1)}, (bottom, top))
    valid, reasons = validate_state(floating)
    assert not valid
    assert "FLOATING_PACKAGE" in reasons


def test_fragile_package_cannot_support_another_package():
    fragile = pkg("A", width=3, length=3, height=1, fragile=True)
    top = pkg("B", width=1, length=1, height=1)
    state = state_with({"A": PlacedPackage(0, 0, 0), "B": PlacedPackage(1, 1, 1)}, (fragile, top))

    valid, reasons = validate_state(state)

    assert not valid
    assert "FRAGILE_SUPPORT" in reasons


def test_weight_capacity_less_equal_and_exceeded():
    truck = Truck(5, 5, 5, 3)
    a = pkg("A", weight=1)
    b = pkg("B", weight=2)
    c = pkg("C", weight=1)

    assert is_valid_state(State(truck, (a, b), {"A": PlacedPackage(0, 0, 0), "B": PlacedPackage(2, 0, 0)}))
    assert is_valid_state(State(truck, (a,), {"A": PlacedPackage(0, 0, 0)}))

    valid, reasons = validate_state(
        State(truck, (a, b, c), {"A": PlacedPackage(0, 0, 0), "B": PlacedPackage(2, 0, 0), "C": PlacedPackage(0, 2, 0)})
    )
    assert not valid
    assert "WEIGHT_EXCEEDED" in reasons


def test_orientation_and_objective():
    package = pkg("A", width=2, length=3, height=4, value=99)
    state = State(Truck(5, 5, 5, 10), (package,), {"A": PlacedPackage(0, 0, 0)})

    assert get_oriented_dimensions(package, HORIZONTAL) == (2, 3, 4)
    assert get_oriented_dimensions(package, VERTICAL) == (2, 4, 3)
    assert calculate_objective(state) == 99


def test_neighbor_operations_do_not_mutate_original_state():
    packages = (pkg("A"), pkg("B"))
    original = state_with({"A": PlacedPackage(0, 0, 0), "B": PlacedPackage(2, 0, 0)}, packages)

    swapped = swap_packages(original, "A", "B")
    moved = move_package(original, "A", (0, 2, 0))
    oriented = change_orientation(original, "A", VERTICAL)

    assert swapped is not None
    assert moved is not None
    assert oriented is not None
    assert original.placements["A"] == PlacedPackage(0, 0, 0)
    assert swapped.placements["A"] == PlacedPackage(2, 0, 0)
    assert moved.placements["A"] == PlacedPackage(0, 2, 0)
    assert oriented.placements["A"].orientation == VERTICAL


def test_generate_initial_state_is_valid_and_seeded():
    packages = [pkg("A"), pkg("B"), pkg("C")]
    truck = Truck(5, 5, 3, 10)

    first = generate_initial_state(truck, packages, seed=7)
    second = generate_initial_state(truck, packages, seed=7)

    assert is_valid_state(first)
    assert first.placements == second.placements
