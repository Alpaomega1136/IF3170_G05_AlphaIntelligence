from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


Orientation = str
Position = Tuple[int, int, int]
Dimensions = Tuple[int, int, int]

HORIZONTAL = "Horizontal"
VERTICAL = "Vertical"
ORIENTATIONS: Tuple[Orientation, Orientation] = (HORIZONTAL, VERTICAL)


# =========================
# Data Models
# =========================


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
