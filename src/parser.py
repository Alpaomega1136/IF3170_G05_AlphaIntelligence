from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from src.core import Package, Truck


@dataclass(frozen=True)
class ProgramInput:
    truck: Truck
    packages: Tuple[Package, ...]
    algorithm_name: str
    parameters: Dict[str, Any]
    additional_trucks: Tuple[Truck, ...] = ()

    @property
    def trucks(self) -> Tuple[Truck, ...]:
        return (self.truck,) + self.additional_trucks


def _require_int(data: dict, key: str, context: str, *, minimum: int | None = None) -> int:
    if key not in data or isinstance(data[key], bool) or not isinstance(data[key], int):
        raise ValueError(f"{context}.{key} must be an integer")

    value = data[key]
    if minimum is not None and value < minimum:
        raise ValueError(f"{context}.{key} must be >= {minimum}")
    return value


def _require_bool(data: dict, key: str, context: str) -> bool:
    if key not in data or not isinstance(data[key], bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return data[key]


def load_input(path: str | Path) -> ProgramInput:
    input_path = Path(path)

    try:
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Input file not found: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Input JSON root must be an object")

    problem = data.get("problem")
    if not isinstance(problem, dict):
        raise ValueError("problem must be an object")

    if ("truck" in problem) == ("trucks" in problem):
        raise ValueError("problem must contain exactly one of truck or trucks")
    trucks_data = problem.get("trucks") if "trucks" in problem else [problem["truck"]]
    if not isinstance(trucks_data, list) or not trucks_data:
        raise ValueError("problem.trucks must be a non-empty array")
    trucks = []
    for index, truck_data in enumerate(trucks_data):
        context = f"problem.trucks[{index}]" if "trucks" in problem else "problem.truck"
        if not isinstance(truck_data, dict):
            raise ValueError(f"{context} must be an object")
        trucks.append(Truck(width=_require_int(truck_data, "width", context, minimum=1),
            length=_require_int(truck_data, "length", context, minimum=1),
            height=_require_int(truck_data, "height", context, minimum=1),
            max_capacity=_require_int(truck_data, "max_capacity", context, minimum=0)))

    packages_data = problem.get("packages")
    if not isinstance(packages_data, list):
        raise ValueError("problem.packages must be an array")

    packages = []
    seen_ids = set()

    for index, package_data in enumerate(packages_data):
        context = f"problem.packages[{index}]"
        if not isinstance(package_data, dict):
            raise ValueError(f"{context} must be an object")

        package_id = package_data.get("id")
        if not isinstance(package_id, str) or not package_id.strip():
            raise ValueError(f"{context}.id must be a non-empty string")
        if package_id in seen_ids:
            raise ValueError(f"Duplicate package id: {package_id}")
        seen_ids.add(package_id)

        packages.append(Package(id=package_id,
                width=_require_int(package_data, "width", context, minimum=1),
                length=_require_int(package_data, "length", context, minimum=1),
                height=_require_int(package_data, "height", context, minimum=1),
                value=_require_int(package_data, "value", context, minimum=0),
                weight=_require_int(package_data, "weight", context, minimum=0),
                is_fragile=_require_bool(package_data, "is_fragile", context),
                eta=_require_int(package_data, "eta", context, minimum=0)))

    algorithm = data.get("algorithm")
    if not isinstance(algorithm, dict):
        raise ValueError("algorithm must be an object")

    algorithm_name = algorithm.get("name")
    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        raise ValueError("algorithm.name must be a non-empty string")

    parameters = algorithm.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("algorithm.parameters must be an object")

    return ProgramInput(truck=trucks[0], packages=tuple(packages),
        algorithm_name=algorithm_name.strip().lower(), parameters=dict(parameters),
        additional_trucks=tuple(trucks[1:]))
