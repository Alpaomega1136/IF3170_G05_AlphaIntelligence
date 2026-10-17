from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.core import (State, calculate_objective, get_oriented_dimensions, get_truck_state, package_map)


TRUCK_EDGE_COLOR = "#334155"
CURRENT_LINE_COLOR = "#2563EB"
BEST_LINE_COLOR = "#F59E0B"
ACCEPTANCE_LINE_COLOR = "#7C3AED"


def _ensure_parent(output_path: Optional[str | Path]) -> None:
    if output_path is None:
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def _cuboid_faces(x: float, y: float, z: float, width: float, length: float, height: float):
    p000 = (x, y, z)
    p100 = (x + width, y, z)
    p110 = (x + width, y + length, z)
    p010 = (x, y + length, z)

    p001 = (x, y, z + height)
    p101 = (x + width, y, z + height)
    p111 = (x + width, y + length, z + height)
    p011 = (x, y + length, z + height)

    return [[p000, p100, p110, p010], [p001, p101, p111, p011], [p000, p100, p101, p001],
        [p010, p110, p111, p011], [p000, p010, p011, p001], [p100, p110, p111, p101]]


def _truck_corners(state: State):
    truck = state.truck

    x0, x1 = 0, truck.width
    y0, y1 = 0, truck.length
    z0, z1 = 0, truck.height

    return {"000": (x0, y0, z0), "100": (x1, y0, z0), "110": (x1, y1, z0), "010": (x0, y1, z0),
        "001": (x0, y0, z1), "101": (x1, y0, z1), "111": (x1, y1, z1), "011": (x0, y1, z1)}


def _draw_edge(ax, p1, p2, *, linewidth: float, alpha: float, zorder: int) -> None:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=TRUCK_EDGE_COLOR,
        linewidth=linewidth, alpha=alpha, zorder=zorder)


def _draw_truck_wireframe_back(ax, state: State) -> None:
    corners = _truck_corners(state)

    edges = (("000", "100"), ("100", "110"), ("110", "010"), ("010", "000"), ("001", "101"),
        ("101", "111"), ("111", "011"), ("011", "001"), ("000", "001"), ("100", "101"),
        ("110", "111"), ("010", "011"))

    for start, end in edges:
        _draw_edge(ax, corners[start], corners[end], linewidth=1.2, alpha=0.60, zorder=1)


def _draw_front_truck_edges(ax, state: State) -> None:
    corners = _truck_corners(state)

    front_edges = (("100", "101"), ("001", "101"), ("101", "111"))

    for start, end in front_edges:
        _draw_edge(ax, corners[start], corners[end], linewidth=2.4, alpha=1.0, zorder=100)


def _package_color_map(state: State):
    sorted_ids = sorted(package.id for package in state.packages)

    cmap = plt.get_cmap("tab20")

    return {package_id: cmap(index % cmap.N) for index, package_id in enumerate(sorted_ids)}


def _draw_state(ax, state: State, title: str, color_map, show_outside: bool) -> None:
    packages = package_map(state)

    ax.computed_zorder = False

    _draw_truck_wireframe_back(ax, state)

    sorted_placements = sorted(state.placements.items(), key=lambda item: item[0])

    legend_handles = []

    for package_id, placement in sorted_placements:
        package = packages[package_id]

        width, length, height = (get_oriented_dimensions(package, placement.orientation))

        color = color_map[package_id]

        faces = _cuboid_faces(placement.x, placement.y, placement.z, width, length, height)

        poly = Poly3DCollection(faces, facecolors=color, edgecolors="#111827", linewidths=0.8,
            alpha=0.72, zorder=10)

        ax.add_collection3d(poly)

        legend_handles.append(Patch(facecolor=color, edgecolor="#111827", label=(f"{package_id} "
                    f"({package.width} x " f"{package.length} x " f"{package.height})"), alpha=0.85)
        )

    _draw_front_truck_edges(ax, state)

    outside_packages = [package.id for package in state.packages
        if package.id not in state.placements]

    total_weight = sum(packages[package_id].weight for package_id in state.placements)

    total_value = calculate_objective(state)

    outside_text = (", ".join(outside_packages) if outside_packages else "None")

    subtitle = f"Total value: {total_value} | Weight: {total_weight}/{state.truck.max_capacity}"
    if show_outside:
        subtitle += f"\nOutside packages: {outside_text}"

    ax.set_title(f"{title}\n{subtitle}", pad=20)

    ax.set_xlabel("Width (x)")
    ax.set_ylabel("Length (y)")
    ax.set_zlabel("Height (z)")

    ax.set_xlim(0, state.truck.width)
    ax.set_ylim(0, state.truck.length)
    ax.set_zlim(0, state.truck.height)

    try:
        ax.set_box_aspect((state.truck.width, state.truck.length, state.truck.height))
    except AttributeError:
        pass

    if legend_handles:
        ax.legend(handles=legend_handles, title="Packages (w x l x h)", loc="upper left",
            bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

def visualize_state(state: State, title: str = "Truck State",
    output_path: Optional[str | Path] = None, show: bool = False) -> None:
    _ensure_parent(output_path)
    fleet = bool(state.additional_trucks)
    columns = min(2, len(state.trucks))
    rows = (len(state.trucks) + columns - 1) // columns
    fig = plt.figure(figsize=(11 * columns, 8 * rows))
    color_map = _package_color_map(state)
    for index in range(len(state.trucks)):
        local = get_truck_state(state, index) if fleet else state
        ax = fig.add_subplot(rows, columns, index + 1, projection="3d")
        _draw_state(ax, local, f"Truck {index + 1}" if fleet else title, color_map, not fleet)
    if fleet:
        outside = ", ".join(p.id for p in state.packages if p.id not in state.placements) or "None"
        fig.suptitle(f"{title} | Fleet value: {calculate_objective(state)}\nOutside packages: {outside}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        fig.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_objective_history(result, output_path: Optional[str | Path] = None, show: bool = False,
) -> None:
    _ensure_parent(output_path)

    iterations = range(len(result.objective_history))

    fig, ax = plt.subplots(figsize=(10, 5))

    if result.metadata.get("algorithm") == "genetic_algorithm":
        ax.plot(iterations, result.objective_history, label="Maximum objective",
            color=BEST_LINE_COLOR, linewidth=2.0, marker="o", markersize=3)
        ax.plot(iterations, result.mean_objective_history, label="Mean objective",
            color=CURRENT_LINE_COLOR, linewidth=1.6, marker="o", markersize=3)
        ax.set_title("Genetic Algorithm: Population Fitness")
        ax.set_xlabel("Generation")
    elif not hasattr(result, "best_objective_history"):
        label = "Restart" if result.metadata.get("algorithm") == "random_restart_hill_climbing" else "Iteration"
        ax.plot(iterations, result.objective_history, label="Objective",
            color=CURRENT_LINE_COLOR, linewidth=1.6, marker="o", markersize=3)
        ax.set_title(f"Objective Value vs {label}")
        ax.set_xlabel(label)
    else:
        ax.plot(iterations, result.objective_history, label="Current objective",
            color=CURRENT_LINE_COLOR, linewidth=1.6)
        ax.plot(iterations, result.best_objective_history, label="Best objective",
            color=BEST_LINE_COLOR, linewidth=2.0)
        objective_mode = result.metadata.get("objective_mode", "value")
        ax.set_title("Objective Value vs Iteration" f"\nObjective mode: {objective_mode}")
        ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_acceptance_probability(result, output_path: Optional[str | Path] = None,
    show: bool = False, worse_only: bool = True) -> None:
    _ensure_parent(output_path)

    if worse_only:
        points = [(index + 1, probability) for index, (delta_e, probability) in enumerate(zip(
                    result.delta_e_history, result.acceptance_probability_history)) if delta_e < 0]

        x_values = [point[0] for point in points]

        y_values = [point[1] for point in points]

        title_suffix = "Worse Moves Only"
    else:
        x_values = list(range(1, len(result.acceptance_probability_history) + 1))

        y_values = (result.acceptance_probability_history)

        title_suffix = "All Moves"

    fig, ax = plt.subplots(figsize=(10, 5))

    if x_values:
        ax.plot(x_values, y_values, color=ACCEPTANCE_LINE_COLOR, linewidth=1.4, marker="o",
            markersize=3)
    else:
        ax.text(0.5, 0.5, "No worse moves were generated.", transform=ax.transAxes, ha="center",
            va="center")

    ax.set_title("Acceptance Probability " r"$e^{\Delta E/T}$ " f"vs Iteration\n{title_suffix}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acceptance probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def save_sa_visualizations(result, output_dir: str | Path, show: bool = False) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_state(result.initial_state, title="Initial State", output_path=(
            output_dir / "initial_state.png"), show=show)

    visualize_state(result.final_state, title="Final State", output_path=(
            output_dir / "final_state.png"), show=show)

    plot_objective_history(result, output_path=(output_dir / "objective_history.png"), show=show)

    plot_acceptance_probability(result, output_path=(output_dir / "acceptance_probability.png"),
        show=show, worse_only=True)
