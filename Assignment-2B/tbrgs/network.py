from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from .config import load_subgraph


@dataclass(frozen=True)
class NetworkNode:
    node_id: str
    label: str
    latitude: float
    longitude: float
    prediction_scats_ids: tuple[str, ...]
    proxy_nodes: tuple[str, ...]


@dataclass
class RouteEdge:
    start: str
    end: str
    distance_km: float
    quarter_hour_flow: float
    hourly_flow: float
    speed_kmh: float
    travel_minutes: float


@dataclass
class RouteRecommendation:
    rank: int
    path: list[str]
    total_minutes: float
    edges: list[RouteEdge]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2.0) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    )
    c = 2.0 * asin(sqrt(a))
    return r * c


class SubgraphNetwork:
    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_subgraph()
        self.name = config["name"]
        self.description = config.get("description", "")
        self.nodes = {
            node_id: NetworkNode(
                node_id=node_id,
                label=node_data["label"],
                latitude=float(node_data["coordinates"][1]),
                longitude=float(node_data["coordinates"][0]),
                prediction_scats_ids=tuple(node_data.get("prediction_scats_ids", [])),
                proxy_nodes=tuple(node_data.get("proxy_nodes", [])),
            )
            for node_id, node_data in config["nodes"].items()
        }
        self.adjacency = {node_id: set() for node_id in self.nodes}
        for start, end in config["edges"]:
            self.adjacency[start].add(end)
            self.adjacency[end].add(start)

    def neighbors(self, node_id: str) -> list[str]:
        return sorted(self.adjacency.get(node_id, []))

    def distance_km(self, start: str, end: str) -> float:
        a = self.nodes[start]
        b = self.nodes[end]
        return haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)

    def all_simple_paths(
        self,
        origin: str,
        destination: str,
        max_depth: int | None = None,
    ) -> list[list[str]]:
        if origin not in self.nodes or destination not in self.nodes:
            raise KeyError("Origin or destination node is not present in the configured network.")

        if origin == destination:
            return [[origin]]

        if max_depth is None:
            max_depth = len(self.nodes)

        paths: list[list[str]] = []

        def dfs(current: str, target: str, visited: set[str], path: list[str]) -> None:
            if len(path) > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return
            for neighbor in self.neighbors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, target, visited, path)
                path.pop()
                visited.remove(neighbor)

        dfs(origin, destination, {origin}, [origin])
        return paths

    def scaled_positions(
        self,
        width: int = 700,
        height: int = 500,
        padding: int = 40,
    ) -> dict[str, tuple[float, float]]:
        lons = [node.longitude for node in self.nodes.values()]
        lats = [node.latitude for node in self.nodes.values()]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        lon_span = max(max_lon - min_lon, 1e-9)
        lat_span = max(max_lat - min_lat, 1e-9)
        x_scale = (width - 2 * padding) / lon_span
        y_scale = (height - 2 * padding) / lat_span

        positions = {}
        for node_id, node in self.nodes.items():
            x = padding + (node.longitude - min_lon) * x_scale
            y = height - padding - (node.latitude - min_lat) * y_scale
            positions[node_id] = (x, y)
        return positions

    def plot_routes(
        self,
        routes: Iterable[RouteRecommendation],
        output_path: str | Path,
    ) -> Path:
        positions = self.scaled_positions()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        for start, neighbors in self.adjacency.items():
            for end in neighbors:
                if start < end:
                    x1, y1 = positions[start]
                    x2, y2 = positions[end]
                    plt.plot([x1, x2], [y1, y2], color="#b3b3b3", linewidth=1.5, zorder=1)

        palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        for idx, route in enumerate(routes):
            color = palette[idx % len(palette)]
            for edge_start, edge_end in zip(route.path[:-1], route.path[1:]):
                x1, y1 = positions[edge_start]
                x2, y2 = positions[edge_end]
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=3, zorder=3)

        for node_id, node in self.nodes.items():
            x, y = positions[node_id]
            plt.scatter(x, y, s=200, color="#111111", zorder=5)
            plt.text(x, y + 10, node_id, ha="center", va="bottom", fontsize=10, fontweight="bold")
            plt.text(x, y - 10, node.label, ha="center", va="top", fontsize=8)

        plt.title(self.name)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path
