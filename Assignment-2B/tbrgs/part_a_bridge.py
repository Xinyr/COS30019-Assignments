from __future__ import annotations

import sys
from pathlib import Path

from .config import PROJECT_ROOT
from .network import SubgraphNetwork
from .travel_time import quarter_hour_to_hourly_flow, travel_time_minutes


PART_A_PROGRAM_DIR = PROJECT_ROOT.parent / "Assignment-2A" / "program"
if str(PART_A_PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PART_A_PROGRAM_DIR))

from search import (  # type: ignore  # noqa: E402
    GraphProblem,
    UndirectedGraph,
    astar_search,
    breadth_first_graph_search,
    depth_first_graph_search,
    greedy_best_first_graph_search,
    iterative_deepening_search,
    recursive_best_first_search,
)


SEARCH_METHODS = {
    "bfs": breadth_first_graph_search,
    "dfs": depth_first_graph_search,
    "gbfs": lambda problem: greedy_best_first_graph_search(problem, problem.h),
    "astar": lambda problem: astar_search(problem, problem.h),
    "cus1": iterative_deepening_search,
    "cus2": lambda problem: recursive_best_first_search(problem, problem.h),
}


def build_weighted_graph(
    network: SubgraphNetwork,
    node_flows: dict[str, float],
    speed_limit_kmh: float = 60.0,
    intersection_delay_seconds: float = 30.0,
    use_congested_branch: bool = False,
):
    graph_dict: dict[str, dict[str, float]] = {node_id: {} for node_id in network.nodes}
    for start, neighbors in network.adjacency.items():
        for end in neighbors:
            if end in graph_dict[start]:
                continue
            distance_km = network.distance_km(start, end)
            quarter_hour_flow = (node_flows[start] + node_flows[end]) / 2.0
            hourly_flow = quarter_hour_to_hourly_flow(quarter_hour_flow)
            minutes = travel_time_minutes(
                distance_km=distance_km,
                hourly_flow=hourly_flow,
                speed_limit_kmh=speed_limit_kmh,
                use_congested_branch=use_congested_branch,
                intersection_delay_seconds=intersection_delay_seconds,
            )
            graph_dict[start][end] = minutes
            graph_dict[end][start] = minutes

    graph = UndirectedGraph(graph_dict)
    graph.locations = {
        node_id: (node.longitude, node.latitude) for node_id, node in network.nodes.items()
    }
    return graph


def solve_with_part_a_search(
    network: SubgraphNetwork,
    node_flows: dict[str, float],
    origin: str,
    destination: str,
    search_method: str,
    speed_limit_kmh: float = 60.0,
    intersection_delay_seconds: float = 30.0,
    use_congested_branch: bool = False,
) -> list[str] | None:
    normalized = search_method.lower()
    if normalized == "ranked_paths":
        return None
    if normalized not in SEARCH_METHODS:
        raise ValueError(
            f"Unsupported search method '{search_method}'. "
            f"Choose from {sorted(['ranked_paths', *SEARCH_METHODS.keys()])}."
        )

    weighted_graph = build_weighted_graph(
        network=network,
        node_flows=node_flows,
        speed_limit_kmh=speed_limit_kmh,
        intersection_delay_seconds=intersection_delay_seconds,
        use_congested_branch=use_congested_branch,
    )
    problem = GraphProblem(origin, destination, weighted_graph)
    solution = SEARCH_METHODS[normalized](problem)
    if solution is None:
        return None
    return [node.state for node in solution.path()]
