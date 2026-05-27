from __future__ import annotations

from pathlib import Path

from .model_service import TrafficPredictionService
from .network import RouteEdge, RouteRecommendation, SubgraphNetwork
from .part_a_bridge import solve_with_part_a_search
from .travel_time import quarter_hour_to_hourly_flow, speed_from_hourly_flow, travel_time_minutes


class RoutingEngine:
    def __init__(
        self,
        network: SubgraphNetwork,
        prediction_service: TrafficPredictionService,
        speed_limit_kmh: float = 60.0,
        intersection_delay_seconds: float = 30.0,
    ):
        self.network = network
        self.prediction_service = prediction_service
        self.speed_limit_kmh = speed_limit_kmh
        self.intersection_delay_seconds = intersection_delay_seconds

    @staticmethod
    def time_to_index(time_text: str) -> int:
        parts = time_text.strip().split(":")
        if len(parts) != 2:
            raise ValueError("Time must be in HH:MM format.")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and minute in {0, 15, 30, 45}):
            raise ValueError("Time must use a valid 24-hour clock and 15-minute interval.")
        return hour * 4 + (minute // 15)

    def _edge_details(
        self,
        start: str,
        end: str,
        node_flows: dict[str, float],
        use_congested_branch: bool,
    ) -> RouteEdge:
        distance_km = self.network.distance_km(start, end)
        quarter_hour_flow = (node_flows[start] + node_flows[end]) / 2.0
        hourly_flow = quarter_hour_to_hourly_flow(quarter_hour_flow)
        speed_kmh = speed_from_hourly_flow(
            hourly_flow=hourly_flow,
            speed_limit_kmh=self.speed_limit_kmh,
            use_congested_branch=use_congested_branch,
        )
        travel_minutes = travel_time_minutes(
            distance_km=distance_km,
            hourly_flow=hourly_flow,
            speed_limit_kmh=self.speed_limit_kmh,
            use_congested_branch=use_congested_branch,
            intersection_delay_seconds=self.intersection_delay_seconds,
        )
        return RouteEdge(
            start=start,
            end=end,
            distance_km=distance_km,
            quarter_hour_flow=quarter_hour_flow,
            hourly_flow=hourly_flow,
            speed_kmh=speed_kmh,
            travel_minutes=travel_minutes,
        )

    def _score_path(
        self,
        path: list[str],
        node_flows: dict[str, float],
        use_congested_branch: bool,
    ) -> RouteRecommendation:
        edges: list[RouteEdge] = []
        total_minutes = 0.0
        for start, end in zip(path[:-1], path[1:]):
            edge = self._edge_details(start, end, node_flows, use_congested_branch)
            edges.append(edge)
            total_minutes += edge.travel_minutes
        return RouteRecommendation(rank=0, path=path, total_minutes=total_minutes, edges=edges)

    def recommend_routes(
        self,
        origin: str,
        destination: str,
        predict_day: str,
        time_text: str,
        model_name: str = "random_forest",
        search_method: str = "ranked_paths",
        top_k: int = 5,
        use_congested_branch: bool = False,
        output_plot_path: str | Path | None = None,
    ) -> dict:
        time_index = self.time_to_index(time_text)
        node_flows = self.prediction_service.predict_all_node_flows(
            model_name=model_name,
            predict_day=predict_day,
            time_index=time_index,
        )
        all_paths = self.network.all_simple_paths(origin, destination)
        if not all_paths:
            raise ValueError(f"No path exists from {origin} to {destination} in the configured subgraph.")

        ranked = [
            self._score_path(path, node_flows=node_flows, use_congested_branch=use_congested_branch)
            for path in all_paths
        ]
        ranked.sort(key=lambda route: (route.total_minutes, len(route.path), route.path))
        ranked = ranked[:top_k]
        for rank, route in enumerate(ranked, start=1):
            route.rank = rank

        selected_route = None
        primary_path = solve_with_part_a_search(
            network=self.network,
            node_flows=node_flows,
            origin=origin,
            destination=destination,
            search_method=search_method,
            speed_limit_kmh=self.speed_limit_kmh,
            intersection_delay_seconds=self.intersection_delay_seconds,
            use_congested_branch=use_congested_branch,
        )
        if primary_path:
            selected_route = self._score_path(
                primary_path,
                node_flows=node_flows,
                use_congested_branch=use_congested_branch,
            )
            selected_route.rank = 1

        plot_path = None
        if output_plot_path is not None:
            plot_path = self.network.plot_routes(ranked, output_plot_path)

        return {
            "origin": origin,
            "destination": destination,
            "predict_day": predict_day,
            "time_text": time_text,
            "time_index": time_index,
            "model_name": model_name,
            "search_method": search_method,
            "node_flows": node_flows,
            "routes": ranked,
            "selected_route": selected_route,
            "plot_path": str(plot_path) if plot_path else None,
        }
