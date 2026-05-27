from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT, load_defaults
from .model_service import TrafficPredictionService
from .network import SubgraphNetwork
from .router import RoutingEngine


def build_parser() -> argparse.ArgumentParser:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="Traffic-Based Route Guidance System")
    parser.add_argument("--gui", action="store_true", help="Launch the Tkinter GUI.")
    parser.add_argument("--origin", default=defaults["default_origin"])
    parser.add_argument("--destination", default=defaults["default_destination"])
    parser.add_argument("--date", default=defaults["default_date"])
    parser.add_argument("--time", default=defaults["default_time"])
    parser.add_argument(
        "--model",
        default=defaults["default_model"],
        choices=["random_forest", "lstm", "gru"],
    )
    parser.add_argument(
        "--search-method",
        default=defaults.get("default_search_method", "astar"),
        choices=["ranked_paths", "astar", "gbfs", "bfs", "dfs", "cus1", "cus2"],
    )
    parser.add_argument("--top-k", type=int, default=int(defaults["default_top_k"]))
    parser.add_argument("--congested", action="store_true", help="Use the congested branch of the flow-speed curve.")
    parser.add_argument(
        "--plot",
        default=str(PROJECT_ROOT / "outputs" / "latest_routes.png"),
        help="Where to save the route plot PNG.",
    )
    return parser


def run_cli(args: argparse.Namespace) -> int:
    network = SubgraphNetwork()
    prediction_service = TrafficPredictionService(network)
    defaults = load_defaults()
    engine = RoutingEngine(
        network,
        prediction_service,
        speed_limit_kmh=defaults["default_speed_limit_kmh"],
        intersection_delay_seconds=defaults["default_intersection_delay_seconds"],
    )
    result = engine.recommend_routes(
        origin=args.origin,
        destination=args.destination,
        predict_day=args.date,
        time_text=args.time,
        model_name=args.model,
        search_method=args.search_method,
        top_k=args.top_k,
        use_congested_branch=args.congested,
        output_plot_path=Path(args.plot),
    )
    print(f"Origin: {result['origin']}")
    print(f"Destination: {result['destination']}")
    print(f"Date: {result['predict_day']} Time: {result['time_text']} ({result['time_index']})")
    print(f"Model: {result['model_name']}")
    print(f"Search method: {result['search_method']}")
    print("Predicted node flows:")
    for node_id, flow in sorted(result["node_flows"].items()):
        print(f"  {node_id}: {flow:.1f}")
    if result["selected_route"] is not None:
        selected = result["selected_route"]
        print(
            f"Selected route ({result['search_method']}): "
            f"{' -> '.join(selected.path)} | {selected.total_minutes:.2f} min"
        )
    print("Top routes:")
    for route in result["routes"]:
        print(f"  Route {route.rank}: {' -> '.join(route.path)} | {route.total_minutes:.2f} min")
    print(f"Plot: {result['plot_path']}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.gui:
        from .gui import launch_gui

        launch_gui()
        return 0
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
