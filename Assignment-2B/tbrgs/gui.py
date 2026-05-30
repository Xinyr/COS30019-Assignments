from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

from .config import PROJECT_ROOT, load_defaults
from .network import RouteRecommendation, SubgraphNetwork

if TYPE_CHECKING:
    from .model_service import TrafficPredictionService
    from .router import RoutingEngine


class TBRGSGui(tk.Tk):
    FOREGROUND_TAG = "node_foreground"

    def __init__(self):
        super().__init__()
        self.title("Traffic-Based Route Guidance System")
        self.geometry("1320x840")

        self.defaults = load_defaults()
        self.network = SubgraphNetwork()
        self.prediction_service: TrafficPredictionService | None = None
        self.router: RoutingEngine | None = None
        self.node_labels = [
            f"{node_id} - {node.label}" for node_id, node in sorted(self.network.nodes.items())
        ]
        self.label_to_id = {label: label.split(" - ", 1)[0] for label in self.node_labels}
        self.canvas_width = 930
        self.canvas_height = 560
        self.positions = self.network.scaled_positions(
            width=self.canvas_width,
            height=self.canvas_height,
            padding=90,
        )
        self.route_line_ids: list[int] = []
        self.node_canvas_ids: dict[str, int] = {}
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.run_thread: threading.Thread | None = None

        self._build_layout()
        self._draw_base_graph()
        self.status_var.set("Ready. Click 'Run TBRGS' to load the traffic models and search routes.")

    def _ensure_router(self) -> None:
        if self.router is not None:
            return

        from .model_service import TrafficPredictionService
        from .router import RoutingEngine

        self.prediction_service = TrafficPredictionService(self.network)
        self.router = RoutingEngine(
            self.network,
            self.prediction_service,
            speed_limit_kmh=self.defaults["default_speed_limit_kmh"],
            intersection_delay_seconds=self.defaults["default_intersection_delay_seconds"],
        )

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        controls = ttk.Frame(root)
        controls.pack(side="left", fill="y", padx=(0, 12))

        viewer = ttk.Frame(root)
        viewer.pack(side="right", fill="both", expand=True)

        ttk.Label(controls, text="Origin").pack(anchor="w")
        self.origin_var = tk.StringVar(value=self._label_for_node(self.defaults["default_origin"]))
        ttk.Combobox(controls, textvariable=self.origin_var, values=self.node_labels, width=32).pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Destination").pack(anchor="w")
        self.destination_var = tk.StringVar(value=self._label_for_node(self.defaults["default_destination"]))
        ttk.Combobox(controls, textvariable=self.destination_var, values=self.node_labels, width=32).pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Date (MM/DD/YYYY)").pack(anchor="w")
        self.date_var = tk.StringVar(value=self.defaults["default_date"])
        ttk.Entry(controls, textvariable=self.date_var, width=34).pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Time (HH:MM)").pack(anchor="w")
        self.time_var = tk.StringVar(value=self.defaults["default_time"])
        ttk.Entry(controls, textvariable=self.time_var, width=34).pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Traffic Model").pack(anchor="w")
        self.model_var = tk.StringVar(value=self.defaults["default_model"])
        ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=["random_forest", "lstm", "gru"],
            width=32,
        ).pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Search Method").pack(anchor="w")
        self.search_method_var = tk.StringVar(
            value=self.defaults.get("default_search_method", "astar")
        )
        ttk.Combobox(
            controls,
            textvariable=self.search_method_var,
            values=["ranked_paths", "astar", "gbfs", "bfs", "dfs", "cus1", "cus2"],
            width=32,
        ).pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Top-k Routes").pack(anchor="w")
        self.top_k_var = tk.IntVar(value=int(self.defaults["default_top_k"]))
        ttk.Spinbox(controls, from_=1, to=5, textvariable=self.top_k_var, width=8).pack(anchor="w", pady=(0, 8))

        self.congested_var = tk.BooleanVar(value=bool(self.defaults["default_use_congested_branch"]))
        ttk.Checkbutton(
            controls,
            text="Use congested flow-speed branch",
            variable=self.congested_var,
        ).pack(anchor="w", pady=(0, 12))

        self.run_button = ttk.Button(controls, text="Run TBRGS", command=self._run_route_search)
        self.run_button.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(controls, textvariable=self.status_var, wraplength=250).pack(anchor="w", pady=(12, 0))

        self.canvas = tk.Canvas(
            viewer,
            width=self.canvas_width,
            height=self.canvas_height,
            background="white",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.canvas.pack(fill="x")

        ttk.Label(viewer, text="Route Output").pack(anchor="w", pady=(12, 4))
        self.output_text = tk.Text(viewer, height=18, wrap="word")
        self.output_text.pack(fill="both", expand=True)

    def _label_for_node(self, node_id: str) -> str:
        node = self.network.nodes[node_id]
        return f"{node_id} - {node.label}"

    def _draw_base_graph(self) -> None:
        self.canvas.delete("all")
        self.route_line_ids.clear()
        self.node_canvas_ids.clear()
        text_placements = self.network.text_placements()

        for start, neighbors in self.network.adjacency.items():
            for end in neighbors:
                if start < end:
                    x1, y1 = self.positions[start]
                    x2, y2 = self.positions[end]
                    self.canvas.create_line(x1, y1, x2, y2, fill="#c0c0c0", width=2)

        for node_id, node in self.network.nodes.items():
            x, y = self.positions[node_id]
            placement = text_placements[node_id]
            oval = self.canvas.create_oval(
                x - 10,
                y - 10,
                x + 10,
                y + 10,
                fill="#111111",
                tags=(self.FOREGROUND_TAG,),
            )
            self.node_canvas_ids[node_id] = oval
            self._create_text_with_background(
                x + placement.id_dx,
                y + placement.id_dy,
                text=node_id,
                font=("Arial", 9, "bold"),
                anchor=self._canvas_anchor(placement.id_dx, placement.id_dy),
            )
            self._create_text_with_background(
                x + placement.label_dx,
                y + placement.label_dy,
                text=self.network.format_canvas_label(node.label),
                font=("Arial", 8),
                anchor=self._canvas_anchor(placement.label_dx, placement.label_dy),
                justify=self._canvas_justify(placement.label_dx),
            )

    def _draw_routes(self, routes: list[RouteRecommendation]) -> None:
        for item_id in self.route_line_ids:
            self.canvas.delete(item_id)
        self.route_line_ids.clear()

        palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        for idx, route in enumerate(routes):
            color = palette[idx % len(palette)]
            for start, end in zip(route.path[:-1], route.path[1:]):
                x1, y1 = self.positions[start]
                x2, y2 = self.positions[end]
                self.route_line_ids.append(
                    self.canvas.create_line(x1, y1, x2, y2, fill=color, width=4 if idx == 0 else 2)
                )
        self.canvas.tag_raise(self.FOREGROUND_TAG)

    def _create_text_with_background(
        self,
        x: float,
        y: float,
        text: str,
        font: tuple[str, int] | tuple[str, int, str],
        anchor: str = "center",
        justify: str = "center",
    ) -> None:
        text_id = self.canvas.create_text(
            x,
            y,
            text=text,
            font=font,
            anchor=anchor,
            justify=justify,
            tags=(self.FOREGROUND_TAG,),
        )
        bbox = self.canvas.bbox(text_id)
        if bbox is None:
            return
        left, top, right, bottom = bbox
        bg_id = self.canvas.create_rectangle(
            left - 5,
            top - 3,
            right + 5,
            bottom + 3,
            fill="white",
            outline="",
            tags=(self.FOREGROUND_TAG,),
        )
        self.canvas.tag_lower(bg_id, text_id)

    @staticmethod
    def _canvas_anchor(dx: float, dy: float) -> str:
        vertical = ""
        horizontal = ""
        if dy > 6:
            vertical = "n"
        elif dy < -6:
            vertical = "s"
        if dx > 8:
            horizontal = "w"
        elif dx < -8:
            horizontal = "e"
        return (vertical + horizontal) or "center"

    @staticmethod
    def _canvas_justify(dx: float) -> str:
        if dx > 8:
            return "left"
        if dx < -8:
            return "right"
        return "center"

    def _run_route_search_worker(self, args: dict) -> None:
        try:
            self._ensure_router()
            assert self.router is not None
            result = self.router.recommend_routes(**args)
            self.worker_queue.put(("success", result))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def _poll_worker_queue(self) -> None:
        try:
            status, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            if self.run_thread is not None and self.run_thread.is_alive():
                self.after(150, self._poll_worker_queue)
            return

        self.run_thread = None
        self.run_button.config(state="normal")

        if status == "success":
            result = payload
            assert isinstance(result, dict)
            self._draw_base_graph()
            self._draw_routes(result["routes"])
            self._render_output(result)
            self.status_var.set(
                f"Completed using {result['model_name']}. Plot saved to {result['plot_path']}."
            )
            return

        exc = payload
        assert isinstance(exc, Exception)
        messagebox.showerror("TBRGS Error", str(exc))
        self.status_var.set("Run failed.")

    def _run_route_search(self) -> None:
        if self.run_thread is not None and self.run_thread.is_alive():
            self.status_var.set("A route search is already running.")
            return

        try:
            origin = self.label_to_id[self.origin_var.get()]
            destination = self.label_to_id[self.destination_var.get()]
            plot_path = PROJECT_ROOT / "outputs" / "latest_routes.png"
            args = dict(
                origin=origin,
                destination=destination,
                predict_day=self.date_var.get(),
                time_text=self.time_var.get(),
                model_name=self.model_var.get(),
                search_method=self.search_method_var.get(),
                top_k=int(self.top_k_var.get()),
                use_congested_branch=bool(self.congested_var.get()),
                output_plot_path=plot_path,
            )

            model_name = self.model_var.get()
            loading_message = "Running route search."
            if model_name in {"lstm", "gru"}:
                loading_message = (
                    "Running route search and training the selected deep model. "
                    "This can take a while, but the window should stay responsive."
                )

            self.status_var.set(loading_message)
            self.run_button.config(state="disabled")

            self.run_thread = threading.Thread(
                target=self._run_route_search_worker,
                args=(args,),
                daemon=True,
            )
            self.run_thread.start()
            self.after(150, self._poll_worker_queue)
        except Exception as exc:
            messagebox.showerror("TBRGS Error", str(exc))
            self.status_var.set("Run failed.")

    def _render_output(self, result: dict) -> None:
        lines = [
            f"Origin: {result['origin']}",
            f"Destination: {result['destination']}",
            f"Date: {result['predict_day']}",
            f"Time: {result['time_text']} (index {result['time_index']})",
            f"Model: {result['model_name']}",
            f"Search method: {result['search_method']}",
            "",
            "Predicted node flows (15-minute volume):",
        ]
        for node_id, flow in sorted(result["node_flows"].items()):
            lines.append(f"  {node_id}: {flow:.1f}")

        if result["selected_route"] is not None:
            selected = result["selected_route"]
            lines.extend(
                [
                    "",
                    f"Selected route ({result['search_method']}):",
                    f"  {' -> '.join(selected.path)} | {selected.total_minutes:.2f} minutes",
                ]
            )

        lines.append("")
        lines.append("Top routes:")
        for route in result["routes"]:
            lines.append(
                f"  Route {route.rank}: {' -> '.join(route.path)} | {route.total_minutes:.2f} minutes"
            )
            for edge in route.edges:
                lines.append(
                    "    "
                    f"{edge.start}->{edge.end}: {edge.distance_km:.2f} km, "
                    f"flow={edge.quarter_hour_flow:.1f}/15min ({edge.hourly_flow:.1f}/h), "
                    f"speed={edge.speed_kmh:.1f} km/h, time={edge.travel_minutes:.2f} min"
                )

        if result["plot_path"]:
            lines.extend(["", f"Saved route plot: {result['plot_path']}"])

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "\n".join(lines))


def launch_gui() -> None:
    app = TBRGSGui()
    app.mainloop()
