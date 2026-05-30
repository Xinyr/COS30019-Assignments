from __future__ import annotations

import unittest

from tbrgs.network import SubgraphNetwork
from tbrgs.router import RoutingEngine


class DummyPredictionService:
    def predict_all_node_flows(self, model_name: str, predict_day: str, time_index: int) -> dict[str, float]:
        return {
            "3120": 24.0,
            "3122": 21.0,
            "3126": 19.0,
            "3127": 18.0,
            "3180": 16.0,
            "3662": 20.0,
            "3804": 23.0,
            "3812": 22.0,
            "4032": 17.0,
        }


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = SubgraphNetwork()
        cls.engine = RoutingEngine(cls.network, DummyPredictionService())

    def test_time_to_index_midnight(self) -> None:
        self.assertEqual(self.engine.time_to_index("00:00"), 0)

    def test_time_to_index_afternoon(self) -> None:
        self.assertEqual(self.engine.time_to_index("13:45"), 55)

    def test_time_to_index_invalid_minute_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.time_to_index("08:10")

    def test_recommend_routes_returns_results(self) -> None:
        result = self.engine.recommend_routes(
            origin="3662",
            destination="3126",
            predict_day="10/15/2006",
            time_text="08:00",
            model_name="random_forest",
            top_k=3,
        )
        self.assertEqual(result["origin"], "3662")
        self.assertEqual(result["destination"], "3126")
        self.assertGreaterEqual(len(result["routes"]), 1)

    def test_top_k_is_respected(self) -> None:
        result = self.engine.recommend_routes(
            origin="3662",
            destination="3126",
            predict_day="10/15/2006",
            time_text="08:00",
            model_name="random_forest",
            top_k=2,
        )
        self.assertLessEqual(len(result["routes"]), 2)

    def test_routes_are_ranked_in_order(self) -> None:
        result = self.engine.recommend_routes(
            origin="3662",
            destination="3126",
            predict_day="10/15/2006",
            time_text="08:00",
            model_name="random_forest",
            top_k=4,
        )
        totals = [route.total_minutes for route in result["routes"]]
        self.assertEqual(totals, sorted(totals))

    def test_edge_metrics_are_positive(self) -> None:
        result = self.engine.recommend_routes(
            origin="3662",
            destination="3126",
            predict_day="10/15/2006",
            time_text="08:00",
            model_name="random_forest",
            top_k=1,
        )
        route = result["routes"][0]
        for edge in route.edges:
            self.assertGreater(edge.distance_km, 0.0)
            self.assertGreater(edge.speed_kmh, 0.0)
            self.assertGreater(edge.travel_minutes, 0.0)


if __name__ == "__main__":
    unittest.main()
