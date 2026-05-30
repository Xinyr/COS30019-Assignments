from __future__ import annotations

import unittest

from tbrgs.config import load_defaults, load_subgraph
from tbrgs.network import SubgraphNetwork


class ConfigAndNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.defaults = load_defaults()
        cls.subgraph = load_subgraph()
        cls.network = SubgraphNetwork(cls.subgraph)

    def test_defaults_include_core_keys(self) -> None:
        expected = {
            "default_model",
            "default_origin",
            "default_destination",
            "default_date",
            "default_time",
            "default_top_k",
        }
        self.assertTrue(expected.issubset(self.defaults))

    def test_subgraph_has_expected_node_count(self) -> None:
        self.assertEqual(len(self.subgraph["nodes"]), 9)

    def test_network_contains_expected_nodes(self) -> None:
        for node_id in ["3662", "4032", "3120", "3126"]:
            self.assertIn(node_id, self.network.nodes)

    def test_neighbors_for_4032(self) -> None:
        self.assertEqual(self.network.neighbors("4032"), ["3120", "3180", "3662"])

    def test_neighbors_for_3120(self) -> None:
        self.assertEqual(self.network.neighbors("3120"), ["3122", "4032"])

    def test_distance_is_positive_and_symmetric(self) -> None:
        forward = self.network.distance_km("3120", "3122")
        backward = self.network.distance_km("3122", "3120")
        self.assertGreater(forward, 0.0)
        self.assertAlmostEqual(forward, backward, places=9)

    def test_known_path_exists(self) -> None:
        paths = self.network.all_simple_paths("3662", "3126")
        self.assertIn(
            ["3662", "4032", "3180", "3127", "3126"],
            paths,
        )

    def test_scaled_positions_include_all_nodes(self) -> None:
        positions = self.network.scaled_positions(width=640, height=480, padding=40)
        self.assertEqual(set(positions), set(self.network.nodes))
        for x, y in positions.values():
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)


if __name__ == "__main__":
    unittest.main()
