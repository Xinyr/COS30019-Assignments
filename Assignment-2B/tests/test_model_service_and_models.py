from __future__ import annotations

import unittest

from GRU import get_locations as get_gru_locations
from LSTM import get_locations as get_lstm_locations
from RandomForest import get_locations as get_rf_locations, load_scats_data
from tbrgs.model_service import TrafficPredictionService
from tbrgs.network import SubgraphNetwork


class ModelServiceAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = SubgraphNetwork()
        cls.service = TrafficPredictionService(cls.network)

    def test_model_service_has_relevant_source_sites(self) -> None:
        self.assertIn("3120", self.service.relevant_source_sites)

    def test_model_service_has_relevant_locations(self) -> None:
        self.assertGreater(len(self.service.relevant_locations), 0)

    def test_representative_location_lookup(self) -> None:
        location = self.service.get_representative_location("3120")
        self.assertIsInstance(location, str)
        self.assertTrue(location)

    def test_random_forest_loader_has_value_columns(self) -> None:
        df = load_scats_data()
        self.assertIn("Location", df.columns)
        self.assertIn("V00", df.columns)
        self.assertIn("V95", df.columns)

    def test_lstm_locations_are_available(self) -> None:
        self.assertGreater(len(get_lstm_locations()), 0)

    def test_gru_locations_are_available(self) -> None:
        self.assertGreater(len(get_gru_locations()), 0)

    def test_random_forest_locations_are_available(self) -> None:
        self.assertGreater(len(get_rf_locations()), 0)


if __name__ == "__main__":
    unittest.main()
