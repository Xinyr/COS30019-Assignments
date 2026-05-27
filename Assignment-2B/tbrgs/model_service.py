from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from GRU import get_locations as get_gru_locations
from GRU import predict_gru_with_results, train_gru_model
from LSTM import get_locations as get_lstm_locations
from LSTM import predict_lstm_with_results, train_lstm_model
from RandomForest import get_locations as get_rf_locations
from RandomForest import predict_random_forest_with_results, train_random_forest_model

from .config import PROJECT_ROOT
from .network import SubgraphNetwork


RAW_DATA_PATH = PROJECT_ROOT / "Scats Data October 2006.xls"


def _load_raw_scats_data(data_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(data_path)
    df = pd.read_excel(path, sheet_name="Data")
    if "SCATS Number" not in df.columns:
        df = pd.read_excel(path, sheet_name="Data", header=1)
    return df.copy()


class TrafficPredictionService:
    def __init__(self, network: SubgraphNetwork, data_path: str | Path = RAW_DATA_PATH):
        self.network = network
        self.data_path = Path(data_path)
        self.raw_df = _load_raw_scats_data(self.data_path)
        self.raw_df["site_id"] = self.raw_df["SCATS Number"].astype(int).astype(str)

        self.site_to_locations: dict[str, list[str]] = {}
        for site_id, group in self.raw_df.groupby("site_id"):
            locations = list(dict.fromkeys(group["Location"].tolist()))
            self.site_to_locations[site_id] = locations

        self.representative_location: dict[str, str] = {
            site_id: locations[0] for site_id, locations in self.site_to_locations.items()
        }

        self.model_locations = {
            "lstm": list(get_lstm_locations(self.data_path)),
            "gru": list(get_gru_locations(self.data_path)),
            "random_forest": list(get_rf_locations(self.data_path)),
        }
        self.location_to_index = {
            model_name: {name: idx for idx, name in enumerate(locations)}
            for model_name, locations in self.model_locations.items()
        }

        self.relevant_source_sites = sorted(
            {
                site_id
                for node in self.network.nodes.values()
                for site_id in node.prediction_scats_ids
            }
        )
        self.relevant_locations = sorted(
            {
                self.get_representative_location(site_id)
                for site_id in self.relevant_source_sites
            }
        )

        self.deep_training_cache: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
            "lstm": {},
            "gru": {},
        }
        self.rf_results_cache: dict[str, Any] | None = None
        self.node_flow_cache: dict[tuple[str, str, int], dict[str, float]] = {}

    def get_representative_location(self, site_id: str) -> str:
        if site_id not in self.representative_location:
            raise KeyError(f"SCATS site {site_id} is not present in the October 2006 workbook.")
        return self.representative_location[site_id]

    def _get_deep_results(self, model_name: str, location_name: str, time_index: int) -> dict[str, Any]:
        cache = self.deep_training_cache[model_name]
        key = (location_name, time_index)
        if key in cache:
            return cache[key]

        if location_name not in self.location_to_index[model_name]:
            raise KeyError(f"Location '{location_name}' is not available for {model_name}.")

        loc_index = self.location_to_index[model_name][location_name]
        if model_name == "lstm":
            results = train_lstm_model(loc_index=loc_index, time_index=time_index, data_path=self.data_path)
        elif model_name == "gru":
            results = train_gru_model(loc_index=loc_index, time_index=time_index, data_path=self.data_path)
        else:
            raise ValueError(f"Unsupported deep model: {model_name}")

        cache[key] = results
        return results

    def _get_random_forest_results(self) -> dict[str, Any]:
        if self.rf_results_cache is None:
            self.rf_results_cache = train_random_forest_model(
                data_path=self.data_path,
                allowed_locations=self.relevant_locations,
                eval_loc_index=0,
                eval_time_index=0,
            )
        return self.rf_results_cache

    def predict_source_site(
        self,
        model_name: str,
        site_id: str,
        predict_day: str,
        time_index: int,
    ) -> float:
        location_name = self.get_representative_location(site_id)

        if model_name == "lstm":
            results = self._get_deep_results("lstm", location_name, time_index)
            prediction = predict_lstm_with_results(results, predict_day)
        elif model_name == "gru":
            results = self._get_deep_results("gru", location_name, time_index)
            prediction = predict_gru_with_results(results, predict_day)
        elif model_name == "random_forest":
            results = self._get_random_forest_results()
            loc_index = list(results["locations"]).index(location_name)
            prediction = predict_random_forest_with_results(results, predict_day, loc_index, time_index)
        else:
            raise ValueError(f"Unknown model '{model_name}'.")

        return float(prediction[0])

    def predict_node_flow(
        self,
        model_name: str,
        node_id: str,
        predict_day: str,
        time_index: int,
        _stack: set[str] | None = None,
    ) -> float:
        if _stack is None:
            _stack = set()
        if node_id in _stack:
            raise RuntimeError(f"Cyclic proxy configuration detected while resolving node {node_id}.")
        _stack.add(node_id)

        node = self.network.nodes[node_id]
        if node.prediction_scats_ids:
            values = [
                self.predict_source_site(model_name, site_id, predict_day, time_index)
                for site_id in node.prediction_scats_ids
            ]
        elif node.proxy_nodes:
            values = [
                self.predict_node_flow(model_name, proxy_node, predict_day, time_index, _stack=_stack)
                for proxy_node in node.proxy_nodes
            ]
        else:
            raise ValueError(f"Node {node_id} has no prediction source configured.")

        _stack.remove(node_id)
        return sum(values) / len(values)

    def predict_all_node_flows(
        self,
        model_name: str,
        predict_day: str,
        time_index: int,
    ) -> dict[str, float]:
        cache_key = (model_name, predict_day, time_index)
        if cache_key in self.node_flow_cache:
            return self.node_flow_cache[cache_key]

        flows = {
            node_id: self.predict_node_flow(model_name, node_id, predict_day, time_index)
            for node_id in self.network.nodes
        }
        self.node_flow_cache[cache_key] = flows
        return flows
