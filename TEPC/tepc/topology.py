"""
Dynamic network topology for the TEPC pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from .config import RunConfig


@dataclass
class TopologyResult:
    features: pd.DataFrame
    adjacency_by_date: Dict[pd.Timestamp, np.ndarray]
    node_names: List[str]


def _correlation_to_adjacency(corr: pd.DataFrame, sigma: float) -> np.ndarray:
    clipped = corr.clip(-0.999, 0.999).to_numpy(dtype=float)
    distance = np.sqrt(np.clip(0.5 * (1.0 - clipped), 0.0, None))
    adjacency = np.exp(-(distance / max(sigma, 1e-6)))
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _laplacian(adjacency: np.ndarray) -> np.ndarray:
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency


def _spectral_entropy(eigvals: np.ndarray) -> float:
    positive = np.clip(eigvals[eigvals > 1e-12], 1e-12, None)
    if positive.size == 0:
        return 0.0
    weights = positive / positive.sum()
    entropy = -np.sum(weights * np.log(weights + 1e-12))
    return float(entropy / np.log(len(weights) + 1e-12))


def compute_dynamic_topology(
    node_transforms: pd.DataFrame,
    config: RunConfig,
    target_node: str = "INRUSD",
) -> TopologyResult:
    node_names = list(node_transforms.columns)
    target_idx = node_names.index(target_node)

    feature_rows = []
    adjacency_by_date: Dict[pd.Timestamp, np.ndarray] = {}

    for end_pos in range(config.corr_window - 1, len(node_transforms)):
        window = node_transforms.iloc[end_pos - config.corr_window + 1 : end_pos + 1]
        if window.isna().any().any():
            continue

        date = node_transforms.index[end_pos]
        corr = window.corr().fillna(0.0)
        adjacency = _correlation_to_adjacency(corr, config.topology_sigma)
        lap = _laplacian(adjacency)
        eigvals = np.linalg.eigvalsh(lap)
        degrees = adjacency.sum(axis=1)
        off_diag = adjacency[~np.eye(adjacency.shape[0], dtype=bool)]
        off_diag = off_diag[off_diag > 0]

        row = {
            "target_degree": float(degrees[target_idx]),
            "target_centrality": float(degrees[target_idx] / max(len(node_names) - 1, 1)),
            "network_density": float(off_diag.mean()) if off_diag.size else 0.0,
            "network_tension": float(off_diag.std()) if off_diag.size else 0.0,
            "fiedler_value": float(eigvals[1]) if len(eigvals) > 1 else 0.0,
            "laplacian_trace": float(np.trace(lap)),
            "spectral_entropy": _spectral_entropy(eigvals),
            "target_mean_corr": float(corr.iloc[target_idx].drop(target_node).mean()),
            "target_corr_dispersion": float(corr.iloc[target_idx].drop(target_node).std()),
        }

        for node_pos, node_name in enumerate(node_names):
            if node_name == target_node:
                continue
            row[f"adj_to_{node_name.lower()}"] = float(adjacency[target_idx, node_pos])

        positive_edges = off_diag if off_diag.size else np.array([0.0])
        for quantile in config.filtration_quantiles:
            q_name = int(round(quantile * 100))
            threshold = float(np.quantile(positive_edges, quantile))
            filtered = np.where(adjacency >= threshold, adjacency, 0.0)
            filtered_lap = _laplacian(filtered)
            filtered_eigs = np.linalg.eigvalsh(filtered_lap)
            components = int((filtered_eigs < 1e-8).sum())
            row[f"filtration_components_q{q_name}"] = components
            row[f"filtration_fiedler_q{q_name}"] = float(filtered_eigs[1]) if len(filtered_eigs) > 1 else 0.0
            row[f"filtration_mass_q{q_name}"] = float(filtered.sum() / 2.0)

        feature_rows.append(pd.Series(row, name=date))
        adjacency_by_date[date] = adjacency

    features = pd.DataFrame(feature_rows).sort_index()
    return TopologyResult(
        features=features,
        adjacency_by_date=adjacency_by_date,
        node_names=node_names,
    )
