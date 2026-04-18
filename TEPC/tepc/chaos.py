"""
Coupled Lorenz dynamics for the TEPC pipeline.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .config import RunConfig


def _laplacian(adjacency: np.ndarray) -> np.ndarray:
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency


def _initial_states(level_window: pd.DataFrame, shock_window: pd.DataFrame) -> np.ndarray:
    states = []
    for column in level_window.columns:
        levels = level_window[column].astype(float)
        shocks = shock_window[column].astype(float).fillna(0.0)
        rng = max(levels.max() - levels.min(), 1e-6)
        range_pos = float((levels.iloc[-1] - levels.min()) / rng)
        vol = float(shocks.tail(10).std()) if len(shocks) > 1 else 0.0
        drift = float(shocks.tail(5).mean()) if len(shocks) > 0 else 0.0

        x0 = np.clip(1.0 + 35.0 * abs(vol) + 4.0 * range_pos, 0.25, 30.0)
        y0 = np.clip(12.0 * drift + 3.0 * (range_pos - 0.5), -20.0, 20.0)
        z0 = np.clip(4.0 + 14.0 * abs(vol) + 8.0 * abs(drift), 0.5, 40.0)
        states.append([x0, y0, z0])

    return np.asarray(states, dtype=float)


def _derivative(states: np.ndarray, laplacian: np.ndarray, epsilon: float, config: RunConfig) -> np.ndarray:
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    coupling_x = laplacian @ x
    coupling_y = laplacian @ y
    coupling_z = laplacian @ z

    dx = config.lorenz_sigma * (y - x) - epsilon * coupling_x
    dy = x * (config.lorenz_rho - z) - y - 0.35 * epsilon * coupling_y
    dz = x * y - config.lorenz_beta * z - 0.15 * epsilon * coupling_z
    return np.column_stack([dx, dy, dz])


def _rk4_step(states: np.ndarray, laplacian: np.ndarray, epsilon: float, config: RunConfig) -> np.ndarray:
    dt = config.integration_dt
    k1 = _derivative(states, laplacian, epsilon, config)
    k2 = _derivative(states + 0.5 * dt * k1, laplacian, epsilon, config)
    k3 = _derivative(states + 0.5 * dt * k2, laplacian, epsilon, config)
    k4 = _derivative(states + dt * k3, laplacian, epsilon, config)
    next_state = states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state = np.nan_to_num(next_state, nan=0.0, posinf=60.0, neginf=-60.0)
    return np.clip(next_state, -60.0, 60.0)


def _sync_index(x: np.ndarray) -> float:
    pairwise = np.abs(x[:, None] - x[None, :])
    return float(1.0 / (1.0 + pairwise.mean()))


def _target_sync(x: np.ndarray, target_idx: int) -> float:
    return float(1.0 / (1.0 + np.abs(x[target_idx] - x).mean()))


def compute_chaos_features(
    node_frame: pd.DataFrame,
    node_transforms: pd.DataFrame,
    adjacency_by_date: Dict[pd.Timestamp, np.ndarray],
    config: RunConfig,
    target_node: str = "INRUSD",
) -> pd.DataFrame:
    node_names = list(node_frame.columns)
    target_idx = node_names.index(target_node)

    feature_rows = []
    for date, adjacency in adjacency_by_date.items():
        level_window = node_frame.loc[:date].tail(config.chaos_lookback_days)
        shock_window = node_transforms.loc[:date].tail(config.chaos_lookback_days)
        if len(level_window) < config.chaos_lookback_days or level_window.isna().any().any():
            continue

        laplacian = _laplacian(adjacency)
        states = _initial_states(level_window, shock_window)
        perturbed = states.copy()
        perturbed[target_idx, 0] += 1e-7

        sync_values = []
        target_sync_values = []
        target_x_values = []
        target_energy_values = []
        lyapunov_logs = []
        prev_divergence = 1e-7
        total_step = 0
        row = {}

        for epsilon in config.coupling_epsilons:
            stage_sync = []
            stage_target_sync = []
            for _ in range(config.integration_steps):
                states = _rk4_step(states, laplacian, epsilon, config)
                perturbed = _rk4_step(perturbed, laplacian, epsilon, config)

                x = states[:, 0]
                sync_val = _sync_index(x)
                target_sync_val = _target_sync(x, target_idx)
                target_energy = float(np.linalg.norm(states[target_idx]))

                sync_values.append(sync_val)
                target_sync_values.append(target_sync_val)
                target_x_values.append(float(x[target_idx]))
                target_energy_values.append(target_energy)
                stage_sync.append(sync_val)
                stage_target_sync.append(target_sync_val)

                divergence = float(np.linalg.norm(states - perturbed))
                divergence = max(divergence, 1e-12)
                lyapunov_logs.append(np.log(divergence / max(prev_divergence, 1e-12)))
                prev_divergence = divergence
                total_step += 1

            eps_name = int(round(epsilon * 1000))
            row[f"sync_eps_{eps_name:03d}"] = float(np.mean(stage_sync))
            row[f"target_sync_eps_{eps_name:03d}"] = float(stage_target_sync[-1])
            row[f"target_x_eps_{eps_name:03d}"] = float(states[target_idx, 0])

        sync_arr = np.asarray(sync_values, dtype=float)
        target_sync_arr = np.asarray(target_sync_values, dtype=float)
        target_x_arr = np.asarray(target_x_values, dtype=float)
        target_energy_arr = np.asarray(target_energy_values, dtype=float)

        hit_idx = np.where(target_sync_arr >= config.sync_threshold)[0]
        time_to_sync = int(hit_idx[0]) if len(hit_idx) else int(total_step)
        ftle = float(np.mean(lyapunov_logs) / max(config.integration_dt, 1e-6)) if lyapunov_logs else 0.0

        row.update(
            {
                "chaos_sync_mean": float(sync_arr.mean()),
                "chaos_sync_peak": float(sync_arr.max()),
                "chaos_target_sync_mean": float(target_sync_arr.mean()),
                "chaos_target_sync_final": float(target_sync_arr[-1]),
                "chaos_target_x_final": float(target_x_arr[-1]),
                "chaos_target_x_span": float(target_x_arr.max() - target_x_arr.min()),
                "chaos_target_energy_mean": float(target_energy_arr.mean()),
                "chaos_time_to_sync": float(time_to_sync),
                "chaos_ftle": ftle,
                "chaos_coupling_response": float(target_x_arr[-1] - target_x_arr[0]),
            }
        )

        feature_rows.append(pd.Series(row, name=date))

    return pd.DataFrame(feature_rows).sort_index()
