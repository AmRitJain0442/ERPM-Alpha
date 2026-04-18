"""
Configuration and experiment definitions for the TEPC module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent
MODULE_ROOT = PACKAGE_DIR.parent
REPO_ROOT = MODULE_ROOT.parent


@dataclass
class DataPaths:
    master_dataset: Path = REPO_ROOT / "Super_Master_Dataset.csv"
    thematic_dataset: Path = REPO_ROOT / "Phase-B" / "merged_training_data.csv"
    goldstein_dataset: Path = REPO_ROOT / "combined_goldstein_exchange_rates.csv"
    political_dataset: Path = REPO_ROOT / "political_news_exchange_merged.csv"
    fred_dataset: Path = REPO_ROOT / "data" / "gold_standard" / "fred" / "fred_wide_format_20251230_021943.csv"
    india_raw_gdelt_dataset: Path = REPO_ROOT / "india_news_gz_combined_sorted.csv"
    usa_raw_gdelt_dataset: Path = REPO_ROOT / "usa_news_combined_sorted.csv"
    tepc_market_dataset: Path = MODULE_ROOT / "data" / "market_nodes_daily.csv"
    tepc_gdelt_dataset: Path = MODULE_ROOT / "data" / "gdelt_daily.csv"


@dataclass
class RunConfig:
    paths: DataPaths = field(default_factory=DataPaths)
    test_days: int = 90
    train_min_days: int = 180
    validation_days: int = 45
    forecast_horizon_days: int = 1
    response_lag_days: int = 2
    volatility_window: int = 5
    breakout_threshold: float = 0.005
    corr_window: int = 30
    chaos_lookback_days: int = 20
    topology_sigma: float = 0.35
    filtration_quantiles: Tuple[float, ...] = (0.3, 0.5, 0.7)
    coupling_epsilons: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.35)
    integration_steps: int = 24
    integration_dt: float = 0.02
    lorenz_sigma: float = 10.0
    lorenz_rho: float = 28.0
    lorenz_beta: float = 8.0 / 3.0
    sync_threshold: float = 0.75
    refit_frequency: int = 5
    random_seed: int = 42
    min_node_coverage: float = 0.65
    output_dir: Optional[Path] = None

    def resolve_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return MODULE_ROOT / "outputs" / stamp


@dataclass
class ExperimentSpec:
    name: str
    include_groups: List[str]
    description: str = ""


DEFAULT_EXPERIMENTS = [
    ExperimentSpec(
        name="macro_baseline",
        include_groups=["macro", "alt"],
        description="Macro and alternative-data baseline without topology or chaos.",
    ),
    ExperimentSpec(
        name="topology_only",
        include_groups=["topology"],
        description="Only dynamic topology and persistent Laplacian features.",
    ),
    ExperimentSpec(
        name="chaos_only",
        include_groups=["chaos"],
        description="Only coupled Lorenz synchronization features.",
    ),
    ExperimentSpec(
        name="topology_chaos",
        include_groups=["topology", "chaos"],
        description="Topology and chaos features without the broader macro baseline.",
    ),
    ExperimentSpec(
        name="tepc_full",
        include_groups=["macro", "alt", "topology", "chaos"],
        description="Full TEPC stack with macro, geopolitical, topology, and chaos features.",
    ),
]


def build_experiment_specs(selected: List[str]) -> List[ExperimentSpec]:
    experiments: Dict[str, ExperimentSpec] = {spec.name: spec for spec in DEFAULT_EXPERIMENTS}
    if selected:
        missing = [name for name in selected if name not in experiments]
        if missing:
            raise ValueError(f"Unknown experiment(s): {missing}")
        return [experiments[name] for name in selected]
    return list(experiments.values())
