"""
Configuration and experiment definitions for the final-phase system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import re


PACKAGE_DIR = Path(__file__).resolve().parent
MODULE_ROOT = PACKAGE_DIR.parent
REPO_ROOT = MODULE_ROOT.parent


@dataclass
class DataPaths:
    master_dataset: Path = REPO_ROOT / "Super_Master_Dataset.csv"
    bilateral_goldstein: Path = REPO_ROOT / "combined_goldstein_exchange_rates.csv"
    thematic_dataset: Path = REPO_ROOT / "Phase-B" / "merged_training_data.csv"
    political_dataset: Path = REPO_ROOT / "political_news_exchange_merged.csv"


@dataclass
class RunConfig:
    paths: DataPaths = field(default_factory=DataPaths)
    test_days: int = 100
    train_min_days: int = 120
    validation_days: int = 30
    refit_frequency: int = 10
    forecast_horizon_days: int = 1
    embargo_days: int = 1
    rich_only: bool = False
    neutral_return_threshold: float = 0.0005
    persona_limit: int = 8
    random_seed: int = 42
    output_dir: Optional[Path] = None
    rule_persona_weight_cap: float = 0.18
    llm_persona_weight_cap: float = 0.28
    llm_temperature: float = 0.15
    llm_max_tokens: int = 500

    def resolve_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return MODULE_ROOT / "outputs" / stamp


@dataclass
class ExperimentSpec:
    name: str
    include_groups: List[str]
    use_memory_features: bool = True
    persona_backend: Optional[str] = None
    description: str = ""


PERSONAS = [
    {
        "name": "Macro Rates Strategist",
        "weight": 1.2,
        "description": "Focuses on DXY, US10Y, cross-asset rate pressure, and broad dollar strength."
    },
    {
        "name": "Oil And Commodities Analyst",
        "weight": 1.1,
        "description": "Focuses on oil shock transmission to INR through import pressure and risk sentiment."
    },
    {
        "name": "India Sentiment Analyst",
        "weight": 1.0,
        "description": "Focuses on India-specific tone, stability, mentions, panic, and event pressure."
    },
    {
        "name": "Global Risk Analyst",
        "weight": 1.0,
        "description": "Focuses on US sentiment, dollar safety demand, and global risk-off/risk-on behavior."
    },
    {
        "name": "Technical Trend Analyst",
        "weight": 1.1,
        "description": "Focuses on rolling trend, breakout, range position, and momentum."
    },
    {
        "name": "Mean Reversion Analyst",
        "weight": 0.9,
        "description": "Focuses on z-score, reversal pressure, and overstretched moves."
    },
    {
        "name": "Carry And Flow Trader",
        "weight": 1.0,
        "description": "Focuses on carry, yield differential proxies, stability, and flow persistence."
    },
    {
        "name": "Risk Manager",
        "weight": 1.2,
        "description": "Focuses on volatility stress, tail risk, disagreement, and caution."
    },
]


DEFAULT_EXPERIMENTS = [
    ExperimentSpec(
        name="stat_ml_full",
        include_groups=["technical", "macro", "master_sentiment", "goldstein", "thematic", "political", "memory"],
        use_memory_features=True,
        persona_backend=None,
        description="Full statistical + ML stack with all structured feature groups."
    ),
    ExperimentSpec(
        name="stat_ml_no_gdelt",
        include_groups=["technical", "macro"],
        use_memory_features=False,
        persona_backend=None,
        description="Macro + technical ablation without GDELT/news blocks."
    ),
    ExperimentSpec(
        name="stat_ml_no_macro",
        include_groups=["technical", "master_sentiment", "goldstein", "thematic", "political"],
        use_memory_features=False,
        persona_backend=None,
        description="News-heavy ablation without macro cross-asset inputs."
    ),
    ExperimentSpec(
        name="memory_only_no_personas",
        include_groups=["memory"],
        use_memory_features=True,
        persona_backend=None,
        description="Uses only compressed market memory features."
    ),
    ExperimentSpec(
        name="rule_personas",
        include_groups=["technical", "macro", "master_sentiment", "goldstein", "thematic", "political", "memory"],
        use_memory_features=True,
        persona_backend="rule",
        description="Full feature stack plus deterministic market personas."
    ),
]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def build_experiment_specs(selected: List[str], llm_models: List[str]) -> List[ExperimentSpec]:
    experiments = {spec.name: spec for spec in DEFAULT_EXPERIMENTS}

    for model_spec in llm_models:
        name = f"llm_{_slugify(model_spec)}"
        experiments[name] = ExperimentSpec(
            name=name,
            include_groups=["technical", "macro", "master_sentiment", "goldstein", "thematic", "political", "memory"],
            use_memory_features=True,
            persona_backend=model_spec,
            description=f"Full feature stack plus LLM personas via {model_spec}.",
        )

    if selected:
        missing = [name for name in selected if name not in experiments]
        if missing:
            raise ValueError(f"Unknown experiment(s): {missing}")
        return [experiments[name] for name in selected]

    return list(experiments.values())
