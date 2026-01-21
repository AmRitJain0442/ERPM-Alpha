"""
Hybrid EGARCH + XGBoost Volatility Model

This module implements the "Feature Extraction" strategy:
1. EGARCH captures baseline volatility (physics of clustering)
2. XGBoost corrects using news features (non-linear shock adjustments)

Why This Works Better Than GARCH-X:
- Python's `arch` library doesn't support external regressors in variance equation
- EGARCH handles volatility clustering and asymmetry
- XGBoost captures complex news interactions that GARCH can't model
- Two-stage approach is industry SOTA for volatility + sentiment modeling

Pipeline:
[Returns] -> [EGARCH] -> [Conditional Vol] -> [XGBoost + GDELT] -> [Adjusted Vol/Price]
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from egarch_model import EGARCHVolatilityModel
from config import XGBOOST_CONFIG, TRAIN_CONFIG, ASYMMETRY_CONFIG

warnings.filterwarnings('ignore')


class HybridVolatilityModel:
    """
    Hybrid EGARCH + XGBoost model for volatility-adjusted price prediction.

    Stage 1: EGARCH extracts conditional volatility (baseline risk estimate)
    Stage 2: XGBoost uses GARCH_Vol + News features to predict adjusted values

    The key insight: GARCH tells us "volatility should be X based on history"
    but XGBoost can say "wait, the Panic Index is 80%, bump it up."
    """

    def __init__(
        self,
        egarch_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        target_type: str = "price"  # 'price', 'returns', or 'volatility'
    ):
        """
        Initialize hybrid model.

        Args:
            egarch_params: Parameters for EGARCH model
            xgb_params: Parameters for XGBoost model
            target_type: What to predict ('price', 'returns', 'volatility')
        """
        self.egarch_params = egarch_params or {
            "p": 1, "q": 1, "o": 1, "dist": "skewt", "model_type": "EGARCH"
        }
        self.xgb_params = xgb_params or XGBOOST_CONFIG

        self.target_type = target_type
        self.egarch_model = None
        self.xgb_model = None
        self.feature_names = None
        self.is_fitted = False

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features for the hybrid model.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()

        # Panic Index: Composite indicator of market stress
        if "Tone_Conflict" in df.columns and "Tone_Overall" in df.columns:
            # Higher conflict + lower overall tone = more panic
            df["Panic_Index"] = (
                -df["Tone_Overall"] +
                abs(df["Tone_Conflict"]) +
                df.get("Volume_Spike", 0) * 0.1
            ).clip(0, 100)

        # Diff Stability: How stable is sentiment?
        if "Tone_Overall" in df.columns:
            df["Diff_Stability"] = df["Tone_Overall"].rolling(5).std().fillna(0)

        # News Shock: Unexpected component of news
        if "Goldstein_Avg" in df.columns:
            ma = df["Goldstein_Avg"].rolling(10).mean()
            df["News_Shock"] = (df["Goldstein_Avg"] - ma).fillna(0)

        # Asymmetric shock indicators
        if "Tone_Overall" in df.columns:
            panic_thresh = ASYMMETRY_CONFIG["panic_threshold"]
            relief_thresh = ASYMMETRY_CONFIG["relief_threshold"]

            df["Is_Panic"] = (df["Tone_Overall"] < panic_thresh).astype(int)
            df["Is_Relief"] = (df["Tone_Overall"] > relief_thresh).astype(int)
            df["Panic_Magnitude"] = np.where(
                df["Tone_Overall"] < panic_thresh,
                abs(df["Tone_Overall"] - panic_thresh),
                0
            )

        # Lagged features for momentum
        if "GARCH_Vol" in df.columns:
            df["GARCH_Vol_Lag1"] = df["GARCH_Vol"].shift(1)
            df["GARCH_Vol_Change"] = df["GARCH_Vol"].pct_change()

        return df

    def prepare_data(
        self,
        price_series: pd.Series,
        features_df: pd.DataFrame,
        scale_returns: float = 100.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for hybrid model training.

        Args:
            price_series: Price series (e.g., INR exchange rate)
            features_df: DataFrame with GDELT and other features
            scale_returns: Scale factor for returns

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Ensure alignment
        df = features_df.copy()

        # Add price and compute returns
        df["Price"] = price_series
        df["Returns"] = df["Price"].pct_change() * scale_returns

        # Realized volatility (proxy: squared returns)
        df["Realized_Vol"] = df["Returns"] ** 2

        # Rolling realized volatility
        df["Realized_Vol_5d"] = df["Returns"].rolling(5).std()
        df["Realized_Vol_20d"] = df["Returns"].rolling(20).std()

        # Drop NaN from returns calculation
        df = df.dropna(subset=["Returns"])

        # Stage 1: Fit EGARCH to get conditional volatility
        print("Stage 1: Fitting EGARCH model...")
        self.egarch_model = EGARCHVolatilityModel(**self.egarch_params)
        self.egarch_model.fit(df["Returns"])

        # Add GARCH conditional volatility as feature
        df["GARCH_Vol"] = self.egarch_model.get_conditional_volatility()

        # Compute derived features
        df = self._compute_derived_features(df)

        # Create target variable
        if self.target_type == "price":
            df["Target"] = df["Price"].shift(-1)  # Next day price
        elif self.target_type == "returns":
            df["Target"] = df["Returns"].shift(-1)  # Next day returns
        else:  # volatility
            df["Target"] = df["Realized_Vol"].shift(-1)  # Next day vol

        # Drop rows with NaN target
        df = df.dropna(subset=["Target"])

        return df

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "Target",
        eval_set: bool = True
    ) -> Dict:
        """
        Fit the XGBoost correction model.

        Args:
            df: Prepared DataFrame with GARCH_Vol and features
            feature_cols: List of feature column names
            target_col: Target column name
            eval_set: Whether to use validation set for early stopping

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_cols

        # Time-based split (no shuffling for time series!)
        train_size = int(len(df) * TRAIN_CONFIG["train_ratio"])
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        print(f"\nStage 2: Training XGBoost Hybrid Model...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(feature_cols)}")

        # Initialize XGBoost
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)

        # Fit with early stopping if validation set
        if eval_set:
            val_size = int(len(X_train) * TRAIN_CONFIG["validation_ratio"])
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_fit = X_train.iloc[:-val_size]
            y_train_fit = y_train.iloc[:-val_size]

            self.xgb_model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train, y_train)

        self.is_fitted = True

        # Evaluate
        train_pred = self.xgb_model.predict(X_train)
        test_pred = self.xgb_model.predict(X_test)

        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        # Store predictions for analysis
        self.train_predictions = pd.Series(train_pred, index=train_df.index)
        self.test_predictions = pd.Series(test_pred, index=test_df.index)
        self.test_actual = y_test

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the hybrid model.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.xgb_model.predict(X[self.feature_names])

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from XGBoost.

        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": self.xgb_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        return importance

    def get_garch_diagnostics(self) -> Dict:
        """Get diagnostics from the EGARCH stage."""
        if self.egarch_model is None:
            raise ValueError("EGARCH model not fitted")

        return self.egarch_model._extract_diagnostics()

    def analyze_asymmetry(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the asymmetric response to good vs bad news.

        Args:
            df: DataFrame with predictions and features

        Returns:
            Dictionary with asymmetry analysis
        """
        if "Is_Panic" not in df.columns:
            return {"error": "Panic indicators not computed"}

        # Separate panic vs relief days
        panic_days = df[df["Is_Panic"] == 1]
        relief_days = df[df["Is_Relief"] == 1]
        normal_days = df[(df["Is_Panic"] == 0) & (df["Is_Relief"] == 0)]

        analysis = {
            "panic_days_count": len(panic_days),
            "relief_days_count": len(relief_days),
            "normal_days_count": len(normal_days),
        }

        if "GARCH_Vol" in df.columns:
            analysis["avg_vol_panic"] = panic_days["GARCH_Vol"].mean()
            analysis["avg_vol_relief"] = relief_days["GARCH_Vol"].mean()
            analysis["avg_vol_normal"] = normal_days["GARCH_Vol"].mean()
            analysis["vol_asymmetry_ratio"] = (
                analysis["avg_vol_panic"] / analysis["avg_vol_relief"]
                if analysis["avg_vol_relief"] > 0 else np.inf
            )

        return analysis


class HybridPipeline:
    """
    Complete pipeline for EGARCH + XGBoost hybrid modeling.

    This class orchestrates the entire workflow from data loading
    to model evaluation and forecasting.
    """

    def __init__(self, target_type: str = "price"):
        """
        Initialize the pipeline.

        Args:
            target_type: What to predict ('price', 'returns', 'volatility')
        """
        self.target_type = target_type
        self.model = None
        self.data = None
        self.results = None

    def load_and_merge_data(
        self,
        price_path: str,
        features_path: str,
        price_col: str = "USD_to_INR",
        date_col: str = "Date"
    ) -> pd.DataFrame:
        """
        Load and merge price and feature data.

        Args:
            price_path: Path to price CSV
            features_path: Path to features CSV
            price_col: Name of price column
            date_col: Name of date column

        Returns:
            Merged DataFrame
        """
        # Load data
        price_df = pd.read_csv(price_path, parse_dates=[date_col])
        features_df = pd.read_csv(features_path, parse_dates=[date_col])

        # Merge on date
        df = pd.merge(features_df, price_df[[date_col, price_col]], on=date_col, how="inner")
        df = df.set_index(date_col).sort_index()

        # Rename price column
        df = df.rename(columns={price_col: "Price"})

        self.data = df
        return df

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        price_col: str = "Price"
    ) -> Dict:
        """
        Run the complete hybrid modeling pipeline.

        Args:
            df: DataFrame with all features
            feature_cols: List of feature columns to use
            price_col: Name of price column

        Returns:
            Dictionary with all results
        """
        print("=" * 70)
        print("  HYBRID EGARCH + XGBoost VOLATILITY MODEL")
        print("=" * 70)

        # Initialize model
        self.model = HybridVolatilityModel(target_type=self.target_type)

        # Prepare data (runs EGARCH)
        prepared_df = self.model.prepare_data(
            price_series=df[price_col],
            features_df=df[feature_cols + [price_col]]
        )

        # Define final feature set (GARCH_Vol + original + derived)
        all_features = [col for col in prepared_df.columns if col not in [
            "Target", "Price", "Returns", "Realized_Vol"
        ] and prepared_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        # Remove any features with all NaN
        all_features = [f for f in all_features if prepared_df[f].notna().sum() > 0]

        # Fill remaining NaN with 0 or forward fill
        prepared_df[all_features] = prepared_df[all_features].fillna(method='ffill').fillna(0)

        print(f"\nFeatures used: {len(all_features)}")
        for f in all_features[:10]:
            print(f"  - {f}")
        if len(all_features) > 10:
            print(f"  ... and {len(all_features) - 10} more")

        # Train model
        metrics = self.model.fit(prepared_df, all_features)

        # Get diagnostics
        garch_diag = self.model.get_garch_diagnostics()
        feature_imp = self.model.get_feature_importance()
        asymmetry = self.model.analyze_asymmetry(prepared_df)

        # Compile results
        self.results = {
            "metrics": metrics,
            "garch_diagnostics": garch_diag,
            "feature_importance": feature_imp,
            "asymmetry_analysis": asymmetry,
            "prepared_data": prepared_df,
            "test_predictions": self.model.test_predictions,
            "test_actual": self.model.test_actual,
        }

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print results summary."""
        if self.results is None:
            return

        metrics = self.results["metrics"]
        garch = self.results["garch_diagnostics"]
        asymmetry = self.results["asymmetry_analysis"]

        print("\n" + "=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)

        print("\n[EGARCH Stage]")
        print(f"  Model: {garch['model_type']}")
        print(f"  Distribution: {garch['distribution']}")
        print(f"  Leverage Effect: {garch['leverage_effect']}")
        print(f"  Gamma (Asymmetry): {garch['gamma']:.4f}")
        print(f"  Persistence: {garch['persistence']:.4f}")
        print(f"  AIC: {garch['aic']:.2f}")

        print("\n[XGBoost Hybrid Stage]")
        print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']:.4f}")

        print("\n[Asymmetry Analysis]")
        if "avg_vol_panic" in asymmetry:
            print(f"  Panic Days: {asymmetry['panic_days_count']}")
            print(f"  Relief Days: {asymmetry['relief_days_count']}")
            print(f"  Avg Vol (Panic): {asymmetry['avg_vol_panic']:.4f}")
            print(f"  Avg Vol (Relief): {asymmetry['avg_vol_relief']:.4f}")
            print(f"  Asymmetry Ratio: {asymmetry['vol_asymmetry_ratio']:.2f}x")

        print("\n[Top 5 Features]")
        top_features = self.results["feature_importance"].head(5)
        for _, row in top_features.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")


def main():
    """Demonstrate hybrid model with sample data."""
    print("=" * 70)
    print("  HYBRID MODEL DEMONSTRATION")
    print("=" * 70)

    # Generate synthetic data for demonstration
    np.random.seed(42)
    n = 300

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    # Simulate exchange rate with volatility clustering
    price = [85.0]
    vol = [0.001]

    for t in range(1, n):
        # GARCH-like volatility
        shock = np.random.standard_t(5) * 0.01
        vol_t = 0.0001 + 0.1 * shock**2 + 0.85 * vol[-1]

        # Asymmetric effect
        if shock < 0:
            vol_t += 0.05 * shock**2

        vol.append(vol_t)
        price.append(price[-1] * (1 + np.sqrt(vol_t) * shock))

    # Simulate GDELT features
    df = pd.DataFrame({
        "Date": dates,
        "Price": price,
        "Tone_Overall": np.random.normal(-1, 2, n),
        "Tone_Conflict": np.random.normal(-3, 2, n),
        "Tone_Economy": np.random.normal(-1, 1.5, n),
        "Goldstein_Avg": np.random.normal(0.3, 0.5, n),
        "Volume_Spike": np.random.exponential(5, n),
    })
    df = df.set_index("Date")

    # Run pipeline
    pipeline = HybridPipeline(target_type="price")

    feature_cols = ["Tone_Overall", "Tone_Conflict", "Tone_Economy",
                    "Goldstein_Avg", "Volume_Spike"]

    results = pipeline.run_full_pipeline(df, feature_cols)

    print("\n[Demonstration complete - use real data for production]")


if __name__ == "__main__":
    main()
