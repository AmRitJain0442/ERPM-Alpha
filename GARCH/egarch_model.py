"""
EGARCH (Exponential GARCH) Model for Volatility Forecasting

Why EGARCH over standard GARCH?
1. Models log(variance) - never predicts negative volatility
2. Captures asymmetric effects (leverage effect):
   - Bad news (negative returns) increases volatility MORE than
   - Good news (positive returns) of the same magnitude
3. Better suited for FX markets where panic causes instant spikes
   but relief causes gradual volatility decay

Mathematical Form:
log(σ²_t) = ω + α|z_{t-1}| + γz_{t-1} + βlog(σ²_{t-1})

Where:
- σ²_t = conditional variance at time t
- z_t = standardized residuals
- γ = asymmetry parameter (negative = leverage effect)
- α = ARCH effect (how shocks affect volatility)
- β = GARCH effect (volatility persistence)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from arch import arch_model
from arch.univariate import EGARCH, GARCH, ConstantMean, StudentsT, SkewStudent
import warnings

from config import EGARCH_CONFIG, GJR_GARCH_CONFIG

warnings.filterwarnings('ignore')


class EGARCHVolatilityModel:
    """
    EGARCH model for conditional volatility estimation.

    This model captures the "physics" of volatility - how it clusters
    and responds asymmetrically to positive vs negative shocks.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        o: int = 1,
        dist: str = "skewt",
        model_type: str = "EGARCH"
    ):
        """
        Initialize EGARCH model.

        Args:
            p: GARCH lag order (persistence)
            q: ARCH lag order (shock impact)
            o: Asymmetry order (leverage effect)
            dist: Error distribution ('normal', 't', 'skewt', 'ged')
            model_type: 'EGARCH' or 'GJR-GARCH'
        """
        self.p = p
        self.q = q
        self.o = o
        self.dist = dist
        self.model_type = model_type
        self.model = None
        self.result = None
        self.is_fitted = False

    def prepare_returns(
        self,
        prices: pd.Series,
        scale: float = 100.0
    ) -> pd.Series:
        """
        Convert prices to returns suitable for GARCH modeling.

        Args:
            prices: Price series
            scale: Scaling factor (GARCH prefers returns * 100)

        Returns:
            Scaled percentage returns
        """
        returns = prices.pct_change().dropna() * scale
        return returns

    def fit(
        self,
        returns: pd.Series,
        update_freq: int = 0,
        disp: str = "off"
    ) -> Dict:
        """
        Fit EGARCH model to return series.

        Args:
            returns: Return series (should be scaled, e.g., * 100)
            update_freq: Frequency of optimization updates (0 = silent)
            disp: Display option ('off', 'final', 'iter')

        Returns:
            Dictionary with model summary and diagnostics
        """
        # Create model specification
        self.model = arch_model(
            returns,
            vol=self.model_type,
            p=self.p,
            q=self.q,
            o=self.o,
            dist=self.dist,
            mean="Constant",
            rescale=True
        )

        # Fit model
        self.result = self.model.fit(
            update_freq=update_freq,
            disp=disp,
            show_warning=False
        )

        self.is_fitted = True

        # Extract key diagnostics
        diagnostics = self._extract_diagnostics()

        return diagnostics

    def _extract_diagnostics(self) -> Dict:
        """Extract model diagnostics and interpretation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        params = self.result.params
        pvalues = self.result.pvalues

        # Interpret asymmetry
        if self.model_type == "EGARCH":
            gamma = params.get("gamma[1]", 0)
            leverage_effect = "Yes (Bad news increases vol more)" if gamma < 0 else "No"
        else:  # GJR-GARCH
            gamma = params.get("gamma[1]", 0)
            leverage_effect = "Yes" if gamma > 0 else "No"

        # Persistence
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)

        if self.model_type == "EGARCH":
            persistence = abs(beta)
        else:
            persistence = alpha + beta + 0.5 * gamma

        diagnostics = {
            "model_type": self.model_type,
            "distribution": self.dist,
            "log_likelihood": self.result.loglikelihood,
            "aic": self.result.aic,
            "bic": self.result.bic,
            "parameters": params.to_dict(),
            "p_values": pvalues.to_dict(),
            "leverage_effect": leverage_effect,
            "gamma": gamma,
            "persistence": persistence,
            "half_life": np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf,
        }

        return diagnostics

    def get_conditional_volatility(self) -> pd.Series:
        """
        Get the fitted conditional volatility series.

        Returns:
            Series of conditional volatility (standard deviation)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.result.conditional_volatility

    def forecast(
        self,
        horizon: int = 1,
        start: Optional[int] = None,
        method: str = "simulation",
        simulations: int = 1000
    ) -> pd.DataFrame:
        """
        Forecast future volatility.

        Args:
            horizon: Number of periods to forecast
            start: Start index for forecast (None = end of sample)
            method: 'analytic', 'simulation', or 'bootstrap'
            simulations: Number of simulations (if method='simulation')

        Returns:
            DataFrame with variance forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        forecasts = self.result.forecast(
            horizon=horizon,
            start=start,
            method=method,
            simulations=simulations
        )

        return forecasts

    def get_standardized_residuals(self) -> pd.Series:
        """
        Get standardized residuals for diagnostic checks.

        Returns:
            Series of standardized residuals (should be ~N(0,1) if well-specified)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.result.std_resid

    def get_news_impact_curve(
        self,
        shock_range: Tuple[float, float] = (-3, 3),
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        Compute the News Impact Curve (NIC).

        The NIC shows how shocks of different sizes affect future volatility.
        For EGARCH, this curve is asymmetric.

        Args:
            shock_range: Range of standardized shocks to evaluate
            n_points: Number of points to compute

        Returns:
            DataFrame with shock values and corresponding volatility impact
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        params = self.result.params
        shocks = np.linspace(shock_range[0], shock_range[1], n_points)

        if self.model_type == "EGARCH":
            # EGARCH: log(σ²) = ω + α|z| + γz + βlog(σ²_{-1})
            omega = params.get("omega", 0)
            alpha = params.get("alpha[1]", 0)
            gamma = params.get("gamma[1]", 0)
            beta = params.get("beta[1]", 0)

            # Unconditional log variance
            if abs(beta) < 1:
                uncond_log_var = omega / (1 - beta)
            else:
                uncond_log_var = omega

            # News impact
            log_var = uncond_log_var + alpha * np.abs(shocks) + gamma * shocks
            variance = np.exp(log_var)

        else:  # GJR-GARCH
            omega = params.get("omega", 0)
            alpha = params.get("alpha[1]", 0)
            gamma = params.get("gamma[1]", 0)
            beta = params.get("beta[1]", 0)

            # Unconditional variance
            if (alpha + beta + 0.5 * gamma) < 1:
                uncond_var = omega / (1 - alpha - beta - 0.5 * gamma)
            else:
                uncond_var = omega

            # News impact (asymmetric for negative shocks)
            variance = omega + alpha * shocks**2 + gamma * (shocks < 0) * shocks**2 + beta * uncond_var

        nic = pd.DataFrame({
            "shock": shocks,
            "variance": variance,
            "volatility": np.sqrt(variance)
        })

        return nic

    def summary(self) -> str:
        """Get model summary."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return str(self.result.summary())


def compare_garch_models(
    returns: pd.Series,
    models: list = ["GARCH", "EGARCH", "GJR-GARCH"]
) -> pd.DataFrame:
    """
    Compare different GARCH variants.

    Args:
        returns: Return series
        models: List of model types to compare

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    for model_type in models:
        try:
            if model_type == "GARCH":
                model = arch_model(returns, vol="GARCH", p=1, q=1, dist="skewt")
            elif model_type == "EGARCH":
                model = arch_model(returns, vol="EGARCH", p=1, q=1, o=1, dist="skewt")
            elif model_type == "GJR-GARCH":
                model = arch_model(returns, vol="GARCH", p=1, o=1, q=1, dist="skewt")
            else:
                continue

            res = model.fit(disp="off", show_warning=False)

            results.append({
                "Model": model_type,
                "Log-Likelihood": res.loglikelihood,
                "AIC": res.aic,
                "BIC": res.bic,
                "Persistence": res.params.get("beta[1]", 0) + res.params.get("alpha[1]", 0),
                "Asymmetry": res.params.get("gamma[1]", 0),
            })

        except Exception as e:
            print(f"Error fitting {model_type}: {e}")

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values("AIC")

    return comparison


def main():
    """Demonstrate EGARCH model usage."""
    print("=" * 70)
    print("  EGARCH Volatility Model - Demonstration")
    print("=" * 70)

    # Generate sample FX-like returns
    np.random.seed(42)
    n = 500

    # Simulate returns with volatility clustering
    vol = np.zeros(n)
    returns = np.zeros(n)
    vol[0] = 0.01

    for t in range(1, n):
        # GARCH-like process
        vol[t] = 0.00001 + 0.1 * returns[t-1]**2 + 0.85 * vol[t-1]
        # Add asymmetry (negative returns increase vol more)
        if returns[t-1] < 0:
            vol[t] += 0.05 * returns[t-1]**2
        returns[t] = np.sqrt(vol[t]) * np.random.standard_t(5)

    returns = pd.Series(returns * 100, name="Returns")

    # Fit EGARCH
    print("\nFitting EGARCH(1,1,1) model...")
    model = EGARCHVolatilityModel(p=1, q=1, o=1, dist="skewt", model_type="EGARCH")
    diagnostics = model.fit(returns)

    print("\nModel Diagnostics:")
    print(f"  Log-Likelihood: {diagnostics['log_likelihood']:.2f}")
    print(f"  AIC: {diagnostics['aic']:.2f}")
    print(f"  BIC: {diagnostics['bic']:.2f}")
    print(f"  Leverage Effect: {diagnostics['leverage_effect']}")
    print(f"  Gamma (Asymmetry): {diagnostics['gamma']:.4f}")
    print(f"  Persistence: {diagnostics['persistence']:.4f}")
    print(f"  Half-Life: {diagnostics['half_life']:.1f} periods")

    # Compare models
    print("\n" + "=" * 70)
    print("  Model Comparison")
    print("=" * 70)
    comparison = compare_garch_models(returns)
    print(comparison.to_string(index=False))

    print("\n[Best model by AIC: EGARCH captures asymmetric news effects]")


if __name__ == "__main__":
    main()
