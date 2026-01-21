"""
Visualization Utilities for EGARCH + XGBoost Hybrid Model

Includes:
- Volatility time series plots
- News Impact Curves (asymmetry visualization)
- Prediction vs Actual comparisons
- Feature importance charts
- Residual diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional, Tuple
import os

from config import OUTPUT_DIR


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_volatility_series(
    conditional_vol: pd.Series,
    realized_vol: Optional[pd.Series] = None,
    returns: Optional[pd.Series] = None,
    title: str = "EGARCH Conditional Volatility",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot conditional volatility from EGARCH model.

    Args:
        conditional_vol: EGARCH conditional volatility series
        realized_vol: Optional realized volatility for comparison
        returns: Optional returns series
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    n_plots = 1 + (realized_vol is not None) + (returns is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    # Plot 1: Conditional volatility
    ax = axes[0]
    ax.plot(conditional_vol.index, conditional_vol.values, color='blue', linewidth=1, label='EGARCH Vol')
    if realized_vol is not None:
        ax.plot(realized_vol.index, realized_vol.values, color='orange', alpha=0.5, label='Realized Vol')
    ax.set_ylabel('Volatility')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plot_idx = 1

    # Plot 2: Returns if provided
    if returns is not None:
        ax = axes[plot_idx]
        ax.plot(returns.index, returns.values, color='gray', linewidth=0.5)
        ax.fill_between(returns.index, returns.values, 0, alpha=0.3,
                        where=returns.values > 0, color='green', label='Positive')
        ax.fill_between(returns.index, returns.values, 0, alpha=0.3,
                        where=returns.values < 0, color='red', label='Negative')
        ax.set_ylabel('Returns (%)')
        ax.set_title('Returns with Volatility Clustering')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_news_impact_curve(
    nic_data: pd.DataFrame,
    title: str = "News Impact Curve (Asymmetric Response)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the News Impact Curve showing asymmetric volatility response.

    The NIC shows how positive vs negative shocks affect volatility differently.
    A steeper left side indicates "leverage effect" - bad news hurts more.

    Args:
        nic_data: DataFrame with 'shock' and 'volatility' columns
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the curve
    ax.plot(nic_data['shock'], nic_data['volatility'], color='blue', linewidth=2)

    # Shade regions
    neg_mask = nic_data['shock'] < 0
    pos_mask = nic_data['shock'] > 0

    ax.fill_between(nic_data['shock'][neg_mask], 0, nic_data['volatility'][neg_mask],
                    alpha=0.3, color='red', label='Bad News (Panic)')
    ax.fill_between(nic_data['shock'][pos_mask], 0, nic_data['volatility'][pos_mask],
                    alpha=0.3, color='green', label='Good News (Relief)')

    # Add zero line
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # Annotations
    min_shock_vol = nic_data[nic_data['shock'] == nic_data['shock'].min()]['volatility'].values[0]
    max_shock_vol = nic_data[nic_data['shock'] == nic_data['shock'].max()]['volatility'].values[0]
    asymmetry = min_shock_vol / max_shock_vol if max_shock_vol > 0 else 1

    ax.annotate(f'Asymmetry: {asymmetry:.2f}x\n(Bad/Good response)',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Standardized Shock (z-score)', fontsize=12)
    ax.set_ylabel('Conditional Volatility', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_predictions(
    actual: pd.Series,
    predicted: pd.Series,
    title: str = "Hybrid Model: EGARCH + XGBoost Predictions",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot actual vs predicted values.

    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Time series comparison
    ax1 = axes[0]
    ax1.plot(actual.index, actual.values, color='black', linewidth=1.5, label='Actual')
    ax1.plot(predicted.index, predicted.values, color='red', linewidth=1, linestyle='--', label='Predicted')
    ax1.set_ylabel('Value')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals
    ax2 = axes[1]
    residuals = actual.values - predicted.values
    ax2.bar(actual.index, residuals, color=['green' if r > 0 else 'red' for r in residuals], alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Residual (Actual - Predicted)')
    ax2.set_title('Prediction Residuals')
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance (XGBoost Hybrid)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance from XGBoost.

    Args:
        importance_df: DataFrame with 'Feature' and 'Importance' columns
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    # Get top N features
    top_features = importance_df.head(top_n).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

    # Color coding: GARCH features in blue, others in green
    colors = ['blue' if 'GARCH' in f or 'Vol' in f else 'green'
              for f in top_features['Feature']]

    ax.barh(top_features['Feature'], top_features['Importance'], color=colors, alpha=0.7)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Volatility Features'),
        Patch(facecolor='green', alpha=0.7, label='News/Other Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_asymmetry_analysis(
    df: pd.DataFrame,
    vol_col: str = "GARCH_Vol",
    panic_col: str = "Is_Panic",
    relief_col: str = "Is_Relief",
    title: str = "Asymmetric Volatility Response to News",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize asymmetric volatility response to panic vs relief.

    Args:
        df: DataFrame with volatility and panic indicators
        vol_col: Volatility column name
        panic_col: Panic indicator column
        relief_col: Relief indicator column
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Box plot comparison
    ax1 = axes[0]
    panic_vol = df[df[panic_col] == 1][vol_col].dropna()
    relief_vol = df[df[relief_col] == 1][vol_col].dropna()
    normal_vol = df[(df[panic_col] == 0) & (df[relief_col] == 0)][vol_col].dropna()

    box_data = [panic_vol, normal_vol, relief_vol]
    bp = ax1.boxplot(box_data, labels=['Panic Days', 'Normal Days', 'Relief Days'],
                     patch_artist=True)

    colors = ['red', 'gray', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Conditional Volatility')
    ax1.set_title('Volatility Distribution by News Type')
    ax1.grid(True, alpha=0.3)

    # Add mean annotations
    means = [panic_vol.mean(), normal_vol.mean(), relief_vol.mean()]
    for i, mean in enumerate(means):
        ax1.annotate(f'μ={mean:.4f}', xy=(i+1, mean), xytext=(i+1.2, mean),
                     fontsize=9, color='black')

    # Plot 2: Time series with panic/relief markers
    ax2 = axes[1]
    ax2.plot(df.index, df[vol_col], color='blue', linewidth=0.8, alpha=0.7, label='Volatility')

    # Mark panic days
    panic_dates = df[df[panic_col] == 1].index
    if len(panic_dates) > 0:
        ax2.scatter(panic_dates, df.loc[panic_dates, vol_col],
                    color='red', s=30, alpha=0.6, label='Panic', zorder=5)

    # Mark relief days
    relief_dates = df[df[relief_col] == 1].index
    if len(relief_dates) > 0:
        ax2.scatter(relief_dates, df.loc[relief_dates, vol_col],
                    color='green', s=30, alpha=0.6, label='Relief', zorder=5)

    ax2.set_ylabel('Conditional Volatility')
    ax2.set_title('Volatility with Panic/Relief Events')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_model_diagnostics(
    residuals: pd.Series,
    title: str = "Model Residual Diagnostics",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot residual diagnostics for model validation.

    Args:
        residuals: Model residuals or standardized residuals
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Residual time series
    ax1 = axes[0, 0]
    ax1.plot(residuals.index, residuals.values, color='blue', linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.axhline(y=2, color='orange', linestyle=':', linewidth=1)
    ax1.axhline(y=-2, color='orange', linestyle=':', linewidth=1)
    ax1.set_title('Standardized Residuals Over Time')
    ax1.set_ylabel('Residual')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram
    ax2 = axes[0, 1]
    ax2.hist(residuals.values, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    # Overlay normal distribution
    x = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy.stats import norm
    ax2.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Q-Q plot
    ax3 = axes[1, 0]
    from scipy.stats import probplot
    probplot(residuals.dropna().values, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: ACF of squared residuals
    ax4 = axes[1, 1]
    squared_resid = residuals.dropna() ** 2
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(squared_resid, ax=ax4, lags=20, alpha=0.05)
    ax4.set_title('ACF of Squared Residuals\n(Should be insignificant if vol model is adequate)')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_summary_dashboard(
    results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive dashboard summarizing all results.

    Args:
        results: Dictionary from HybridPipeline.run_full_pipeline()
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12))

    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Predictions
    ax1 = fig.add_subplot(gs[0, :2])
    actual = results['test_actual']
    pred = results['test_predictions']
    ax1.plot(actual.index, actual.values, 'k-', label='Actual', linewidth=1.5)
    ax1.plot(pred.index, pred.values, 'r--', label='Predicted', linewidth=1)
    ax1.set_title('Hybrid Model Predictions vs Actual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Feature Importance
    ax2 = fig.add_subplot(gs[0, 2])
    top_feat = results['feature_importance'].head(8).sort_values('Importance', ascending=True)
    colors = ['blue' if 'GARCH' in f or 'Vol' in f else 'green' for f in top_feat['Feature']]
    ax2.barh(top_feat['Feature'], top_feat['Importance'], color=colors, alpha=0.7)
    ax2.set_title('Top Features')
    ax2.set_xlabel('Importance')

    # 3. Metrics text box
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    metrics = results['metrics']
    garch = results['garch_diagnostics']

    metrics_text = f"""
    EGARCH DIAGNOSTICS
    ─────────────────
    Model: {garch['model_type']}
    Leverage Effect: {garch['leverage_effect']}
    Gamma: {garch['gamma']:.4f}
    Persistence: {garch['persistence']:.4f}
    AIC: {garch['aic']:.1f}

    XGBOOST METRICS
    ─────────────────
    Test RMSE: {metrics['test_rmse']:.4f}
    Test MAE: {metrics['test_mae']:.4f}
    Test R²: {metrics['test_r2']:.4f}
    Train Samples: {metrics['n_train']}
    Test Samples: {metrics['n_test']}
    """

    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Residual histogram
    ax4 = fig.add_subplot(gs[1, 1])
    residuals = actual.values - pred.values
    ax4.hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('Prediction Residuals')
    ax4.set_xlabel('Error')

    # 5. Asymmetry analysis
    ax5 = fig.add_subplot(gs[1, 2])
    asym = results['asymmetry_analysis']
    if 'avg_vol_panic' in asym:
        categories = ['Panic', 'Normal', 'Relief']
        values = [asym.get('avg_vol_panic', 0), asym.get('avg_vol_normal', 0), asym.get('avg_vol_relief', 0)]
        colors = ['red', 'gray', 'green']
        ax5.bar(categories, values, color=colors, alpha=0.7)
        ax5.set_title('Avg Volatility by News Type')
        ax5.set_ylabel('Conditional Volatility')
    else:
        ax5.text(0.5, 0.5, 'Asymmetry analysis\nnot available', ha='center', va='center')
        ax5.set_title('Asymmetry Analysis')

    plt.suptitle('EGARCH + XGBoost Hybrid Model Dashboard', fontsize=16, y=0.98)

    if save_path:
        ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded. Import and use individual functions.")
