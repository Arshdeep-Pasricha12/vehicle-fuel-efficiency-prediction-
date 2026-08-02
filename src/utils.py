"""
utils.py — Shared helper functions for the Vehicle Fuel Efficiency project
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def load_raw(path: str = 'data/auto-mpg.csv') -> pd.DataFrame:
    """Load the raw Auto-MPG CSV, treating '?' as NaN."""
    df = pd.read_csv(path, na_values='?')
    return df
def load_cleaned(path: str = 'data/auto-mpg-cleaned.csv') -> pd.DataFrame:
    return pd.read_csv(path)
def load_features(path: str = 'data/auto-mpg-features.csv') -> pd.DataFrame:
    return pd.read_csv(path)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to a DataFrame."""
    df = df.copy()
    df['power_to_weight']      = df['horsepower'] / df['weight']
    df['displacement_per_cyl'] = df['displacement'] / df['cylinders']
    df['log_weight']           = np.log(df['weight'])
    df['log_displacement']     = np.log(df['displacement'])
    df['log_horsepower']       = np.log(df['horsepower'])
    df['year_actual']          = df['model_year'] + 1900
    df['weight_class_enc'] = pd.cut(
        df['weight'],
        bins=[0, 2500, 3500, 4500, 10_000],
        labels=[0, 1, 2, 3]
    ).astype(float)
    df['era_enc'] = pd.cut(
        df['model_year'],
        bins=[69, 73, 78, 83],
        labels=[0, 1, 2]
    ).astype(float)
    return df
def regression_metrics(y_true, y_pred, label: str = '') -> dict:
    """Return dict of RMSE, MAE, R² for a prediction set."""
    metrics = {
        'label': label,
        'R2':    round(r2_score(y_true, y_pred), 4),
        'RMSE':  round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'MAE':   round(mean_absolute_error(y_true, y_pred), 4),
    }
    return metrics
def print_metrics(y_true, y_pred, label: str = 'Model'):
    m = regression_metrics(y_true, y_pred, label)
    print(f'[{m["label"]}]  R²={m["R2"]:.4f} | RMSE={m["RMSE"]:.4f} | MAE={m["MAE"]:.4f}')
def plot_actual_vs_predicted(y_true, y_pred, title: str = 'Actual vs Predicted',
                              save_path: str = None):
    """Scatter plot of actual vs predicted values with perfect prediction line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=40, color='#3498db',
               edgecolors='white', linewidths=0.4)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect')
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_title(f'{title}\nR²={r2:.4f} | RMSE={rmse:.4f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_feature_importance(model, feature_names: list, top_n: int = 15,
                             title: str = 'Feature Importance', save_path: str = None):
    """Horizontal bar chart of feature importances."""
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(fi.index, fi.values, color='#3498db', edgecolor='white')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
