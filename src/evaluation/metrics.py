"""
Evaluation metrics for disease severity prediction.

This module provides comprehensive metrics for regression tasks including
R², RMSE, MAE, and per-fold aggregation utilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import pandas as pd


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Handle edge case where variance is 0
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else float('-inf')
    
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    return float(np.mean((y_true - y_pred) ** 2))


def calculate_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient and p-value.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    r, p = stats.pearsonr(y_true, y_pred)
    return float(r), float(p)


def calculate_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient and p-value.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    rho, p = stats.spearmanr(y_true, y_pred)
    return float(rho), float(p)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value as percentage
    """
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return float(mape)


def calculate_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate explained variance score.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Explained variance score
    """
    var_y = np.var(y_true)
    var_residual = np.var(y_true - y_pred)
    
    if var_y == 0:
        return 0.0 if var_residual == 0 else float('-inf')
    
    return float(1 - (var_residual / var_y))


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate all regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., 'val_', 'test_')
        
    Returns:
        Dictionary with all metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Calculate all metrics
    metrics = {
        f'{prefix}r2': calculate_r2(y_true, y_pred),
        f'{prefix}rmse': calculate_rmse(y_true, y_pred),
        f'{prefix}mae': calculate_mae(y_true, y_pred),
        f'{prefix}mse': calculate_mse(y_true, y_pred),
        f'{prefix}explained_variance': calculate_explained_variance(y_true, y_pred),
    }
    
    # Add correlations
    pearson_r, pearson_p = calculate_pearson_correlation(y_true, y_pred)
    spearman_rho, spearman_p = calculate_spearman_correlation(y_true, y_pred)
    
    metrics.update({
        f'{prefix}pearson_r': pearson_r,
        f'{prefix}pearson_p': pearson_p,
        f'{prefix}spearman_rho': spearman_rho,
        f'{prefix}spearman_p': spearman_p,
    })
    
    # Add MAPE if no zeros in ground truth
    if not np.any(y_true == 0):
        metrics[f'{prefix}mape'] = calculate_mape(y_true, y_pred)
    
    # Add basic statistics
    metrics.update({
        f'{prefix}n_samples': len(y_true),
        f'{prefix}mean_true': float(np.mean(y_true)),
        f'{prefix}std_true': float(np.std(y_true)),
        f'{prefix}mean_pred': float(np.mean(y_pred)),
        f'{prefix}std_pred': float(np.std(y_pred)),
        f'{prefix}bias': float(np.mean(y_pred - y_true)),
    })
    
    return metrics


def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, float]],
    metric_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across cross-validation folds.
    
    Args:
        fold_metrics: List of metric dictionaries, one per fold
        metric_names: Optional list of metric names to aggregate.
                     If None, aggregates all numeric metrics.
        
    Returns:
        Dictionary with aggregated statistics:
        {
            'metric_name': {
                'mean': float,
                'std': float,
                'min': float,
                'max': float,
                'median': float,
                'values': List[float]
            }
        }
    """
    if not fold_metrics:
        return {}
    
    # Get metric names if not provided
    if metric_names is None:
        # Get all numeric metrics from first fold
        metric_names = [
            key for key, value in fold_metrics[0].items()
            if isinstance(value, (int, float)) and not np.isnan(value)
        ]
    
    aggregated = {}
    
    for metric_name in metric_names:
        # Collect values across folds
        values = []
        for fold in fold_metrics:
            if metric_name in fold:
                val = fold[metric_name]
                if not np.isnan(val) and not np.isinf(val):
                    values.append(val)
        
        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'values': values,
                'n_folds': len(values)
            }
    
    return aggregated


def format_metrics_table(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics"
) -> str:
    """
    Format metrics as a readable table string.
    
    Args:
        metrics: Dictionary of metrics
        title: Table title
        
    Returns:
        Formatted string
    """
    # Group metrics by category
    main_metrics = ['r2', 'rmse', 'mae', 'mse']
    correlation_metrics = ['pearson_r', 'spearman_rho']
    stat_metrics = ['mean_true', 'std_true', 'mean_pred', 'std_pred', 'bias']
    
    output = [
        "",
        "=" * 70,
        f"{title:^70}",
        "=" * 70,
        ""
    ]
    
    # Main metrics
    output.append("Primary Metrics:")
    output.append("-" * 70)
    for metric in main_metrics:
        # Try to find metric with any prefix
        for key in metrics:
            if key.endswith(metric):
                output.append(f"  {key:30s}: {metrics[key]:>10.4f}")
                break
    
    # Correlation metrics
    output.append("\nCorrelation Metrics:")
    output.append("-" * 70)
    for metric in correlation_metrics:
        for key in metrics:
            if metric in key:
                output.append(f"  {key:30s}: {metrics[key]:>10.4f}")
                
    # Statistics
    output.append("\nData Statistics:")
    output.append("-" * 70)
    for metric in stat_metrics:
        for key in metrics:
            if key.endswith(metric):
                output.append(f"  {key:30s}: {metrics[key]:>10.4f}")
    
    # Sample size
    for key in metrics:
        if key.endswith('n_samples'):
            output.append(f"  {key:30s}: {metrics[key]:>10.0f}")
    
    output.append("=" * 70)
    output.append("")
    
    return "\n".join(output)


def format_aggregated_metrics(
    aggregated: Dict[str, Dict[str, float]],
    title: str = "Cross-Validation Results"
) -> str:
    """
    Format aggregated metrics as a readable table.
    
    Args:
        aggregated: Dictionary of aggregated metrics
        title: Table title
        
    Returns:
        Formatted string
    """
    output = [
        "",
        "=" * 80,
        f"{title:^80}",
        "=" * 80,
        "",
        f"{'Metric':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}",
        "-" * 80
    ]
    
    # Sort metrics by name
    for metric_name in sorted(aggregated.keys()):
        stats = aggregated[metric_name]
        output.append(
            f"{metric_name:<30} "
            f"{stats['mean']:>11.4f} "
            f"{stats['std']:>11.4f} "
            f"{stats['min']:>11.4f} "
            f"{stats['max']:>11.4f}"
        )
    
    output.append("=" * 80)
    output.append("")
    
    return "\n".join(output)


def save_metrics_to_csv(
    metrics: Dict[str, float],
    filepath: str,
    append: bool = False
) -> None:
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save CSV
        append: If True, append to existing file
    """
    df = pd.DataFrame([metrics])
    
    if append:
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)


def save_aggregated_metrics_to_csv(
    aggregated: Dict[str, Dict[str, float]],
    filepath: str
) -> None:
    """
    Save aggregated metrics to CSV file.
    
    Args:
        aggregated: Dictionary of aggregated metrics
        filepath: Path to save CSV
    """
    # Flatten the nested dictionary
    rows = []
    for metric_name, stats in aggregated.items():
        row = {'metric': metric_name}
        row.update(stats)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)


# Example usage
if __name__ == "__main__":
    print("Metrics module loaded.")
    print("\nExample usage:")
    print("""
    from src.evaluation.metrics import calculate_all_metrics
    
    # Calculate metrics
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    metrics = calculate_all_metrics(y_true, y_pred, prefix='test_')
    print(format_metrics_table(metrics))
    """)
