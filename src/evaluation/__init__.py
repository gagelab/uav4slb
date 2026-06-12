"""
Evaluation module for model assessment.

This module provides tools for evaluating trained models including:
- Comprehensive metrics (R², RMSE, MAE, etc.)
- Test set evaluation
- Cross-validation aggregation
- Result visualization
"""

from .metrics import (
    calculate_r2,
    calculate_rmse,
    calculate_mae,
    calculate_mse,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    calculate_all_metrics,
    aggregate_fold_metrics,
    format_metrics_table,
    format_aggregated_metrics,
    save_metrics_to_csv,
    save_aggregated_metrics_to_csv,
)

from .evaluator import Evaluator

__all__ = [
    # Metrics
    'calculate_r2',
    'calculate_rmse',
    'calculate_mae',
    'calculate_mse',
    'calculate_pearson_correlation',
    'calculate_spearman_correlation',
    'calculate_all_metrics',
    'aggregate_fold_metrics',
    'format_metrics_table',
    'format_aggregated_metrics',
    'save_metrics_to_csv',
    'save_aggregated_metrics_to_csv',
    # Evaluator
    'Evaluator',
]
