"""
Model evaluator for test set evaluation.

This module provides the Evaluator class for loading trained models and
evaluating them on test datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import (
    calculate_all_metrics,
    format_metrics_table,
    save_metrics_to_csv
)


logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class for model evaluation on test sets.
    
    Handles:
    - Loading trained models from checkpoints
    - Running inference on test data
    - Computing comprehensive metrics
    - Saving results and visualizations
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model (architecture only, weights loaded from checkpoint)
            device: Device to run evaluation on
            checkpoint_path: Path to model checkpoint
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir) if output_dir else Path('results')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Log checkpoint info
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
        if 'val_r2' in checkpoint:
            logger.info(f"Validation R²: {checkpoint['val_r2']:.4f}")
        if 'val_loss' in checkpoint:
            logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint
    
    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            Tuple of (predictions, ground truth labels)
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        logger.info("Generating predictions...")
        for batch in tqdm(dataloader, desc="Predicting"):
            # Handle different batch formats
            if len(batch) == 2:
                images, targets = batch
            elif len(batch) == 3:
                images, targets, _ = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Handle model outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
            
            # Collect results
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions, targets
    
    def evaluate(
        self,
        dataloader: DataLoader,
        split_name: str = 'test',
        save_predictions: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            split_name: Name of the split ('test', 'val', etc.)
            save_predictions: Whether to save predictions to file
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {split_name} set...")
        
        # Generate predictions
        predictions, targets = self.predict(dataloader)
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            targets,
            predictions,
            prefix=f'{split_name}_'
        )
        
        # Print metrics
        print(format_metrics_table(metrics, title=f"{split_name.capitalize()} Set Evaluation"))
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(predictions, targets, split_name)
        
        # Save metrics
        self._save_metrics(metrics, split_name)
        
        # Create visualizations
        self._create_visualizations(predictions, targets, split_name)
        
        return metrics
    
    def _save_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        split_name: str
    ) -> None:
        """Save predictions to file."""
        import pandas as pd
        
        output_file = self.output_dir / f'{split_name}_predictions.csv'
        
        df = pd.DataFrame({
            'target': targets,
            'prediction': predictions,
            'error': predictions - targets,
            'abs_error': np.abs(predictions - targets),
        })
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")
    
    def _save_metrics(
        self,
        metrics: Dict[str, float],
        split_name: str
    ) -> None:
        """Save metrics to JSON and CSV."""
        # JSON
        json_file = self.output_dir / f'{split_name}_metrics.json'
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {json_file}")
        
        # CSV
        csv_file = self.output_dir / f'{split_name}_metrics.csv'
        save_metrics_to_csv(metrics, str(csv_file))
    
    def _create_visualizations(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        split_name: str
    ) -> None:
        """Create and save visualization plots."""
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot: Predicted vs True
        ax = axes[0, 0]
        ax.scatter(targets, predictions, alpha=0.5, s=20)
        
        # Add identity line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # Add regression line
        z = np.polyfit(targets, predictions, 1)
        p = np.poly1d(z)
        ax.plot(targets, p(targets), 'b-', lw=2, alpha=0.8, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_xlabel('True Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Predicted vs True Values', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² to plot
        from .metrics import calculate_r2
        r2 = calculate_r2(targets, predictions)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residual plot
        ax = axes[0, 1]
        residuals = predictions - targets
        ax.scatter(predictions, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax = axes[1, 0]
        errors = predictions - targets
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.axvline(x=np.mean(errors), color='g', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(errors):.3f}')
        ax.set_xlabel('Prediction Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Residuals)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        output_file = self.output_dir / f'{split_name}_evaluation_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_file}")
    
    def compare_with_baseline(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        baseline_predictions: np.ndarray,
        split_name: str = 'test'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare current model with baseline predictions.
        
        Args:
            predictions: Current model predictions
            targets: Ground truth
            baseline_predictions: Baseline model predictions
            split_name: Name of split
            
        Returns:
            Dictionary with comparison metrics
        """
        # Calculate metrics for both
        current_metrics = calculate_all_metrics(targets, predictions, prefix='current_')
        baseline_metrics = calculate_all_metrics(targets, baseline_predictions, prefix='baseline_')
        
        # Combine and calculate improvements
        comparison = {
            'current': current_metrics,
            'baseline': baseline_metrics,
            'improvement': {}
        }
        
        # Calculate improvement for key metrics
        for metric in ['r2', 'rmse', 'mae']:
            current_val = current_metrics.get(f'current_{metric}', 0)
            baseline_val = baseline_metrics.get(f'baseline_{metric}', 0)
            
            if metric == 'r2':
                # Higher is better
                improvement = current_val - baseline_val
            else:
                # Lower is better (RMSE, MAE)
                improvement = baseline_val - current_val
            
            comparison['improvement'][metric] = improvement
        
        # Print comparison
        print("\n" + "="*70)
        print(f"{'Model Comparison':^70}")
        print("="*70)
        print(f"{'Metric':<20} {'Current':<15} {'Baseline':<15} {'Improvement':<15}")
        print("-"*70)
        
        for metric in ['r2', 'rmse', 'mae']:
            current_val = current_metrics.get(f'current_{metric}', 0)
            baseline_val = baseline_metrics.get(f'baseline_{metric}', 0)
            improvement = comparison['improvement'][metric]
            
            print(f"{metric.upper():<20} {current_val:<15.4f} {baseline_val:<15.4f} {improvement:+15.4f}")
        
        print("="*70 + "\n")
        
        # Save comparison
        comparison_file = self.output_dir / f'{split_name}_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison


# Example usage
if __name__ == "__main__":
    print("Evaluator module loaded.")
    print("\nExample usage:")
    print("""
    from src.models import create_efficientnetv2_s
    from src.evaluation import Evaluator
    import torch
    
    # Create model
    model = create_efficientnetv2_s(pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        checkpoint_path='checkpoints/best_model.pt',
        output_dir='results/evaluation'
    )
    
    # Evaluate on test set
    metrics = evaluator.evaluate(test_loader, split_name='test')
    """)
