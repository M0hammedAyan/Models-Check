"""
Model comparison utilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare multiple depth estimation models."""
    
    def __init__(self):
        """Initialize comparator."""
        self.comparison_results = {}
    
    def add_model_metrics(self, model_name: str, metrics: Dict):
        """
        Add metrics for a model.
        
        Args:
            model_name: Name of the model
            metrics: Metrics dictionary
        """
        self.comparison_results[model_name] = metrics
        logger.info(f"Added metrics for {model_name}")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table from all models.
        
        Returns:
            Pandas DataFrame with comparison
        """
        rows = []
        
        for model_name, metrics in self.comparison_results.items():
            row = {'Model': model_name}
            
            # Extract depth metrics
            if 'depth_metrics' in metrics:
                for metric_name, metric_values in metrics['depth_metrics'].items():
                    if isinstance(metric_values, dict):
                        row[f"{metric_name}_mean"] = metric_values.get('mean', np.nan)
                        row[f"{metric_name}_std"] = metric_values.get('std', np.nan)
                    else:
                        row[metric_name] = metric_values
            
            # Extract temporal metrics
            if 'temporal_metrics' in metrics:
                for metric_name, metric_value in metrics['temporal_metrics'].items():
                    row[f"temporal_{metric_name}"] = metric_value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def get_best_model(self, metric_name: str, higher_is_better: bool = False) -> str:
        """
        Get best model for a specific metric.
        
        Args:
            metric_name: Name of metric to evaluate
            higher_is_better: Whether higher values are better
            
        Returns:
            Name of best model
        """
        values = {}
        
        for model_name, metrics in self.comparison_results.items():
            if 'depth_metrics' in metrics and metric_name in metrics['depth_metrics']:
                metric_val = metrics['depth_metrics'][metric_name]
                if isinstance(metric_val, dict):
                    values[model_name] = metric_val.get('mean', np.nan)
                else:
                    values[model_name] = metric_val
        
        if not values:
            return None
        
        # Remove NaN values
        values = {k: v for k, v in values.items() if not np.isnan(v)}
        
        if not values:
            return None
        
        best_model = max(values.items(), key=lambda x: x[1] if higher_is_better else -x[1])
        return best_model[0]
    
    def get_ranking(self, metric_name: str, higher_is_better: bool = False) -> List[tuple]:
        """
        Get ranking of models for a specific metric.
        
        Args:
            metric_name: Name of metric
            higher_is_better: Whether higher is better
            
        Returns:
            List of (model_name, value) tuples ranked
        """
        values = {}
        
        for model_name, metrics in self.comparison_results.items():
            if 'depth_metrics' in metrics and metric_name in metrics['depth_metrics']:
                metric_val = metrics['depth_metrics'][metric_name]
                if isinstance(metric_val, dict):
                    values[model_name] = metric_val.get('mean', np.nan)
                else:
                    values[model_name] = metric_val
        
        # Remove NaN
        values = {k: v for k, v in values.items() if not np.isnan(v)}
        
        # Sort
        ranking = sorted(values.items(),
                        key=lambda x: x[1],
                        reverse=higher_is_better)
        
        return ranking
    
    def create_comparison_figure(self, output_path: str,
                                metrics_to_plot: Optional[List[str]] = None):
        """
        Create comparison figure.
        
        Args:
            output_path: Path to save figure
            metrics_to_plot: Metrics to include in plot
        """
        df = self.create_comparison_table()
        
        if metrics_to_plot is None:
            # Use common metrics
            metrics_to_plot = ['mae_mean', 'rmse_mean', 'abs_rel_mean', 'silog_mean']
        
        # Filter to available metrics
        available = [m for m in metrics_to_plot if m in df.columns]
        
        if not available:
            logger.warning("No metrics available to plot")
            return
        
        fig, axes = plt.subplots(len(available), 1, figsize=(10, 3 * len(available)))
        
        if len(available) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available):
            df_sorted = df.sort_values(metric)
            ax.barh(df_sorted['Model'], df_sorted[metric])
            ax.set_xlabel(metric)
            ax.set_title(f"Model Comparison - {metric}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison figure saved to {output_path}")


def compare_refined_vs_non_refined(results_refined: Dict,
                                   results_non_refined: Dict) -> Dict:
    """
    Compare refined vs non-refined outputs.
    
    Args:
        results_refined: Results with temporal refinement
        results_non_refined: Results without temporal refinement
        
    Returns:
        Comparison results
    """
    comparison = {}
    
    for metric in results_refined.get('depth_metrics', {}).keys():
        refined_val = results_refined['depth_metrics'][metric]
        non_refined_val = results_non_refined['depth_metrics'][metric]
        
        if isinstance(refined_val, dict) and isinstance(non_refined_val, dict):
            refined_mean = refined_val.get('mean', 0)
            non_refined_mean = non_refined_val.get('mean', 0)
            
            improvement = (non_refined_mean - refined_mean) / (non_refined_mean + 1e-8) * 100
            
            comparison[metric] = {
                'refined': refined_mean,
                'non_refined': non_refined_mean,
                'improvement_percent': improvement
            }
    
    return comparison
