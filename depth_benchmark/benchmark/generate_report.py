"""
Report generation and visualization utilities.
"""

import os
import logging
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive benchmark reports."""
    
    def __init__(self, output_dir: str):
        """Initialize report generator."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_all_reports(self, results: Dict):
        """Generate all report types."""
        self.generate_metrics_plots(results)
        self.generate_ranking_table(results)
        self.generate_summary(results)
    
    def generate_metrics_plots(self, results: Dict):
        """Generate metric comparison plots."""
        logger.info("Generating metrics plots...")
        
        models = list(results.keys())
        
        # Extract metrics
        mae_values = []
        rmse_values = []
        abs_rel_values = []
        silog_values = []
        
        for model_name in models:
            model_results = results[model_name]
            metrics = model_results.get('depth_metrics', {})
            
            mae = metrics.get('mae', {})
            mae_values.append(mae.get('mean', 0) if isinstance(mae, dict) else mae)
            
            rmse = metrics.get('rmse', {})
            rmse_values.append(rmse.get('mean', 0) if isinstance(rmse, dict) else rmse)
            
            abs_rel = metrics.get('abs_rel', {})
            abs_rel_values.append(abs_rel.get('mean', 0) if isinstance(abs_rel, dict) else abs_rel)
            
            silog = metrics.get('silog', {})
            silog_values.append(silog.get('mean', 0) if isinstance(silog, dict) else silog)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_data = [
            (mae_values, 'MAE (Lower is Better)'),
            (rmse_values, 'RMSE (Lower is Better)'),
            (abs_rel_values, 'Absolute Relative Error (Lower is Better)'),
            (silog_values, 'SILog Error (Lower is Better)')
        ]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        for ax, (values, title) in zip(axes.flat, metrics_data):
            bars = ax.bar(models, values, color=colors)
            ax.set_ylabel('Error Value')
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics plot saved to {plot_path}")
    
    def generate_ranking_table(self, results: Dict):
        """Generate model ranking table."""
        logger.info("Generating ranking table...")
        
        # Calculate scores
        rankings = {}
        
        for model_name, model_results in results.items():
            metrics = model_results.get('depth_metrics', {})
            
            score = 0
            weights = {
                'mae': 1.0,
                'rmse': 1.0,
                'abs_rel': 0.5,
                'silog': 0.5,
                'delta_1.2500': 0.3,
                'delta_1.5625': 0.2,
                'delta_1.9531': 0.1,
            }
            
            for metric_name, weight in weights.items():
                if metric_name in metrics:
                    metric_val = metrics[metric_name]
                    if isinstance(metric_val, dict):
                        value = metric_val.get('mean', 0)
                    else:
                        value = metric_val
                    
                    # Normalize (lower is better for most metrics)
                    score += (1.0 - value) * weight
            
            rankings[model_name] = score
        
        # Sort
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        # Write table
        table_path = os.path.join(self.output_dir, 'model_rankings.txt')
        
        with open(table_path, 'w') as f:
            f.write("MODEL RANKING\n")
            f.write("=" * 50 + "\n\n")
            f.write("Rank | Model                    | Score\n")
            f.write("-" * 50 + "\n")
            
            for rank, (model_name, score) in enumerate(sorted_rankings, 1):
                f.write(f"{rank:4d} | {model_name:24s} | {score:.4f}\n")
        
        logger.info(f"Ranking table saved to {table_path}")
    
    def generate_summary(self, results: Dict):
        """Generate summary statistics."""
        logger.info("Generating summary...")
        
        summary_path = os.path.join(self.output_dir, 'summary.json')
        
        summary = {
            'num_models': len(results),
            'models': list(results.keys()),
            'best_models': {},
            'average_metrics': {}
        }
        
        # Find best models
        metrics_list = ['mae', 'rmse', 'abs_rel', 'silog']
        
        for metric_name in metrics_list:
            best_model = None
            best_value = float('inf')
            
            for model_name, model_results in results.items():
                metrics = model_results.get('depth_metrics', {})
                if metric_name in metrics:
                    metric_val = metrics[metric_name]
                    if isinstance(metric_val, dict):
                        value = metric_val.get('mean', float('inf'))
                    else:
                        value = metric_val
                    
                    if value < best_value:
                        best_value = value
                        best_model = model_name
            
            if best_model:
                summary['best_models'][metric_name] = {
                    'model': best_model,
                    'value': best_value
                }
        
        # Calculate averages
        for metric_name in metrics_list:
            values = []
            for model_results in results.values():
                metrics = model_results.get('depth_metrics', {})
                if metric_name in metrics:
                    metric_val = metrics[metric_name]
                    if isinstance(metric_val, dict):
                        value = metric_val.get('mean')
                    else:
                        value = metric_val
                    if value is not None:
                        values.append(value)
            
            if values:
                summary['average_metrics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
