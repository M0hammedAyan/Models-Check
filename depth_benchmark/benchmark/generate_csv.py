"""
CSV generation utilities for benchmark results.
"""

import csv
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CSVExporter:
    """Export benchmark results to CSV."""
    
    @staticmethod
    def export_frame_metrics(results: Dict, output_path: str):
        """
        Export frame-by-frame metrics to CSV.
        
        Args:
            results: Benchmark results
            output_path: Output CSV path
        """
        rows = []
        
        # Extract per-frame metrics
        if 'frame_metrics' in results:
            for frame_idx, metrics in enumerate(results['frame_metrics']):
                row = {'frame_index': frame_idx}
                row.update(metrics)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Frame metrics exported to {output_path}")
    
    @staticmethod
    def export_model_comparison(models_results: Dict[str, Dict], output_path: str):
        """
        Export model comparison to CSV.
        
        Args:
            models_results: Dictionary of {model_name: results}
            output_path: Output CSV path
        """
        rows = []
        
        for model_name, results in models_results.items():
            row = {'model': model_name}
            
            # Depth metrics
            if 'depth_metrics' in results:
                for metric_name, metric_val in results['depth_metrics'].items():
                    if isinstance(metric_val, dict):
                        row[f"{metric_name}_mean"] = metric_val.get('mean', np.nan)
                        row[f"{metric_name}_std"] = metric_val.get('std', np.nan)
                    else:
                        row[metric_name] = metric_val
            
            # Temporal metrics
            if 'temporal_metrics' in results:
                row.update(results['temporal_metrics'])
            
            # Performance metrics
            if 'performance' in results:
                row.update(results['performance'])
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Model comparison exported to {output_path}")
    
    @staticmethod
    def export_delta_accuracy_table(results: Dict, output_path: str):
        """
        Export delta accuracy table to CSV.
        
        Args:
            results: Results containing delta accuracy metrics
            output_path: Output CSV path
        """
        rows = []
        
        for model_name, model_results in results.items():
            row = {'model': model_name}
            
            if 'depth_metrics' in model_results:
                for metric_name, metric_val in model_results['depth_metrics'].items():
                    if 'delta_' in metric_name:
                        if isinstance(metric_val, dict):
                            row[metric_name] = metric_val.get('mean', np.nan)
                        else:
                            row[metric_name] = metric_val
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Delta accuracy table exported to {output_path}")
    
    @staticmethod
    def export_pose_metrics(pose_results: Dict, output_path: str):
        """
        Export pose estimation metrics to CSV.
        
        Args:
            pose_results: Pose estimation results
            output_path: Output CSV path
        """
        rows = []
        
        if 'joint_errors' in pose_results:
            for frame_idx, joint_errors in enumerate(pose_results['joint_errors']):
                row = {'frame_index': frame_idx}
                row.update(joint_errors)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Pose metrics exported to {output_path}")


class BenchmarkSummaryWriter:
    """Write benchmark summary reports."""
    
    @staticmethod
    def write_text_report(results: Dict, output_path: str):
        """
        Write text report of benchmark results.
        
        Args:
            results: Benchmark results
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEPTH ESTIMATION BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model results
            if 'models' in results:
                f.write("MODEL PERFORMANCE SUMMARY\n")
                f.write("-" * 80 + "\n\n")
                
                for model_name, model_results in results['models'].items():
                    f.write(f"\n{model_name}\n")
                    f.write("-" * 40 + "\n")
                    
                    if 'depth_metrics' in model_results:
                        f.write("Depth Metrics:\n")
                        for metric_name, metric_val in model_results['depth_metrics'].items():
                            if isinstance(metric_val, dict):
                                f.write(f"  {metric_name}:\n")
                                f.write(f"    mean: {metric_val.get('mean', 'N/A'):.6f}\n")
                                f.write(f"    std:  {metric_val.get('std', 'N/A'):.6f}\n")
                            else:
                                f.write(f"  {metric_name}: {metric_val:.6f}\n")
                    
                    if 'temporal_metrics' in model_results:
                        f.write("\nTemporal Metrics:\n")
                        for metric_name, metric_val in model_results['temporal_metrics'].items():
                            f.write(f"  {metric_name}: {metric_val:.6f}\n")
                    
                    if 'performance' in model_results:
                        f.write("\nPerformance Metrics:\n")
                        for metric_name, metric_val in model_results['performance'].items():
                            f.write(f"  {metric_name}: {metric_val:.4f}\n")
        
        logger.info(f"Text report written to {output_path}")
    
    @staticmethod
    def write_markdown_report(results: Dict, output_path: str):
        """
        Write Markdown report of benchmark results.
        
        Args:
            results: Benchmark results
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write("# Depth Estimation Benchmark Report\n\n")
            
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            if 'models' in results:
                f.write("## Model Performance Summary\n\n")
                
                f.write("| Model | MAE | RMSE | AbsRel | SILog | Consistency |\n")
                f.write("|-------|-----|------|--------|-------|-------------|\n")
                
                for model_name, model_results in results['models'].items():
                    mae = "N/A"
                    rmse = "N/A"
                    abs_rel = "N/A"
                    silog = "N/A"
                    consistency = "N/A"
                    
                    if 'depth_metrics' in model_results:
                        metrics = model_results['depth_metrics']
                        if 'mae' in metrics:
                            val = metrics['mae']
                            mae = f"{val.get('mean', 0):.4f}" if isinstance(val, dict) else f"{val:.4f}"
                        if 'rmse' in metrics:
                            val = metrics['rmse']
                            rmse = f"{val.get('mean', 0):.4f}" if isinstance(val, dict) else f"{val:.4f}"
                        if 'abs_rel' in metrics:
                            val = metrics['abs_rel']
                            abs_rel = f"{val.get('mean', 0):.4f}" if isinstance(val, dict) else f"{val:.4f}"
                        if 'silog' in metrics:
                            val = metrics['silog']
                            silog = f"{val.get('mean', 0):.4f}" if isinstance(val, dict) else f"{val:.4f}"
                    
                    if 'temporal_metrics' in model_results:
                        temporal = model_results['temporal_metrics']
                        if 'consistency' in temporal:
                            consistency = f"{temporal['consistency']:.4f}"
                    
                    f.write(f"| {model_name} | {mae} | {rmse} | {abs_rel} | {silog} | {consistency} |\n")
        
        logger.info(f"Markdown report written to {output_path}")
