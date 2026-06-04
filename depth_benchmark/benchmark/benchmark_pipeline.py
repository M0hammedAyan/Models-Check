"""
Main benchmarking pipeline for depth estimation models.
"""

import os
import logging
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time

from common import (
    BenchmarkConfig, DepthMetrics, TemporalMetrics, PoseMetrics,
    DepthVisualizer, PoseVisualizer, VideoComparison,
    VideoLoader, RealSenseLoader, apply_temporal_refinement,
    MediaPipePose, PoseProcessor, BodyPartGroups, utils
)
from models import ModelFactory

logger = logging.getLogger(__name__)


class BenchmarkPipeline:
    """Main benchmarking pipeline."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize pipeline.
        
        Args:
            config: BenchmarkConfig instance
        """
        self.config = config
        self.device = torch.device(config.device)
        self.output_paths = utils.create_output_dirs(config.output_dir)
        
        # Initialize components
        self.models = {}
        self.results = {}
        
        # Setup logging
        utils.setup_logging(
            log_level='INFO',
            log_file=os.path.join(config.output_dir, 'benchmark.log')
        )
        
        # Set seed
        utils.set_seed(config.seed)
        
        logger.info("BenchmarkPipeline initialized")
    
    def load_models(self):
        """Load all models specified in config."""
        logger.info("Loading models...")
        
        for model_name, model_config in self.config.models.items():
            try:
                logger.info(f"Loading {model_name}...")
                
                model = ModelFactory.create_model(
                    model_config.model_type,
                    device=self.config.device,
                    use_fp16=model_config.mixed_precision
                )
                
                self.models[model_name] = model
                logger.info(f"  ✓ {model_name} loaded successfully")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to load {model_name}: {e}")
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process video through all models.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Results dictionary
        """
        logger.info(f"Processing video: {video_path}")
        
        video_results = {
            'video_path': video_path,
            'models': {}
        }
        
        # Load video
        with VideoLoader(video_path, target_fps=self.config.data.output_fps) as loader:
            frames_rgb = []
            frames_depth = {}
            
            # Extract frames
            logger.info("Extracting frames...")
            for frame, frame_idx, timestamp in tqdm(loader.get_frames()):
                frames_rgb.append(frame)
                
                # Initialize depth storage for each model
                for model_name in self.models.keys():
                    if model_name not in frames_depth:
                        frames_depth[model_name] = []
            
            # Run inference for each model
            for model_name, model in self.models.items():
                logger.info(f"Running inference for {model_name}...")
                model_depths = []
                
                with torch.no_grad():
                    for frame in tqdm(frames_rgb):
                        depth = model.predict(frame)
                        model_depths.append(depth)
                
                frames_depth[model_name] = model_depths
                
                # Apply temporal refinement
                if self.config.processing.apply_temporal_refinement:
                    logger.info(f"Applying temporal refinement for {model_name}...")
                    depths_array = np.array(model_depths)
                    rgb_array = np.array(frames_rgb)
                    
                    refined_depths = apply_temporal_refinement(
                        depths_array,
                        rgb_array,
                        method='gaussian',
                        device=self.config.device
                    )
                    
                    frames_depth[model_name] = [refined_depths[i] for i in range(len(refined_depths))]
                
                video_results['models'][model_name] = {
                    'depths': frames_depth[model_name],
                    'num_frames': len(frames_depth[model_name])
                }
        
        return video_results, frames_rgb
    
    def process_realsense_bag(self, bag_path: str) -> Tuple[Dict, List, List]:
        """
        Process RealSense .bag file.
        
        Args:
            bag_path: Path to .bag file
            
        Returns:
            (results, rgb_frames, reference_depths)
        """
        logger.info(f"Processing RealSense bag: {bag_path}")
        
        bag_results = {
            'bag_path': bag_path,
            'models': {}
        }
        
        with RealSenseLoader(bag_path, align_to_rgb=True) as loader:
            frames_rgb = []
            frames_depth_ref = []
            frames_depth = {}
            
            # Extract frames
            logger.info("Extracting RGB-D frames...")
            for rgb, depth_ref, frame_idx, timestamp in tqdm(loader.get_frames()):
                frames_rgb.append(rgb)
                frames_depth_ref.append(depth_ref)
                
                for model_name in self.models.keys():
                    if model_name not in frames_depth:
                        frames_depth[model_name] = []
            
            # Run inference
            for model_name, model in self.models.items():
                logger.info(f"Running inference for {model_name}...")
                model_depths = []
                
                with torch.no_grad():
                    for frame in tqdm(frames_rgb):
                        depth = model.predict(frame)
                        model_depths.append(depth)
                
                frames_depth[model_name] = model_depths
                
                # Apply temporal refinement
                if self.config.processing.apply_temporal_refinement:
                    logger.info(f"Applying temporal refinement for {model_name}...")
                    depths_array = np.array(model_depths)
                    rgb_array = np.array(frames_rgb)
                    
                    refined_depths = apply_temporal_refinement(
                        depths_array,
                        rgb_array,
                        method='gaussian'
                    )
                    
                    frames_depth[model_name] = [refined_depths[i] for i in range(len(refined_depths))]
                
                bag_results['models'][model_name] = {
                    'depths': frames_depth[model_name],
                    'num_frames': len(frames_depth[model_name])
                }
        
        return bag_results, frames_rgb, frames_depth_ref
    
    def compute_metrics(self, predicted_depths: List[np.ndarray],
                       reference_depths: List[np.ndarray]) -> Dict:
        """
        Compute all metrics comparing predicted vs reference depths.
        
        Args:
            predicted_depths: List of predicted depth maps
            reference_depths: List of reference depth maps
            
        Returns:
            Metrics dictionary
        """
        logger.info("Computing metrics...")
        
        metrics = {
            'depth_metrics': {},
            'temporal_metrics': {},
        }
        
        # Frame-by-frame metrics
        frame_metrics = []
        
        for pred, ref in zip(predicted_depths, reference_depths):
            # Align depths
            mask = ref > 0
            
            if mask.sum() > 0:
                frame_metric = DepthMetrics.compute_all_metrics(pred, ref, mask=mask)
                frame_metrics.append(frame_metric)
        
        # Aggregate frame metrics
        if frame_metrics:
            metric_keys = frame_metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in frame_metrics if not np.isnan(m.get(key, np.nan))]
                if values:
                    metrics['depth_metrics'][key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                    }
        
        # Temporal metrics
        if len(predicted_depths) > 2:
            predicted_array = np.array(predicted_depths)
            reference_array = np.array(reference_depths)
            
            metrics['temporal_metrics']['consistency'] = float(
                TemporalMetrics.temporal_consistency(predicted_array)
            )
            metrics['temporal_metrics']['flickering'] = float(
                TemporalMetrics.flickering_score(predicted_array)
            )
            metrics['temporal_metrics']['stability'] = float(
                TemporalMetrics.motion_stability(predicted_array)
            )
        
        return metrics
    
    def run_pose_estimation(self, rgb_frames: List[np.ndarray],
                           depth_frames: List[np.ndarray]) -> Dict:
        """
        Run pose estimation on RGB frames.
        
        Args:
            rgb_frames: List of RGB frames
            depth_frames: List of depth maps
            
        Returns:
            Pose results
        """
        if not self.config.pose.enable_pose:
            return {}
        
        logger.info("Running pose estimation...")
        
        pose_detector = MediaPipePose(
            min_confidence=self.config.pose.min_confidence
        )
        
        pose_results = []
        
        for frame in tqdm(rgb_frames):
            result = pose_detector.detect(frame)
            pose_results.append(result)
        
        # Smooth pose sequence
        pose_results = PoseProcessor.smooth_pose_sequence(
            pose_results,
            window_size=self.config.processing.temporal_window_size
        )
        
        pose_detector.close()
        
        return {
            'poses': pose_results,
            'num_frames': len(pose_results),
            'detected_frames': sum(1 for p in pose_results if p is not None)
        }
    
    def save_results(self, results: Dict, output_name: str):
        """Save results to JSON."""
        output_path = os.path.join(self.output_paths['metrics'], f"{output_name}_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = self._convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: BenchmarkPipeline._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BenchmarkPipeline._convert_to_serializable(item) for item in obj]
        return obj
