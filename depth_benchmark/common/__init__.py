"""
Common utilities for depth benchmarking pipeline.
"""

from .config import BenchmarkConfig, DataConfig, ProcessingConfig, PoseConfig, ModelConfig
from .utils import (
    setup_logging, set_seed, create_output_dirs, get_device,
    normalize_depth, resize_depth, torch_to_numpy, numpy_to_torch,
    colorize_depth, save_depth_map, load_depth_map
)
from .video_loader import VideoLoader, RealSenseLoader, FrameExtractor
from .depth_alignment import DepthAligner, DepthResizer, CoordinateTransformer, align_depth_maps
from .metrics import DepthMetrics, TemporalMetrics, PoseMetrics, ErrorMap
from .pose_estimation import MediaPipePose, PoseProcessor, BodyPartGroups
from .temporal_refinement import DepthCrafterRefinement, apply_temporal_refinement
from .visualization import DepthVisualizer, PoseVisualizer, VideoComparison

__all__ = [
    'BenchmarkConfig', 'DataConfig', 'ProcessingConfig', 'PoseConfig', 'ModelConfig',
    'setup_logging', 'set_seed', 'create_output_dirs', 'get_device',
    'normalize_depth', 'resize_depth', 'torch_to_numpy', 'numpy_to_torch',
    'colorize_depth', 'save_depth_map', 'load_depth_map',
    'VideoLoader', 'RealSenseLoader', 'FrameExtractor',
    'DepthAligner', 'DepthResizer', 'CoordinateTransformer', 'align_depth_maps',
    'DepthMetrics', 'TemporalMetrics', 'PoseMetrics', 'ErrorMap',
    'MediaPipePose', 'PoseProcessor', 'BodyPartGroups',
    'DepthCrafterRefinement', 'apply_temporal_refinement',
    'DepthVisualizer', 'PoseVisualizer', 'VideoComparison',
]
