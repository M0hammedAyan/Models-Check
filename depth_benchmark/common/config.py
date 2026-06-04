"""
Configuration management for depth benchmarking pipeline.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for individual depth model."""
    name: str
    model_type: str
    input_size: tuple
    use_cuda: bool = True
    mixed_precision: bool = True
    checkpoint_path: str = None
    
    
@dataclass
class DataConfig:
    """Configuration for input data."""
    video_path: str = None
    bag_path: str = None
    output_fps: int = 30
    resolution: tuple = (1920, 1080)
    
    
@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    apply_temporal_refinement: bool = True
    temporal_window_size: int = 5
    batch_size: int = 1
    enable_tiling_4k: bool = True
    tile_size: int = 512
    use_fp16: bool = True
    

@dataclass
class PoseConfig:
    """Configuration for pose estimation."""
    enable_pose: bool = True
    pose_model: str = "mediapipe"  # mediapipe or openpose
    min_confidence: float = 0.5
    

@dataclass
class BenchmarkConfig:
    """Master configuration."""
    data: DataConfig
    processing: ProcessingConfig
    pose: PoseConfig
    models: Dict[str, ModelConfig]
    output_dir: str
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary."""
        data = DataConfig(**config_dict.get('data', {}))
        processing = ProcessingConfig(**config_dict.get('processing', {}))
        pose = PoseConfig(**config_dict.get('pose', {}))
        
        models = {}
        for name, model_cfg in config_dict.get('models', {}).items():
            models[name] = ModelConfig(name=name, **model_cfg)
        
        return cls(
            data=data,
            processing=processing,
            pose=pose,
            models=models,
            output_dir=config_dict.get('output_dir', './outputs'),
            device=config_dict.get('device', 'cuda'),
            num_workers=config_dict.get('num_workers', 4),
            seed=config_dict.get('seed', 42)
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data': asdict(self.data),
            'processing': asdict(self.processing),
            'pose': asdict(self.pose),
            'models': {name: asdict(cfg) for name, cfg in self.models.items()},
            'output_dir': self.output_dir,
            'device': self.device,
            'num_workers': self.num_workers,
            'seed': self.seed
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Configuration saved to {yaml_path}")
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            'data': asdict(self.data),
            'processing': asdict(self.processing),
            'pose': asdict(self.pose),
            'models': {name: asdict(cfg) for name, cfg in self.models.items()},
            'output_dir': self.output_dir,
            'device': self.device,
            'num_workers': self.num_workers,
            'seed': self.seed
        }
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {json_path}")


# Default configuration template
DEFAULT_CONFIG = {
    'data': {
        'video_path': None,
        'bag_path': None,
        'output_fps': 30,
        'resolution': [1920, 1080]
    },
    'processing': {
        'apply_temporal_refinement': True,
        'temporal_window_size': 5,
        'batch_size': 1,
        'enable_tiling_4k': True,
        'tile_size': 512,
        'use_fp16': True
    },
    'pose': {
        'enable_pose': True,
        'pose_model': 'mediapipe',
        'min_confidence': 0.5
    },
    'models': {
        'midas': {
            'model_type': 'midas',
            'input_size': [384, 384],
            'use_cuda': True,
            'mixed_precision': True
        },
        'depth_anything': {
            'model_type': 'depth_anything',
            'input_size': [518, 518],
            'use_cuda': True,
            'mixed_precision': True
        },
        'depth_anything_v2': {
            'model_type': 'depth_anything_v2',
            'input_size': [518, 518],
            'use_cuda': True,
            'mixed_precision': True
        }
    },
    'output_dir': './outputs',
    'device': 'cuda',
    'num_workers': 4,
    'seed': 42
}


def create_default_config(output_path: str = 'config.yaml'):
    """Create and save default configuration file."""
    config = BenchmarkConfig.from_dict(DEFAULT_CONFIG)
    config.to_yaml(output_path)
    logger.info(f"Default configuration created at {output_path}")
    return config
