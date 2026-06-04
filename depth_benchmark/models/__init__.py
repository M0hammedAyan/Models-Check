"""
Model runners for depth estimation.
"""

from .base_model import BaseDepthModel, TiledInference, DepthNormalizer, ImagePreprocessor
from .midas_runner import MiDaSRunner
from .depth_anything_runner import DepthAnythingRunner
from .depth_anything_v2_runner import DepthAnythingV2Runner
from .zoedepth_runner import ZoeDepthRunner
from .dpt_large_runner import DPTLargeRunner
from .depth_pro_runner import DepthProRunner
from .leres_runner import LeReSRunner
from .bts_runner import BTSRunner
from .adabins_runner import AdaBinsRunner
from .unidepth_runner import UniDepthRunner
from .metric3d_runner import Metric3DRunner
from .iebins_runner import IEBinsRunner
from .depthfm_runner import DepthFMRunner
from .lotus_runner import LotusRunner
from .ecodepth_runner import ECoDepthRunner
from .binsformer_runner import BinsFormerRunner
from .patchfusion_runner import PatchFusionRunner


class ModelFactory:
    """Factory for creating depth estimation models."""
    
    AVAILABLE_MODELS = {
        'midas': MiDaSRunner,
        'depth_anything': DepthAnythingRunner,
        'depth_anything_v2': DepthAnythingV2Runner,
        'zoedepth': ZoeDepthRunner,
        'dpt_large': DPTLargeRunner,
        'depth_pro': DepthProRunner,
        'leres': LeReSRunner,
        'bts': BTSRunner,
        'adabins': AdaBinsRunner,
        'unidepth': UniDepthRunner,
        'metric3d': Metric3DRunner,
        'iebins': IEBinsRunner,
        'depthfm': DepthFMRunner,
        'lotus': LotusRunner,
        'ecodepth': ECoDepthRunner,
        'binsformer': BinsFormerRunner,
        'patchfusion': PatchFusionRunner,
    }
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> BaseDepthModel:
        """
        Create a depth estimation model.
        
        Args:
            model_name: Model name (key in AVAILABLE_MODELS)
            **kwargs: Additional arguments passed to model constructor
            
        Returns:
            Model instance
        """
        if model_name not in cls.AVAILABLE_MODELS:
            available = ', '.join(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        
        model_class = cls.AVAILABLE_MODELS[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """List all available models."""
        return list(cls.AVAILABLE_MODELS.keys())


__all__ = [
    'BaseDepthModel', 'TiledInference', 'DepthNormalizer', 'ImagePreprocessor',
    'MiDaSRunner', 'DepthAnythingRunner', 'DepthAnythingV2Runner',
    'ZoeDepthRunner', 'DPTLargeRunner', 'DepthProRunner',
    'LeReSRunner', 'BTSRunner',
    'AdaBinsRunner', 'UniDepthRunner', 'Metric3DRunner', 'IEBinsRunner',
    'DepthFMRunner', 'LotusRunner', 'ECoDepthRunner', 'BinsFormerRunner', 'PatchFusionRunner',
    'ModelFactory',
]
