"""
ZoeDepth depth estimation model runner.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Tuple
from pathlib import Path
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class ZoeDepthRunner(BaseDepthModel):
    """ZoeDepth depth estimation."""
    
    def __init__(self, model_type: str = 'nyu', device: str = 'cuda',
                 use_fp16: bool = True):
        """Initialize ZoeDepth."""
        super().__init__('ZoeDepth_' + model_type, device, use_fp16,
                        input_size=(384, 384))
        
        self.model_type = model_type
        self.load_model()
    
    def load_model(self):
        """Load ZoeDepth model."""
        import sys
        upstream_path = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'ZoeDepth')
        if upstream_path not in sys.path:
            sys.path.insert(0, upstream_path)
        try:
            import zoedepth
        except ImportError:
            raise ImportError(
                "ZoeDepth not installed. Install with: "
                "pip install git+https://github.com/isl-org/ZoeDepth.git"
            )
        
        logger.info(f"Loading ZoeDepth {self.model_type}")
        
        # Load model using ZoeDepth's config system
        from zoedepth.utils.config import get_config
        from zoedepth.models.builder import build_model
        if self.model_type == 'nyu':
            cfg = get_config("zoedepth_nk", "infer",
                             pretrained_resource="url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt")
        else:
            cfg = get_config("zoedepth", "infer",
                             pretrained_resource="url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt",
                             config_version="kitti")
        self.model = build_model(cfg)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"ZoeDepth {self.model_type} loaded successfully")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image."""
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        image_tensor = image_tensor / 255.0
        
        return image_tensor
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        with torch.no_grad():
            depth = self.model.infer(image)
        
        return depth
    
    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess depth map."""
        depth_np = depth.squeeze().cpu().numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        
        depth_resized = cv2.resize(
            depth_np,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        return depth_resized.astype(np.float32)
