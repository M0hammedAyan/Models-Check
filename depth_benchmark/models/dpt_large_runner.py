"""
DPT-Large depth estimation model runner.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Tuple
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class DPTLargeRunner(BaseDepthModel):
    """DPT-Large depth estimation."""
    
    def __init__(self, device: str = 'cuda', use_fp16: bool = True):
        """Initialize DPT-Large."""
        super().__init__('DPT_Large', device, use_fp16,
                        input_size=(384, 384))
        
        self.load_model()
    
    def load_model(self):
        """Load DPT-Large model."""
        logger.info("Loading DPT-Large")
        
        try:
            import timm
            if 'dpt_large_384' in timm.list_models():
                self.model = timm.create_model('dpt_large_384', pretrained=True, in_chans=3)
            else:
                raise ImportError("dpt_large_384 not in timm registry")
        except Exception:
            logger.info("timm dpt not available, falling back to torch.hub")
            self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True, compile=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("DPT-Large loaded successfully")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image."""
        image_resized = cv2.resize(
            image,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        image_norm = ImagePreprocessor.normalize_image(image_resized)
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        
        return image_tensor
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        with torch.no_grad():
            depth = self.model(image)
        
        if isinstance(depth, (list, tuple)):
            depth = depth[0]
        
        return depth
    
    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess depth map."""
        depth_np = depth.squeeze().cpu().float().numpy()
        depth_np = 1.0 / (depth_np + 1e-8)
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        
        depth_resized = cv2.resize(
            depth_np,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        return depth_resized.astype(np.float32)
