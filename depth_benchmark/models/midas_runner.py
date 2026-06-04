"""
MiDaS v3 depth estimation model runner.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Tuple, Optional
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class MiDaSRunner(BaseDepthModel):
    """MiDaS v3 depth estimation."""
    
    def __init__(self, model_type: str = 'large', device: str = 'cuda',
                 use_fp16: bool = True):
        """
        Initialize MiDaS.
        
        Args:
            model_type: 'small', 'base', or 'large'
            device: 'cuda' or 'cpu'
            use_fp16: Use mixed precision
        """
        input_sizes = {
            'small': (256, 256),
            'base': (384, 384),
            'large': (384, 384)
        }
        
        super().__init__('MiDaS_' + model_type, device, use_fp16,
                        input_size=input_sizes.get(model_type, (384, 384)))
        
        self.model_type = model_type
        self.load_model()
    
    def load_model(self):
        """Load MiDaS model."""
        logger.info(f"Loading MiDaS {self.model_type}")
        
        try:
            import timm
            model_names = {
                'small': 'midasv3_small',
                'base': 'midasv3',
                'large': 'midasv3_large'
            }
            model_name = model_names.get(self.model_type, 'midasv3')
            
            if model_name in timm.list_models():
                self.model = timm.create_model(model_name, pretrained=True, in_chans=3)
            else:
                raise ImportError(f"Model {model_name} not in timm registry")
        except Exception:
            logger.info("timm midas not available, falling back to torch.hub")
            repo = 'intel-isl/MiDaS'
            model_map = {'small': 'MiDaS_small', 'base': 'MiDaS', 'large': 'MiDaS'}
            hub_model = model_map.get(self.model_type, 'MiDaS')
            self.model = torch.hub.load(repo, hub_model, trust_repo=True)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"MiDaS {self.model_type} loaded successfully")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for MiDaS."""
        # Resize
        image_resized = ImagePreprocessor.resize_to_net_input(
            image, self.input_size, keep_aspect_ratio=True
        )
        
        # Normalize
        image_norm = ImagePreprocessor.normalize_image(image_resized)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        
        return image_tensor
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        with torch.no_grad():
            depth = self.model(image)
        
        # MiDaS returns disparity, convert to depth
        if isinstance(depth, (list, tuple)):
            depth = depth[0]
        
        # Ensure single output
        if depth.dim() > 3:
            depth = depth.squeeze(0)
        
        return depth
    
    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess depth map."""
        # Convert to numpy
        depth_np = depth.squeeze().cpu().float().numpy()
        
        # Invert if disparity
        depth_np = 1.0 / (depth_np + 1e-8)
        
        # Normalize
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        
        # Resize to original shape
        depth_resized = cv2.resize(
            depth_np,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        return depth_resized.astype(np.float32)
