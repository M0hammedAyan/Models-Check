"""
BTS (From Big to Small) depth estimation model runner.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Tuple
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class BTSRunner(BaseDepthModel):
    """BTS depth estimation."""
    
    def __init__(self, model_type: str = 'bts_sift', device: str = 'cuda',
                 use_fp16: bool = True):
        """Initialize BTS."""
        super().__init__('BTS_' + model_type, device, use_fp16,
                        input_size=(384, 384))
        
        self.model_type = model_type
        self.load_model()
    
    def load_model(self):
        """Load BTS model."""
        import sys
        import argparse
        from pathlib import Path
        bts_path = Path(__file__).resolve().parent.parent / 'upstream_repos' / 'bts' / 'pytorch'
        if str(bts_path) not in sys.path:
            sys.path.insert(0, str(bts_path))
        try:
            from bts import BtsModel
        except ImportError:
            raise ImportError(
                "BTS not found. Clone with: "
                "git clone https://github.com/cleinc/bts"
            )
        
        logger.info(f"Loading BTS {self.model_type}")
        
        params = argparse.Namespace(
            encoder='densenet161_bts',
            max_depth=80,
            bts_size=512,
            input_height=384,
            input_width=384,
            dataset='nyu',
        )
        
        self.model = BtsModel(params=params)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"BTS {self.model_type} loaded (no pretrained weights)")
    
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
            focal = torch.tensor([518.0], device=self.device)
            outputs = self.model(image, focal)
            depth = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        
        return depth
    
    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess depth map."""
        depth_np = depth.squeeze().cpu().float().numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        
        depth_resized = cv2.resize(
            depth_np,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        return depth_resized.astype(np.float32)
