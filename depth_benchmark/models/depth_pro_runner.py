"""
Depth Pro depth estimation model runner.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Tuple
from pathlib import Path
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class DepthProRunner(BaseDepthModel):
    """Depth Pro depth estimation."""
    
    def __init__(self, device: str = 'cuda', use_fp16: bool = True):
        """Initialize Depth Pro."""
        super().__init__('DepthPro', device, use_fp16,
                        input_size=(1536, 1536))
        
        self.load_model()
    
    def load_model(self):
        """Load Depth Pro model."""
        import sys
        upstream_path = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'ml-depth-pro')
        if upstream_path not in sys.path:
            sys.path.insert(0, upstream_path)
        try:
            from depth_pro import create_model_and_transforms
        except ImportError:
            raise ImportError(
                "Depth Pro not installed. Install with: "
                "pip install git+https://github.com/apple/depth-pro.git"
            )
        
        logger.info("Loading Depth Pro")
        
        from depth_pro.depth_pro import DepthProConfig
        config = DepthProConfig(
            patch_encoder_preset='dinov2l16_384',
            image_encoder_preset='dinov2l16_384',
            decoder_features=256,
            use_fov_head=False,
        )
        self.model, self.transforms = create_model_and_transforms(config=config, device=self.device)
        
        ckpt_path = str(Path(__file__).resolve().parent.parent / 'checkpoints' / 'depth_pro.pt')
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_filtered = {k: v for k, v in state.items() if not k.startswith('fov.')}
        missing, unexpected = self.model.load_state_dict(state_filtered, strict=False)
        if missing:
            logger.warning(f"Depth Pro missing keys (non-fatal): {missing}")
        if unexpected:
            logger.warning(f"Depth Pro unexpected keys (non-fatal): {unexpected}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Depth Pro loaded successfully")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image."""
        from PIL import Image
        image_pil = Image.fromarray(image)
        image_tensor = self.transforms(image_pil)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass - use model.infer or direct forward."""
        with torch.no_grad():
            f_px = torch.tensor([500.0], device=self.device)
            depth = self.model.infer(image, f_px=f_px)['depth']
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
