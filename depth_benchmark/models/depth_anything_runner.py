"""
Depth Anything depth estimation model runner (v1).
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Tuple, Optional
from pathlib import Path
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class DepthAnythingRunner(BaseDepthModel):
    """Depth Anything v1 depth estimation."""
    
    def __init__(self, model_type: str = 'large', device: str = 'cuda',
                 use_fp16: bool = True):
        """
        Initialize Depth Anything.
        
        Args:
            model_type: 'small', 'base', or 'large'
            device: 'cuda' or 'cpu'
            use_fp16: Use mixed precision
        """
        input_sizes = {
            'small': (518, 518),
            'base': (518, 518),
            'large': (518, 518)
        }
        
        super().__init__('DepthAnything_' + model_type, device, use_fp16,
                        input_size=input_sizes.get(model_type, (518, 518)))
        
        self.model_type = model_type
        self.load_model()
    
    def load_model(self):
        """Load Depth Anything model."""
        import os
        os.environ['XFORMERS_DISABLED'] = '1'
        import sys
        upstream_path = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'Depth-Anything')
        if upstream_path not in sys.path:
            sys.path.insert(0, upstream_path)
        try:
            from depth_anything.dpt import DPT_DINOv2
            from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        except ImportError:
            raise ImportError(
                "Depth Anything not installed. Install with: "
                "git clone https://github.com/LiheYoung/Depth-Anything && "
                "cd Depth-Anything && pip install -e ."
            )
        
        logger.info(f"Loading Depth Anything {self.model_type}")
        
        # Model mapping
        model_configs = {
            'small': 'vits',
            'base': 'vitb',
            'large': 'vitl'
        }
        
        model_config = model_configs.get(self.model_type, 'vitl')
        
        self.model = DPT_DINOv2(
            encoder=model_config,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            localhub=False
        )
        
        # Load pretrained weights (try GitHub releases as HF repo is gated)
        ckpt_map = {
            'small': 'depth_anything_vits14.pth',
            'base': 'depth_anything_vitb14.pth',
            'large': 'depth_anything_vitl14.pth',
        }
        ckpt_name = ckpt_map[self.model_type]
        cache = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        local_path = cache / ckpt_name
        
        if not local_path.exists():
            urls = [
                f'https://github.com/LiheYoung/Depth-Anything/releases/download/v1.0/{ckpt_name}',
                f'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/{ckpt_name}',
            ]
            downloaded = False
            for url in urls:
                try:
                    logger.info(f"Trying {url}")
                    torch.hub.download_url_to_file(url, str(local_path))
                    downloaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed: {e}")
            if not downloaded:
                raise RuntimeError(f"Could not download depth_anything checkpoint. "
                                   f"Place {ckpt_name} manually in {cache}")
        
        state_dict = torch.load(str(local_path), map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Depth Anything {self.model_type} loaded successfully")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for Depth Anything."""
        # Resize
        image_resized = cv2.resize(
            image,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_CUBIC
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
        
        return depth
    
    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess depth map."""
        # Convert to numpy
        depth_np = depth.squeeze().cpu().float().numpy()
        
        # Normalize
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        
        # Resize to original shape
        depth_resized = cv2.resize(
            depth_np,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        return depth_resized.astype(np.float32)
