"""
Depth Anything v2 depth estimation model runner.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Tuple
from pathlib import Path
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class DepthAnythingV2Runner(BaseDepthModel):
    """Depth Anything v2 depth estimation."""
    
    def __init__(self, model_type: str = 'small', device: str = 'cuda',
                 use_fp16: bool = True):
        """Initialize Depth Anything V2."""
        input_sizes = {
            'small': (518, 518),
            'base': (518, 518),
            'large': (518, 518)
        }
        
        super().__init__('DepthAnythingV2_' + model_type, device, use_fp16,
                        input_size=input_sizes.get(model_type, (518, 518)))
        
        self.model_type = model_type
        self.load_model()
    
    def load_model(self):
        """Load Depth Anything v2 model."""
        import os
        os.environ['XFORMERS_DISABLED'] = '1'
        import sys
        upstream_path = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'Depth-Anything-V2')
        if upstream_path not in sys.path:
            sys.path.insert(0, upstream_path)
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ImportError(
                "Depth Anything v2 not installed. Install with: "
                "pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
            )
        
        logger.info(f"Loading Depth Anything V2 {self.model_type}")
        
        encoder_map = {'small': 'vits', 'base': 'vitb', 'large': 'vitl'}
        encoder = encoder_map.get(self.model_type, 'vits')

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        mc = model_configs[encoder]

        self.model = DepthAnythingV2(
            encoder=mc['encoder'],
            features=mc['features'],
            out_channels=mc['out_channels']
        )
        
        # Download weights
        ckpt_map = {
            'vits': 'depth_anything_v2_vits.pth',
            'vitb': 'depth_anything_v2_vitb.pth',
            'vitl': 'depth_anything_v2_vitl.pth',
        }
        cache = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        local_path = cache / ckpt_map[encoder]
        
        if not local_path.exists():
            try:
                from huggingface_hub import hf_hub_download
                _da_repo_map = {'small': 'Small', 'base': 'Base', 'large': 'Large'}
                hf_hub_download(
                    repo_id=f'depth-anything/Depth-Anything-V2-{_da_repo_map.get(self.model_type, "Small")}',
                    filename=ckpt_map[encoder],
                    local_dir=cache,
                    local_dir_use_symlinks=False,
                )
            except Exception:
                logger.warning(f"Could not download Depth-Anything-V2 checkpoint. "
                               f"Place {ckpt_map[encoder]} manually in {cache}")

        if local_path.exists():
            state = torch.load(str(local_path), map_location='cpu')
            self.model.load_state_dict(state)

        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Depth Anything V2 {self.model_type} loaded successfully")
    
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
