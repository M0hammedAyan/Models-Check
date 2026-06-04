"""
LeReS depth estimation model runner.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Tuple
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class LeReSRunner(BaseDepthModel):
    """LeReS depth estimation."""
    
    def __init__(self, device: str = 'cuda', use_fp16: bool = True):
        """Initialize LeReS."""
        super().__init__('LeReS', device, use_fp16,
                        input_size=(384, 384))
        
        self.load_model()
    
    def load_model(self):
        """Load LeReS model."""
        try:
            from leres.network import Network

            logger.info("Loading LeReS via leres.network.Network")
            self.model = Network(use_cuda=self.device.type == 'cuda')
            logger.info("LeReS loaded successfully (leres.network)")
            return
        except Exception:
            logger.debug("leres.network not available, falling back to AnyDepth adapter")

        # Try to use AIGeeksGroup/AnyDepth (AnyDepth) as a fallback. We create a
        # lightweight adapter that builds a depth model via AnyDepth's
        # `build_depther` API. This adapter uses a very small dummy backbone so
        # the model can be instantiated; for best results replace the dummy
        # backbone with a proper DINOv2/DINOv3 backbone as used by AnyDepth.
        try:
            # Import build_depther from the AnyDepth package (when installed)
            try:
                # Prefer installed package
                from anydepth.model import build_depther  # type: ignore
            except Exception:
                # Fall back to local checkout under upstream_repos if present
                import importlib.util
                from pathlib import Path

                repo_path = Path(__file__).parents[1] / 'upstream_repos' / 'AnyDepth' / 'model'
                if repo_path.exists():
                    spec = importlib.util.spec_from_file_location('anydepth.model', str(repo_path / '__init__.py'))
                    anydepth_model = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(anydepth_model)  # type: ignore
                    build_depther = getattr(anydepth_model, 'build_depther')
                else:
                    raise ImportError('AnyDepth not installed and local checkout not found')

            import torch.nn as nn
            import torch

            class _DummyBackbone(nn.Module):
                """Minimal backbone implementing the interface expected by AnyDepth.

                This is a placeholder: it provides `embed_dim`, `n_blocks`,
                `patch_size`, and a `get_intermediate_layers` method. Replace with
                a proper DINO/DINOv2 backbone for production use.
                """

                def __init__(self, embed_dim: int = 384, n_blocks: int = 12, patch_size: int = 16):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.n_blocks = n_blocks
                    self.patch_size = patch_size
                    self.input_pad_size = patch_size
                    # a tiny conv to map RGB -> embed_dim
                    self.proj = nn.Conv2d(3, embed_dim, kernel_size=1)

                def get_intermediate_layers(self, x, n=1, reshape=True, return_class_token=True, norm=False):
                    # x: [B, C, H, W]
                    feats = self.proj(x)  # [B, embed_dim, H, W]
                    B, C, H, W = feats.shape
                    # create a simple class token as global avgpool
                    class_token = feats.mean(dim=[2, 3])  # [B, C]
                    # Return one stage (patch feats, class token)
                    if reshape:
                        return [(feats, class_token)]
                    else:
                        return [(feats.view(B, C, -1), class_token)]

            logger.info("Building AnyDepth-compatible model (placeholder backbone)")
            backbone = _DummyBackbone()

            # build a small model: 1 output channel (depth)
            self.model = build_depther(
                backbone=backbone,
                backbone_out_layers=[backbone.n_blocks - 1],
                n_output_channels=1,
                head_type='sdt',
            )

            # move to device and eval
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("LeReS adapter loaded via AnyDepth.build_depther (placeholder backbone)")
        except Exception as e:
            raise ImportError(
                "LeReS not installed and AnyDepth fallback failed.\n"
                "Install the original LeReS/AnyDepth package or provide a compatible backbone.\n"
                f"Error: {e}"
            )
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image."""
        image_resized = cv2.resize(
            image,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32) / 255.0
        
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
        depth_np = depth.squeeze().cpu().numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        
        depth_resized = cv2.resize(
            depth_np,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        return depth_resized.astype(np.float32)
