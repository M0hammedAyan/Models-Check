"""DepthFM: fast monocular depth with flow matching."""
import sys
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
from PIL import Image
from PIL.Image import Resampling
from .base_model import BaseDepthModel

logger = logging.getLogger(__name__)


class DepthFMRunner(BaseDepthModel):
    """DepthFM flow-matching depth estimation."""

    def __init__(self, num_steps: int = 2, ensemble_size: int = 4, device: str = 'cuda', use_fp16: bool = True):
        super().__init__('DepthFM', device, use_fp16, input_size=(672, 672))
        self.num_steps = num_steps
        self.ensemble_size = ensemble_size
        self.load_model()

    def load_model(self):
        upstream = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'depth-fm')
        if upstream not in sys.path:
            sys.path.insert(0, upstream)

        from depthfm import DepthFM

        cache = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        ckpt = cache / 'depthfm-v1.ckpt'

        if not ckpt.exists():
            logger.info("Downloading DepthFM checkpoint...")
            torch.hub.download_url_to_file(
                'https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt',
                str(ckpt)
            )

        logger.info("Loading DepthFM (this may take a moment)...")
        self.model = DepthFM(str(ckpt))
        self.model = self.model.to(self.device).eval()
        logger.info("DepthFM loaded")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        orig_h, orig_w = image.shape[:2]
        scale = min(self.input_size[0] / orig_w, self.input_size[1] / orig_h)
        new_w = int(round(orig_w * scale / 64) * 64)
        new_h = int(round(orig_h * scale / 64) * 64)
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return img.to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        dtype = torch.float16 if self.use_fp16 else torch.float32
        with torch.autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu', dtype=dtype):
            depth = self.model.predict_depth(image, num_steps=self.num_steps, ensemble_size=self.ensemble_size)
        return depth

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = depth.squeeze().cpu().float().numpy()
        depth_np = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        return depth_norm.astype(np.float32)
