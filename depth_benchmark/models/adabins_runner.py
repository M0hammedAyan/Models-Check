"""
AdaBins depth estimation model runner.
"""
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
from .base_model import BaseDepthModel, ImagePreprocessor

logger = logging.getLogger(__name__)


class AdaBinsRunner(BaseDepthModel):
    """AdaBins depth estimation."""

    def __init__(self, dataset: str = 'nyu', device: str = 'cuda', use_fp16: bool = True):
        super().__init__('AdaBins_' + dataset, device, use_fp16, input_size=(480, 640))
        self.dataset = dataset
        self.min_depth = 1e-3
        self.max_depth = 10 if dataset == 'nyu' else 80
        self.load_model()

    def load_model(self):
        import sys
        upstream = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'AdaBins')
        if upstream not in sys.path:
            sys.path.insert(0, upstream)

        adabins_models_path = str(Path(upstream) / 'models')
        sys.path.insert(0, adabins_models_path)

        old_models = sys.modules.pop('models', None)
        try:
            import models as _adabins_models
            UnetAdaptiveBins = _adabins_models.UnetAdaptiveBins
        finally:
            if old_models:
                sys.modules['models'] = old_models

        import model_io

        logger.info(f"Loading AdaBins {self.dataset}")
        self.model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)

        # Download weights if needed
        cache = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)

        if self.dataset == 'nyu':
            url = 'https://github.com/shariqfarooq123/AdaBins/releases/download/v1.0/AdaBins_nyu-256-2fb686a.pth'
            local = cache / 'AdaBins_nyu-256-2fb686a.pth'
        else:
            url = 'https://github.com/shariqfarooq123/AdaBins/releases/download/v1.0/AdaBins_kitti-256-2fb686a.pth'
            local = cache / 'AdaBins_kitti-256-2fb686a.pth'

        if not local.exists():
            logger.info(f"Downloading AdaBins weights...")
            torch.hub.download_url_to_file(url, str(local))

        ckpt = torch.load(str(local), map_location='cpu')
        if 'epoch' in ckpt:
            model, _, _ = model_io.load_checkpoint(str(local), self.model)
            self.model = model
        else:
            self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Download backbone if needed
        if not hasattr(self.model, '_backbone_loaded'):
            try:
                _ = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ap',
                                   pretrained=True, trust_repo=True)
                self.model._backbone_loaded = True
            except Exception:
                logger.warning("Could not load efficientnet backbone from hub, may fail at runtime")

        logger.info(f"AdaBins {self.dataset} loaded")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = image.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))
        img = (img - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img.unsqueeze(0).to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        bins, pred = self.model(image)
        return pred

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = np.clip(depth.squeeze().cpu().float().numpy(), self.min_depth, self.max_depth)
        depth_resized = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
        return depth_norm.astype(np.float32)
