"""
IEBins depth estimation runner (Swin-based).
"""
import sys
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
from .base_model import BaseDepthModel

logger = logging.getLogger(__name__)


class IEBinsRunner(BaseDepthModel):
    """IEBins iterative elastic bins depth estimation."""

    def __init__(self, model_size: str = 'swinl', dataset: str = 'nyu', device: str = 'cuda', use_fp16: bool = True):
        super().__init__('IEBins_' + dataset + '_' + model_size, device, use_fp16, input_size=(480, 640))
        self.dataset = dataset
        self.encoder = 'large07' if 'l' in model_size else 'base07'
        self.max_depth = 10 if dataset == 'nyu' else 80
        self.load_model()

    def load_model(self):
        upstream = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'IEBins' / 'iebins')
        if upstream not in sys.path:
            sys.path.insert(0, upstream)

        from networks.NewCRFDepth import NewCRFDepth
        from utils import post_process_depth, flip_lr

        self._flip_lr = flip_lr
        self._post_process_depth = post_process_depth

        logger.info(f"Loading IEBins {self.dataset} {self.encoder}")
        self.model = NewCRFDepth(version=self.encoder, inv_depth=False, max_depth=self.max_depth, pretrained=None)
        self.model = torch.nn.DataParallel(self.model)

        # Checkpoint paths
        ckpt_map = {
            ('nyu', 'large07'): 'IEBins_nyu_swinl.pth',
            ('nyu', 'base07'): 'IEBins_nyu_swint.pth',
            ('kitti', 'large07'): 'IEBins_kitti_swinl.pth',
            ('kitti', 'base07'): 'IEBins_kitti_swint.pth',
        }
        cache = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)

        key = (self.dataset, self.encoder)
        local = cache / ckpt_map.get(key, ckpt_map[('nyu', 'large07')])

        if local.exists():
            checkpoint = torch.load(str(local), map_location='cpu', weights_only=False)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded IEBins checkpoint {local}")
        else:
            logger.warning(f"IEBins checkpoint not found at {local}. "
                           f"Download from Google Drive and place in {cache}")
            logger.warning("NYU-SwinL: https://drive.google.com/file/d/14Rn-vxvpXO2EXRaWqCPmh2JufvOurwtl")
            logger.warning("KITTI-SwinL: https://drive.google.com/file/d/1xaVLDq7zJ-C2GtFvABolSUtK7gzvNQNd")

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"IEBins loaded")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        if self.dataset == 'kitti':
            top = int(h - 352)
            left = int((w - 1216) / 2)
            img = img[top:top + 352, left:left + 1216, :]
            h, w = img.shape[:2]
        # Pad to multiples of 32 so backbone down/up paths align
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        self._pad = (pad_h, pad_w)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        img = (img - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img.unsqueeze(0).to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pred_depths_r_list, _, _ = self.model(image)
        pred = pred_depths_r_list[-1]
        img_flipped = self._flip_lr(image)
        pred_flip, _, _ = self.model(img_flipped)
        pred = self._post_process_depth(pred, pred_flip[-1])
        # Crop padding
        pad_h, pad_w = getattr(self, '_pad', (0, 0))
        if pad_h > 0:
            pred = pred[:, :, :-pad_h, :]
        if pad_w > 0:
            pred = pred[:, :, :, :-pad_w]
        return pred

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = np.clip(depth.squeeze().cpu().float().numpy(), 1e-3, self.max_depth)
        if depth_np.shape[:2] != original_shape:
            depth_np = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        return depth_norm.astype(np.float32)
