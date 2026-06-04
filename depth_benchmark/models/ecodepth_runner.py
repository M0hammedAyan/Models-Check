"""ECoDepth: effective conditioning of diffusion for monocular depth."""
import sys
import math
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from .base_model import BaseDepthModel

logger = logging.getLogger(__name__)


class ECoDepthRunner(BaseDepthModel):
    """ECoDepth diffusion-based monocular depth."""

    def __init__(self, scene: str = 'indoor', device: str = 'cuda', use_fp16: bool = True):
        super().__init__('ECoDepth_' + scene, device, use_fp16, input_size=(480, 640))
        self.scene = scene
        self.max_depth = 10.0 if scene == 'indoor' else 80.0
        self.no_of_classes = 100 if scene == 'indoor' else 200
        self.load_model()

    def load_model(self):
        upstream = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'EcoDepth')
        if upstream not in sys.path:
            sys.path.insert(0, upstream)

        from model import EcoDepth
        from utils import download_model

        logger.info(f"Loading ECoDepth {self.scene}")

        class Args:
            max_depth = self.max_depth
            no_of_classes = self.no_of_classes
            train_from_scratch = False
            eval_crop = 'no_crop'
            enable_v2 = True
            ckpt_path = ''
            scene = self.scene

        self.model = EcoDepth(Args())
        version = 'ecodepthv2'
        model_str = f'weights_{self.scene}.ckpt'
        download_model(model_str, version)
        checkpoint_dir = Path(__file__).resolve().parent.parent / 'checkpoints'
        ckpt = checkpoint_dir / model_str
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location='cpu', weights_only=False)
            self.model.load_state_dict(state['state_dict'])

        self.model = self.model.to(self.device).eval()
        logger.info(f"ECoDepth loaded")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        h, w = image.shape[:2]
        base_area = 3 * 480 * 640
        curr_area = w * h
        scale = math.sqrt(base_area / curr_area)
        new_w, new_h = int(scale * w), int(scale * h)
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        img = transforms.ToTensor()(img).unsqueeze(0)
        return img.to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            depth = self.model(image)
        return depth

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = np.clip(depth.squeeze().cpu().numpy(), 1e-3, self.max_depth)
        depth_np = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        return depth_norm.astype(np.float32)
