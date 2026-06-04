"""PatchFusion: tile-based high-res monocular metric depth."""
import sys
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
import torch.nn.functional as F
from torchvision import transforms
from .base_model import BaseDepthModel

logger = logging.getLogger(__name__)


class PatchFusionRunner(BaseDepthModel):
    """PatchFusion tile-based high-res depth estimation."""

    def __init__(self, model_name: str = 'Zhyever/patchfusion_depth_anything_vitl14',
                 cai_mode: str = 'r128', device: str = 'cuda', use_fp16: bool = True):
        super().__init__('PatchFusion', device, use_fp16, input_size=(2160, 3840))
        self.model_name = model_name
        self.cai_mode = cai_mode
        self.load_model()

    def load_model(self):
        pf_dir = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'PatchFusion')
        # Keep pf_dir last among our additions so our models/ package isn't shadowed
        if pf_dir in sys.path:
            sys.path.remove(pf_dir)

        external = str(Path(pf_dir) / 'external')
        if external in sys.path:
            sys.path.remove(external)

        # Insert external first so zoedepth resolves from there
        sys.path.insert(0, external)
        sys.path.insert(0, pf_dir)
        # Do NOT add external/zoedepth to sys.path — it shadows our models/

        logger.info(f"Loading PatchFusion from {self.model_name}")
        try:
            from estimator.models.patchfusion import PatchFusion
        except ImportError:
            raise ImportError("PatchFusion requires additional dependencies. "
                              "Install with: conda env create -f environment.yml")

        self.model = PatchFusion.from_pretrained(self.model_name).to(self.device).eval()
        self.default_resolution = self.model.tile_cfg['image_raw_shape']
        self.resizer = self.model.resizer
        logger.info(f"PatchFusion loaded (default res: {self.default_resolution})")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        img = transforms.ToTensor()(np.asarray(img))
        return img.to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_lr = self.resizer(image.unsqueeze(dim=0)).float().to(self.device)
        image_hr = F.interpolate(image.unsqueeze(dim=0), self.default_resolution,
                                 mode='bicubic', align_corners=True).float().to(self.device)
        process_num = 4
        depth, _ = self.model(mode='infer', cai_mode=self.cai_mode, process_num=process_num,
                              image_lr=image_lr, image_hr=image_hr)
        depth = F.interpolate(depth, image.shape[-2:])[0, 0]
        return depth.unsqueeze(0).unsqueeze(0)

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = depth.squeeze().cpu().numpy()
        depth_np = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        return depth_norm.astype(np.float32)
