"""BinsFormer: adaptive bins for monocular depth estimation."""
import sys
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
from .base_model import BaseDepthModel

logger = logging.getLogger(__name__)


class BinsFormerRunner(BaseDepthModel):
    """BinsFormer with adaptive bins for depth estimation."""

    def __init__(self, model_size: str = 'swinl', dataset: str = 'nyu',
                 device: str = 'cuda', use_fp16: bool = True):
        super().__init__('BinsFormer_' + dataset + '_' + model_size, device, use_fp16, input_size=(480, 640))
        self.model_size = model_size
        self.dataset = dataset
        self.load_model()

    def load_model(self):
        toolbox = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'Monocular-Depth-Estimation-Toolbox')
        if toolbox not in sys.path:
            sys.path.insert(0, toolbox)

        logger.info(f"Loading BinsFormer {self.dataset} {self.model_size}")
        try:
            from depth.apis.inference import init_depther
        except ImportError:
            raise ImportError("BinsFormer requires MMSegmentation toolbox. Install with: "
                              "pip install mmsegmentation openmim && mim install mmcv-full")

        cfg_map = {
            ('nyu', 'swinl'): 'configs/binsformer/binsformer_swinl_22k_w7_nyu.py',
            ('nyu', 'swint'): 'configs/binsformer/binsformer_swint_w7_nyu.py',
            ('kitti', 'swinl'): 'configs/binsformer/binsformer_swinl_22k_w7_kitti.py',
        }
        key = (self.dataset, self.model_size)
        cfg_rel = cfg_map.get(key, cfg_map[('nyu', 'swint')])
        cfg_path = str(Path(toolbox) / cfg_rel)

        ckpt_map = {
            ('nyu', 'swinl'): 'binsformer_swinl_22k_w7_nyu.pth',
            ('nyu', 'swint'): 'binsformer_swint_w7_nyu.pth',
            ('kitti', 'swinl'): 'binsformer_swinl_22k_w7_kitti.pth',
        }
        cache = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        local = cache / ckpt_map[key]

        if not local.exists():
            logger.warning(f"BinsFormer checkpoint not found at {local}")
            logger.warning("Download from Google Drive and place in cache dir:")
            logger.warning("NYU-SwinL: https://drive.google.com/file/d/1j1FmtXKSOD5e6HWBBd_3cwI2M11Jd_nB")
            logger.warning("NYU-SwinT: https://drive.google.com/file/d/1tcWx_BQBNJHpP5-RUWGWjpVRfeUiUMzJ")

        self.model = init_depther(cfg_path, str(local) if local.exists() else None, device=self.device)
        logger.info(f"BinsFormer loaded")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return img.to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        from depth.apis.inference import inference_depther
        result = inference_depther(self.model, image)
        depth = torch.from_numpy(result[0]).float()
        return depth.unsqueeze(0).unsqueeze(0)

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = depth.squeeze().cpu().numpy()
        depth_np = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        return depth_norm.astype(np.float32)
