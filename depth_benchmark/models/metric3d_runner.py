import sys
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
from .base_model import BaseDepthModel

logger = logging.getLogger(__name__)


class Metric3DRunner(BaseDepthModel):
    """Metric3D metric depth estimation."""

    def __init__(self, model_type: str = 'v2', backbone: str = 'vitl', device: str = 'cuda', use_fp16: bool = True):
        super().__init__('Metric3D_' + model_type, device, use_fp16, input_size=(616, 1064))
        self.model_type = model_type
        self.backbone = backbone
        self.pad_info = None
        self.load_model()

    def load_model(self):
        logger.info(f"Loading Metric3D {self.model_type} with {self.backbone} backbone")

        upstream = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'Metric3D')
        if upstream not in sys.path:
            sys.path.insert(0, upstream)

        hub_entry_map = {
            'vitl': 'metric3d_vit_large',
            'vits': 'metric3d_vit_small',
            'convnext_large': 'metric3d_convnext_large',
            'convnext_tiny': 'metric3d_convnext_tiny',
            'vitg': 'metric3d_vit_giant2',
        }

        backbone_key = self.backbone.replace('vit', '').strip()
        if backbone_key == 'l':
            bkey = 'vitl'
        elif backbone_key == 's':
            bkey = 'vits'
        elif backbone_key == 'g' or backbone_key == 'giant2':
            bkey = 'vitg'
        else:
            bkey = self.backbone

        entry = hub_entry_map.get(bkey, 'metric3d_vit_large')

        try:
            self.model = torch.hub.load('yvanyin/metric3d', entry, pretrain=True)
        except Exception as e:
            logger.warning(f"torch.hub.load failed: {e}, falling back to local config")
            from mmengine.config import Config
            from mono.model.monodepth_model import get_configured_monodepth_model

            cfg_map = {
                'vitl': 'mono/configs/HourglassDecoder/vit.raft5.large.py',
                'vits': 'mono/configs/HourglassDecoder/vit.raft5.small.py',
            }
            cfg_path = Path(upstream) / cfg_map.get(bkey, 'mono/configs/HourglassDecoder/vit.raft5.large.py')
            cfg = Config.fromfile(str(cfg_path))
            self.model = get_configured_monodepth_model(cfg)

            ckpt_urls = {
                'vitl': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
                'vits': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
            }
            ckpt_url = ckpt_urls.get(bkey)
            if ckpt_url:
                cache = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
                cache.mkdir(parents=True, exist_ok=True)
                ckpt_name = ckpt_url.split('/')[-1]
                local = cache / ckpt_name
                if local.exists():
                    state = torch.load(str(local), map_location='cpu')
                    self.model.load_state_dict(state['model_state_dict'], strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Metric3D loaded")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        H, W = self.input_size
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        h, w = img_rgb.shape[:2]
        scale = min(H / h, W / w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        padding = [123.675, 116.28, 103.53]
        pad_h = H - nh
        pad_w = W - nw
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        self.pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        img_padded = cv2.copyMakeBorder(img_resized, pad_h_half, pad_h - pad_h_half,
                                        pad_w_half, pad_w - pad_w_half,
                                        cv2.BORDER_CONSTANT, value=padding)

        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).float()
        img_tensor = (img_tensor - mean) / std
        return img_tensor.unsqueeze(0).to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pred_depth, confidence, output_dict = self.model.inference({'input': image})
        return pred_depth

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = depth.squeeze().cpu().float().numpy()
        if self.pad_info:
            pi = self.pad_info
            depth_np = depth_np[pi[0]:depth_np.shape[0] - pi[1], pi[2]:depth_np.shape[1] - pi[3]]
        if depth_np.shape[:2] != original_shape:
            depth_np = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        return depth_norm.astype(np.float32)
