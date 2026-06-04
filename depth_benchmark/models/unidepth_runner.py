"""
UniDepth universal metric depth estimation runner.
"""
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple
from .base_model import BaseDepthModel

logger = logging.getLogger(__name__)


class UniDepthRunner(BaseDepthModel):
    """UniDepth universal metric depth estimation."""

    def __init__(self, model_type: str = 'v2', device: str = 'cuda', use_fp16: bool = True):
        super().__init__('UniDepth_' + model_type, device, use_fp16, input_size=(672, 672))
        self.model_type = model_type
        self.load_model()

    def load_model(self):
        import sys
        upstream = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'UniDepth')
        if upstream not in sys.path:
            sys.path.insert(0, upstream)

        logger.info(f"Loading UniDepth {self.model_type}")

        repo_id_map = {
            'v2': 'lpiccinelli/unidepth-v2-vitl14',
            'v1': 'lpiccinelli/unidepth-v1-vitl14',
        }
        repo_id = repo_id_map.get(self.model_type, 'lpiccinelli/unidepth-v2-vitl14')

        try:
            from unidepth.models import UniDepthV2
            self.model = UniDepthV2.from_pretrained(repo_id)
        except Exception:
            try:
                from unidepth.models import UniDepthV1
                self.model = UniDepthV1.from_pretrained(repo_id)
            except Exception:
                from huggingface_hub import hf_hub_download
                import json
                from unidepth.models import UniDepthV2
                config_path = hf_hub_download(repo_id, 'config.json')
                with open(config_path) as f:
                    config = json.load(f)
                self.model = UniDepthV2(config)
                weights_path = hf_hub_download(repo_id, 'pytorch_model.bin')
                self.model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"UniDepth {self.model_type} loaded")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        return tensor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        predictions = self.model.infer(image)
        return predictions['depth']

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = depth.squeeze().cpu().float().numpy()
        if depth_np.shape[:2] != original_shape:
            depth_np = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        return depth_norm.astype(np.float32)
