"""Lotus: diffusion-based visual foundation model for depth."""
import sys
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from .base_model import BaseDepthModel

logger = logging.getLogger(__name__)


class LotusRunner(BaseDepthModel):
    """Lotus single-step diffusion depth estimation."""

    def __init__(self, mode: str = 'regression', model_id: Optional[str] = None,
                 device: str = 'cuda', use_fp16: bool = True):
        super().__init__('Lotus', device, use_fp16, input_size=(768, 768))
        self.mode = mode
        self.model_id = model_id or ('jingheya/lotus-depth-d-v2-0-disparity'
                                     if mode == 'regression'
                                     else 'jingheya/lotus-depth-g-v2-0-disparity')
        self.load_model()

    def load_model(self):
        upstream = str(Path(__file__).resolve().parent.parent / 'upstream_repos' / 'Lotus')
        # Insert at front to ensure Lotus' utils package shadows any other
        if upstream in sys.path:
            sys.path.remove(upstream)
        sys.path.insert(0, upstream)

        import importlib
        pipe_mod = importlib.import_module('pipeline')
        LotusDPipeline = pipe_mod.LotusDPipeline
        LotusGPipeline = pipe_mod.LotusGPipeline

        dtype = torch.float16 if self.use_fp16 else torch.float32
        logger.info(f"Loading Lotus {self.mode} from {self.model_id}")

        pipe_cls = LotusGPipeline if self.mode == 'generation' else LotusDPipeline
        self.pipeline = pipe_cls.from_pretrained(self.model_id, torch_dtype=dtype)
        self.pipeline = self.pipeline.to(self.device)
        logger.info(f"Lotus loaded")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = np.array(img).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img / 127.5 - 1.0
        return img.to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        task_emb = torch.tensor([[1.0, 0.0]], device=self.device).float()
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

        pred = self.pipeline(
            rgb_in=image,
            prompt='',
            num_inference_steps=1,
            output_type='np',
            timesteps=[999],
            task_emb=task_emb,
            processing_res=768,
            match_input_res=True,
        ).images[0]

        depth = torch.from_numpy(pred.mean(axis=-1)).float()
        return depth.unsqueeze(0).unsqueeze(0)

    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        depth_np = depth.squeeze().cpu().numpy()
        depth_np = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        return depth_norm.astype(np.float32)
