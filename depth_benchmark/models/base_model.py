"""
Base model runner class and utilities for depth estimation models.
"""

import torch
import numpy as np
import cv2
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDepthModel(ABC):
    """Base class for depth estimation models."""
    
    def __init__(self, model_name: str, device: str = 'cuda',
                 use_fp16: bool = True, input_size: Optional[Tuple[int, int]] = None):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            device: 'cuda' or 'cpu'
            use_fp16: Use mixed precision
            input_size: Input size (height, width)
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and device == 'cuda'
        self.input_size = input_size or (384, 384)
        self.model = None
        self.processor = None
        
        logger.info(f"Initializing {model_name} on {device}")
    
    @abstractmethod
    def load_model(self):
        """Load model weights."""
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image."""
        pass
    
    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass
    
    @abstractmethod
    def postprocess(self, depth: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess depth map."""
        pass
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth for single image.
        
        Args:
            image: RGB image (H x W x 3)
            
        Returns:
            Depth map (H x W)
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        image_tensor = self.preprocess(image)
        
        # Forward
        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    depth = self.forward(image_tensor)
            else:
                depth = self.forward(image_tensor)
        
        # Postprocess
        depth_map = self.postprocess(depth, original_shape)
        
        return depth_map
    
    def predict_batch(self, images: list) -> list:
        """Predict depth for batch of images."""
        return [self.predict(img) for img in images]
    
    def get_model_size(self) -> Tuple[float, float]:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers()) / 1024 / 1024
        return param_size, buffer_size


class DepthNormalizer:
    """Normalize depth maps."""
    
    @staticmethod
    def normalize_depth(depth: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        if depth.max() <= depth.min():
            return np.zeros_like(depth)
        return (depth - depth.min()) / (depth.max() - depth.min())
    
    @staticmethod
    def denormalize_depth(depth: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Denormalize from [0, 1] to [min_val, max_val]."""
        return depth * (max_val - min_val) + min_val


class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    @staticmethod
    def resize_to_net_input(image: np.ndarray, target_size: Tuple[int, int],
                           keep_aspect_ratio: bool = True) -> np.ndarray:
        """Resize image to network input size."""
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        if keep_aspect_ratio:
            # Calculate scale to fit
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Pad
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            
            padded = cv2.copyMakeBorder(
                resized,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT
            )
            return padded
        else:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def normalize_image(image: np.ndarray,
                       mean: list = [0.485, 0.456, 0.406],
                       std: list = [0.229, 0.224, 0.225]) -> np.ndarray:
        """Normalize image with ImageNet statistics."""
        image = image.astype(np.float32) / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image


class TiledInference:
    """Perform tiled inference for high-resolution images."""
    
    def __init__(self, model: BaseDepthModel, tile_size: int = 512,
                 overlap: int = 64):
        """
        Initialize tiled inference.
        
        Args:
            model: Depth model
            tile_size: Size of each tile
            overlap: Overlap between tiles for blending
        """
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Perform tiled inference.
        
        Args:
            image: Input image
            
        Returns:
            Depth map
        """
        h, w = image.shape[:2]
        
        # If image is smaller than tile, just infer directly
        if h <= self.tile_size and w <= self.tile_size:
            return self.model.predict(image)
        
        depth_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        stride = self.tile_size - self.overlap
        
        # Generate tiles
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Get tile
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)
                
                tile = image[y_start:y_end, x_start:x_end]
                
                # Infer
                tile_depth = self.model.predict(tile)
                
                # Create weight mask for blending
                weight = self._create_weight_mask(tile_depth.shape)
                
                # Accumulate
                depth_map[y_start:y_end, x_start:x_end] += tile_depth * weight
                weight_map[y_start:y_end, x_start:x_end] += weight
        
        # Normalize by weights
        depth_map = depth_map / (weight_map + 1e-8)
        
        return depth_map
    
    def _create_weight_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create Gaussian weight mask for blending."""
        h, w = shape
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x, y)
        
        # Gaussian weight
        weight = np.exp(-(xx**2 + yy**2) / 0.5)
        return weight


def load_model_checkpoint(model: BaseDepthModel, checkpoint_path: str):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.model.load_state_dict(checkpoint['model'])
    else:
        model.model.load_state_dict(checkpoint)
    
    model.model.to(model.device)
    model.model.eval()
    logger.info("Checkpoint loaded successfully")
