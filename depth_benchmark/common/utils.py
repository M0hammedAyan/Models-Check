"""
Utility functions for depth benchmarking pipeline.
"""

import os
import logging
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Tuple, Optional
import random

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def create_output_dirs(output_dir: str) -> dict:
    """Create output directory structure."""
    paths = {
        'root': output_dir,
        'raw_depth': os.path.join(output_dir, 'raw_depth'),
        'refined_depth': os.path.join(output_dir, 'refined_depth'),
        'metrics': os.path.join(output_dir, 'metrics'),
        'visualizations': os.path.join(output_dir, 'visualizations'),
        'pose_results': os.path.join(output_dir, 'pose_results'),
    }
    
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directories created at {output_dir}")
    return paths


def get_device() -> torch.device:
    """Get optimal device (CUDA if available, else CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device


def check_gpu_memory(min_gb: float = 2.0) -> bool:
    """Check if GPU has sufficient memory."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available")
        return False
    
    available_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU memory available: {available_gb:.2f} GB")
    
    if available_gb < min_gb:
        logger.warning(f"Insufficient GPU memory. Required: {min_gb}GB, Available: {available_gb:.2f}GB")
        return False
    return True


def normalize_depth(depth: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Normalize depth map to [min_val, max_val] range."""
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    
    if depth_min == depth_max:
        return np.full_like(depth, (min_val + max_val) / 2, dtype=np.float32)
    
    normalized = (depth - depth_min) / (depth_max - depth_min) * (max_val - min_val) + min_val
    return normalized.astype(np.float32)


def denormalize_depth(depth: np.ndarray, orig_min: float, orig_max: float) -> np.ndarray:
    """Denormalize depth from [0, 1] to original range."""
    return depth * (orig_max - orig_min) + orig_min


def resize_depth(depth: np.ndarray, size: Tuple[int, int], interpolation: str = 'bilinear') -> np.ndarray:
    """Resize depth map with proper interpolation."""
    h, w = depth.shape[-2:]
    
    if interpolation == 'bilinear':
        cv2_interp = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        cv2_interp = cv2.INTER_CUBIC
    elif interpolation == 'nearest':
        cv2_interp = cv2.INTER_NEAREST
    else:
        cv2_interp = cv2.INTER_LINEAR
    
    if len(depth.shape) == 3:  # Batch
        resized = np.zeros((depth.shape[0], size[0], size[1]), dtype=depth.dtype)
        for i in range(depth.shape[0]):
            resized[i] = cv2.resize(depth[i], (size[1], size[0]), interpolation=cv2_interp)
        return resized
    else:
        return cv2.resize(depth, (size[1], size[0]), interpolation=cv2_interp)


def pad_to_multiple(depth: np.ndarray, multiple: int = 32) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad depth map to multiple of given value."""
    h, w = depth.shape[-2:]
    h_pad = (multiple - h % multiple) % multiple
    w_pad = (multiple - w % multiple) % multiple
    
    if len(depth.shape) == 3:  # Batch
        padded = np.pad(depth, ((0, 0), (0, h_pad), (0, w_pad)), mode='edge')
    else:
        padded = np.pad(depth, ((0, h_pad), (0, w_pad)), mode='edge')
    
    return padded, (h_pad, w_pad)


def remove_padding(depth: np.ndarray, pad_info: Tuple[int, int]) -> np.ndarray:
    """Remove padding from depth map."""
    h_pad, w_pad = pad_info
    
    if h_pad == 0 and w_pad == 0:
        return depth
    
    if len(depth.shape) == 3:  # Batch
        return depth[:, :-h_pad, :-w_pad] if h_pad > 0 or w_pad > 0 else depth
    else:
        return depth[:-h_pad, :-w_pad] if h_pad > 0 or w_pad > 0 else depth


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array."""
    return tensor.detach().cpu().numpy()


def numpy_to_torch(array: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor."""
    tensor = torch.from_numpy(array).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_image_mean_std() -> Tuple[list, list]:
    """Get ImageNet normalization statistics."""
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image using ImageNet statistics."""
    mean, std = get_image_mean_std()
    image = image.astype(np.float32) / 255.0
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image."""
    mean, std = get_image_mean_std()
    for i in range(3):
        image[:, :, i] = image[:, :, i] * std[i] + mean[i]
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray, cmap: str = 'viridis') -> np.ndarray:
    """Convert depth map to colored visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    depth = normalize_depth(depth, 0.0, 1.0)
    
    cmap_obj = cm.get_cmap(cmap)
    depth_colored = cmap_obj(depth)[:, :, :3]
    
    return (depth_colored * 255).astype(np.uint8)


def save_depth_map(depth: np.ndarray, path: str, scale: float = 1000.0):
    """Save depth map as 16-bit PNG."""
    depth_uint16 = np.clip(depth * scale, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, depth_uint16)
    logger.info(f"Depth map saved to {path}")


def load_depth_map(path: str, scale: float = 1000.0) -> np.ndarray:
    """Load depth map from 16-bit PNG."""
    depth_uint16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth = depth_uint16.astype(np.float32) / scale
    return depth


def calculate_flops(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> float:
    """Calculate FLOPs for a model (requires fvcore)."""
    try:
        from fvcore.nn import FlopCounterMode
        dummy_input = torch.randn(1, *input_shape).cuda()
        
        with FlopCounterMode(model) as flops_counter:
            model(dummy_input)
        
        flops = flops_counter.total_flops
        logger.info(f"Model FLOPs: {flops / 1e9:.2f}G")
        return flops
    except ImportError:
        logger.warning("fvcore not installed, cannot calculate FLOPs")
        return 0.0


def measure_inference_time(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                          num_iterations: int = 100, warmup: int = 10) -> float:
    """Measure average inference time."""
    import time
    
    dummy_input = torch.randn(1, *input_shape).cuda()
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time = (time.time() - start_time) / num_iterations * 1000  # ms
    logger.info(f"Average inference time: {elapsed_time:.2f}ms")
    return elapsed_time
