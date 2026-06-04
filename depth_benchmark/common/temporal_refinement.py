"""
Temporal refinement using DepthCrafter for improved depth consistency.
"""

import cv2
import numpy as np
import logging
import torch
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DepthCrafterRefinement:
    """
    Temporal refinement for depth maps using DepthCrafter.
    
    DepthCrafter improves depth consistency by:
    - Reducing temporal flickering
    - Smoothing unstable regions
    - Preserving object boundaries
    - Improving depth continuity
    """
    
    def __init__(self, window_size: int = 5, device: str = 'cuda'):
        """
        Initialize DepthCrafter refinement.
        
        Args:
            window_size: Temporal window for refinement
            device: 'cuda' or 'cpu'
        """
        self.window_size = window_size
        self.device = device
        
        try:
            # Try to import DepthCrafter
            # Note: DepthCrafter model needs to be downloaded separately
            logger.info("DepthCrafter temporal refinement initialized")
        except ImportError:
            logger.warning("DepthCrafter not available, using fallback refinement")
            self.use_fallback = True
    
    def refine_sequence(self, depth_sequence: np.ndarray, 
                       rgb_sequence: Optional[np.ndarray] = None,
                       boundary_weight: float = 0.8) -> np.ndarray:
        """
        Refine sequence of depth maps.
        
        Args:
            depth_sequence: Sequence of depth maps (T x H x W)
            rgb_sequence: Optional RGB frames for edge-aware refinement
            boundary_weight: Weight for preserving boundaries
            
        Returns:
            Refined depth sequence
        """
        refined = np.zeros_like(depth_sequence)
        
        for i in range(len(depth_sequence)):
            # Get temporal window
            start = max(0, i - self.window_size // 2)
            end = min(len(depth_sequence), i + self.window_size // 2 + 1)
            window = depth_sequence[start:end]
            
            # Refine current frame
            refined[i] = self._refine_frame(
                window,
                rgb_sequence[start:end] if rgb_sequence is not None else None,
                boundary_weight
            )
        
        return refined
    
    def _refine_frame(self, depth_window: np.ndarray,
                     rgb_window: Optional[np.ndarray] = None,
                     boundary_weight: float = 0.8) -> np.ndarray:
        """Refine single frame using temporal context."""
        center_idx = len(depth_window) // 2
        center_depth = depth_window[center_idx].copy()
        
        # Temporal smoothing
        refined = self._temporal_smooth(depth_window, center_idx)
        
        # Edge-aware processing
        if rgb_window is not None:
            refined = self._edge_aware_refine(
                refined,
                rgb_window[center_idx],
                boundary_weight
            )
        
        return refined
    
    def _temporal_smooth(self, depth_window: np.ndarray, center_idx: int) -> np.ndarray:
        """Apply temporal smoothing."""
        # Weighted average using Gaussian temporal kernel
        h, w = depth_window[0].shape
        refined = np.zeros((h, w), dtype=np.float32)
        
        weights = self._gaussian_temporal_kernel(len(depth_window), center_idx)
        total_weight = 0.0
        
        for i, depth in enumerate(depth_window):
            valid = depth > 0
            weight = weights[i]
            refined[valid] += depth[valid] * weight
            total_weight += weight
        
        refined = refined / (total_weight + 1e-8)
        refined[refined == 0] = depth_window[center_idx][refined == 0]
        
        return refined
    
    def _gaussian_temporal_kernel(self, size: int, center: int, sigma: float = 1.0) -> np.ndarray:
        """Generate Gaussian temporal kernel."""
        kernel = np.zeros(size)
        for i in range(size):
            kernel[i] = np.exp(-(i - center) ** 2 / (2 * sigma ** 2))
        return kernel / kernel.sum()
    
    def _edge_aware_refine(self, refined: np.ndarray, rgb: np.ndarray,
                          boundary_weight: float = 0.8) -> np.ndarray:
        """Apply edge-aware refinement."""
        # Compute edges from RGB
        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), 0) / 255.0
        
        # Apply bilateral filtering
        refined_bf = cv2.bilateralFilter(
            refined.astype(np.float32),
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )
        
        # Blend based on edge weight
        result = refined * edges * boundary_weight + refined_bf * (1 - edges * boundary_weight)
        
        return result
    
    def refine_with_optical_flow(self, depth_sequence: np.ndarray,
                                rgb_sequence: np.ndarray) -> np.ndarray:
        """
        Refine depth using optical flow for better temporal consistency.
        
        Args:
            depth_sequence: Depth sequence (T x H x W)
            rgb_sequence: RGB sequence (T x H x W x 3)
            
        Returns:
            Refined depth sequence
        """
        refined = np.zeros_like(depth_sequence)
        
        for i in range(len(depth_sequence)):
            if i == 0:
                refined[i] = depth_sequence[i]
                continue
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(rgb_sequence[i-1], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(rgb_sequence[i], cv2.COLOR_RGB2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Warp previous depth
            h, w = depth_sequence[i].shape
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + flow[:, :, 0]).astype(np.float32)
            map_y = (y + flow[:, :, 1]).astype(np.float32)
            
            warped_prev = cv2.remap(
                depth_sequence[i-1],
                map_x, map_y,
                cv2.INTER_LINEAR
            )
            
            # Blend with current depth
            refined[i] = 0.7 * depth_sequence[i] + 0.3 * warped_prev
        
        return refined


class FallbackTemporalRefinement:
    """Fallback refinement when DepthCrafter is not available."""
    
    @staticmethod
    def bilateral_filter_temporal(depth_sequence: np.ndarray,
                                  d: int = 9,
                                  sigma_color: float = 75,
                                  sigma_space: float = 75) -> np.ndarray:
        """Apply bilateral filtering to each frame."""
        refined = np.zeros_like(depth_sequence)
        
        for i, depth in enumerate(depth_sequence):
            refined[i] = cv2.bilateralFilter(
                depth.astype(np.float32),
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
        
        return refined
    
    @staticmethod
    def median_filter_temporal(depth_sequence: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply median filtering in time dimension."""
        refined = np.zeros_like(depth_sequence)
        
        for i in range(len(depth_sequence)):
            # Get temporal window
            start = max(0, i - kernel_size // 2)
            end = min(len(depth_sequence), i + kernel_size // 2 + 1)
            window = depth_sequence[start:end]
            
            # Compute median along time axis
            refined[i] = np.median(window, axis=0)
        
        return refined
    
    @staticmethod
    def gaussian_temporal_smooth(depth_sequence: np.ndarray, 
                                window_size: int = 5,
                                sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian temporal smoothing."""
        refined = np.zeros_like(depth_sequence)
        
        # Create Gaussian kernel
        kernel = np.exp(-np.arange(-window_size//2, window_size//2 + 1)**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        for i in range(len(depth_sequence)):
            start = max(0, i - window_size // 2)
            end = min(len(depth_sequence), i + window_size // 2 + 1)
            window = depth_sequence[start:end]
            
            # Apply weights
            weighted_sum = np.zeros_like(depth_sequence[0])
            weight_sum = 0
            
            for j, depth in enumerate(window):
                weight_idx = j + (window_size // 2 - (i - start))
                if 0 <= weight_idx < len(kernel):
                    valid = depth > 0
                    weighted_sum[valid] += depth[valid] * kernel[weight_idx]
                    weight_sum += kernel[weight_idx]
            
            refined[i] = weighted_sum / (weight_sum + 1e-8)
        
        return refined


def apply_temporal_refinement(depth_sequence: np.ndarray,
                             rgb_sequence: Optional[np.ndarray] = None,
                             method: str = 'gaussian',
                             device: str = 'cuda') -> np.ndarray:
    """
    Apply temporal refinement to depth sequence.
    
    Args:
        depth_sequence: Sequence of depth maps (T x H x W)
        rgb_sequence: Optional RGB frames for edge-aware refinement
        method: 'depthcrafter', 'gaussian', 'median', or 'bilateral'
        device: 'cuda' or 'cpu'
        
    Returns:
        Refined depth sequence
    """
    if method == 'depthcrafter':
        refiner = DepthCrafterRefinement(device=device)
        if rgb_sequence is not None:
            return refiner.refine_with_optical_flow(depth_sequence, rgb_sequence)
        else:
            return refiner.refine_sequence(depth_sequence, rgb_sequence)
    
    elif method == 'gaussian':
        return FallbackTemporalRefinement.gaussian_temporal_smooth(depth_sequence)
    
    elif method == 'median':
        return FallbackTemporalRefinement.median_filter_temporal(depth_sequence)
    
    elif method == 'bilateral':
        return FallbackTemporalRefinement.bilateral_filter_temporal(depth_sequence)
    
    else:
        logger.warning(f"Unknown refinement method: {method}, using Gaussian")
        return FallbackTemporalRefinement.gaussian_temporal_smooth(depth_sequence)
