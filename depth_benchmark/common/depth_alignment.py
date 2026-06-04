"""
Depth alignment and coordinate transformation utilities.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from scipy import optimize

logger = logging.getLogger(__name__)


class DepthAligner:
    """Align predicted depth with reference depth (RealSense)."""
    
    def __init__(self, align_method: str = 'scale_shift'):
        """
        Initialize depth aligner.
        
        Args:
            align_method: 'scale_shift' or 'affine'
        """
        self.align_method = align_method
        self.scale = 1.0
        self.shift = 0.0
    
    def compute_alignment(self, predicted: np.ndarray, reference: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> dict:
        """
        Compute alignment parameters.
        
        Args:
            predicted: Predicted depth map
            reference: Reference (RealSense) depth map
            mask: Validity mask (1 = valid, 0 = invalid)
            
        Returns:
            Dictionary with alignment parameters
        """
        if mask is None:
            mask = (reference > 0) & (~np.isnan(predicted)) & (~np.isnan(reference))
        else:
            mask = mask & (reference > 0) & (~np.isnan(predicted)) & (~np.isnan(reference))
        
        if mask.sum() < 10:
            logger.warning("Too few valid pixels for alignment")
            return {'scale': 1.0, 'shift': 0.0, 'valid': False}
        
        pred_flat = predicted[mask].flatten()
        ref_flat = reference[mask].flatten()
        
        if self.align_method == 'scale_shift':
            # Solve: ref = scale * pred + shift
            A = np.vstack([pred_flat, np.ones(len(pred_flat))]).T
            result = np.linalg.lstsq(A, ref_flat, rcond=None)
            params = result[0]
            
            self.scale = params[0]
            self.shift = params[1]
        
        elif self.align_method == 'affine':
            # Solve: ref = a * pred + b (same as scale_shift)
            A = np.vstack([pred_flat, np.ones(len(pred_flat))]).T
            result = np.linalg.lstsq(A, ref_flat, rcond=None)
            params = result[0]
            
            self.scale = params[0]
            self.shift = params[1]
        
        logger.info(f"Alignment computed: scale={self.scale:.4f}, shift={self.shift:.4f}")
        
        return {
            'scale': self.scale,
            'shift': self.shift,
            'valid': True
        }
    
    def align(self, depth: np.ndarray) -> np.ndarray:
        """Apply alignment to depth map."""
        return self.scale * depth + self.shift
    
    def align_batch(self, depths: np.ndarray) -> np.ndarray:
        """Apply alignment to batch of depth maps."""
        return self.scale * depths + self.shift


class DepthResizer:
    """Resize depth maps while preserving structure."""
    
    @staticmethod
    def resize_to_match(source: np.ndarray, target_shape: Tuple[int, int],
                       method: str = 'bicubic') -> np.ndarray:
        """
        Resize source to match target shape.
        
        Args:
            source: Source depth map
            target_shape: Target shape (height, width)
            method: 'nearest', 'bilinear', or 'bicubic'
            
        Returns:
            Resized depth map
        """
        if method == 'nearest':
            interp = cv2.INTER_NEAREST
        elif method == 'bilinear':
            interp = cv2.INTER_LINEAR
        elif method == 'bicubic':
            interp = cv2.INTER_CUBIC
        else:
            interp = cv2.INTER_LINEAR
        
        resized = cv2.resize(source, (target_shape[1], target_shape[0]), 
                            interpolation=interp)
        return resized
    
    @staticmethod
    def preserve_edges_resize(depth: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Resize while preserving edge sharpness.
        
        Uses edge-aware interpolation.
        """
        # Detect edges
        edges = cv2.Canny(depth.astype(np.uint8), 50, 150)
        
        # Resize depth
        new_height = int(depth.shape[0] * scale_factor)
        new_width = int(depth.shape[1] * scale_factor)
        resized_depth = cv2.resize(depth, (new_width, new_height), 
                                  interpolation=cv2.INTER_CUBIC)
        
        # Resize edges and sharpen
        resized_edges = cv2.resize(edges, (new_width, new_height), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Apply unsharp mask on edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sharpening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Sharpen edges
        mask = (resized_edges > 0).astype(np.float32)
        blurred = cv2.GaussianBlur(resized_depth, (5, 5), 0)
        sharpened = resized_depth + 0.5 * (resized_depth - blurred) * mask
        
        return sharpened


class CoordinateTransformer:
    """Transform coordinates between depth and RGB images."""
    
    def __init__(self, depth_intrinsics: dict, rgb_intrinsics: dict):
        """
        Initialize transformer.
        
        Args:
            depth_intrinsics: Camera intrinsics for depth camera
            rgb_intrinsics: Camera intrinsics for RGB camera
        """
        self.depth_K = self._build_camera_matrix(depth_intrinsics)
        self.rgb_K = self._build_camera_matrix(rgb_intrinsics)
    
    @staticmethod
    def _build_camera_matrix(intrinsics: dict) -> np.ndarray:
        """Build camera matrix from intrinsics."""
        K = np.array([
            [intrinsics.get('fx', 500), 0, intrinsics.get('ppx', 320)],
            [0, intrinsics.get('fy', 500), intrinsics.get('ppy', 240)],
            [0, 0, 1]
        ])
        return K
    
    def depth_to_3d(self, x: float, y: float, depth: float) -> np.ndarray:
        """
        Convert depth image pixel to 3D point.
        
        Args:
            x, y: Pixel coordinates
            depth: Depth value (meters)
            
        Returns:
            3D point [X, Y, Z]
        """
        K_inv = np.linalg.inv(self.depth_K)
        pixel = np.array([x, y, 1])
        point_3d = K_inv @ pixel * depth
        return point_3d
    
    def depth_map_to_3d(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert entire depth map to 3D point cloud.
        
        Args:
            depth: Depth map (H x W)
            
        Returns:
            Point cloud (H x W x 3)
        """
        h, w = depth.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        K_inv = np.linalg.inv(self.depth_K)
        
        # Normalize pixel coordinates
        x_norm = (x - self.depth_K[0, 2]) / self.depth_K[0, 0]
        y_norm = (y - self.depth_K[1, 2]) / self.depth_K[1, 1]
        
        # Scale by depth
        X = x_norm * depth
        Y = y_norm * depth
        Z = depth
        
        point_cloud = np.stack([X, Y, Z], axis=-1)
        return point_cloud
    
    def point_3d_to_depth(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        Project 3D point to depth image.
        
        Args:
            point_3d: 3D point [X, Y, Z]
            
        Returns:
            (x, y) pixel coordinates
        """
        pixel = self.depth_K @ point_3d
        x = pixel[0] / pixel[2]
        y = pixel[1] / pixel[2]
        return x, y


def align_depth_maps(predicted: np.ndarray, reference: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    method: str = 'scale_shift') -> np.ndarray:
    """
    Convenience function to align predicted depth with reference.
    
    Args:
        predicted: Predicted depth map
        reference: Reference depth map
        mask: Validity mask
        method: Alignment method
        
    Returns:
        Aligned depth map
    """
    aligner = DepthAligner(align_method=method)
    aligner.compute_alignment(predicted, reference, mask=mask)
    return aligner.align(predicted)


def compute_depth_statistics(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Compute statistics for depth map."""
    if mask is None:
        mask = depth > 0
    
    valid_depth = depth[mask]
    
    if len(valid_depth) == 0:
        return {'valid_pixels': 0}
    
    stats = {
        'valid_pixels': mask.sum(),
        'min': valid_depth.min(),
        'max': valid_depth.max(),
        'mean': valid_depth.mean(),
        'median': np.median(valid_depth),
        'std': valid_depth.std(),
        'q25': np.percentile(valid_depth, 25),
        'q75': np.percentile(valid_depth, 75),
    }
    
    return stats
