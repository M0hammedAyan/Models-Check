"""
Visualization utilities for depth maps, comparisons, and pose overlays.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import imageio

logger = logging.getLogger(__name__)


class DepthVisualizer:
    """Visualize depth maps and comparisons."""
    
    COLORMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet', 'hot']
    
    @staticmethod
    def colorize_depth(depth: np.ndarray, cmap: str = 'viridis',
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None) -> np.ndarray:
        """
        Convert depth map to RGB visualization.
        
        Args:
            depth: Depth map (H x W)
            cmap: Colormap name
            vmin, vmax: Value range for normalization
            
        Returns:
            RGB image (H x W x 3)
        """
        # Remove NaN and invalid values
        depth_clean = depth.copy()
        depth_clean[np.isnan(depth_clean)] = 0
        depth_clean[depth_clean < 0] = 0
        
        # Normalize
        if vmin is None:
            vmin = np.percentile(depth_clean[depth_clean > 0], 2)
        if vmax is None:
            vmax = np.percentile(depth_clean[depth_clean > 0], 98)
        
        if vmax <= vmin:
            vmax = vmin + 1e-5
        
        normalized = (depth_clean - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
        
        # Apply colormap
        cmap_obj = cm.get_cmap(cmap)
        colored = cmap_obj(normalized)[:, :, :3]
        
        return (colored * 255).astype(np.uint8)
    
    @staticmethod
    def create_comparison_image(rgb: np.ndarray, depth1: np.ndarray, depth2: np.ndarray,
                               depth3: Optional[np.ndarray] = None,
                               titles: List[str] = None,
                               cmap: str = 'viridis') -> np.ndarray:
        """
        Create side-by-side comparison image.
        
        Args:
            rgb: RGB image (H x W x 3)
            depth1: First depth map (H x W)
            depth2: Second depth map (H x W)
            depth3: Optional third depth map
            titles: Titles for each image
            cmap: Colormap
            
        Returns:
            Comparison image
        """
        if titles is None:
            titles = ['RGB', 'Depth 1', 'Depth 2']
            if depth3 is not None:
                titles.append('Depth 3')
        
        images = [rgb,
                 DepthVisualizer.colorize_depth(depth1, cmap),
                 DepthVisualizer.colorize_depth(depth2, cmap)]
        
        if depth3 is not None:
            images.append(DepthVisualizer.colorize_depth(depth3, cmap))
        
        # Resize all to same size
        h, w = rgb.shape[:2]
        resized = []
        for img in images:
            if img.shape[:2] != (h, w):
                resized.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR))
            else:
                resized.append(img)
        
        # Create grid
        num_images = len(resized)
        rows = 1
        cols = num_images
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (ax, img, title) in enumerate(zip(axes, resized, titles)):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return image
    
    @staticmethod
    def create_error_map_visualization(predicted: np.ndarray, reference: np.ndarray,
                                      error_type: str = 'abs',
                                      cmap: str = 'hot') -> np.ndarray:
        """
        Create error map visualization.
        
        Args:
            predicted: Predicted depth
            reference: Reference depth
            error_type: 'abs', 'rel', or 'log'
            cmap: Colormap
            
        Returns:
            Error visualization
        """
        mask = reference > 0
        
        if error_type == 'abs':
            error = np.abs(predicted - reference)
        elif error_type == 'rel':
            error = np.abs(predicted - reference) / (reference + 1e-8)
        elif error_type == 'log':
            error = np.abs(np.log(predicted + 1e-8) - np.log(reference + 1e-8))
        else:
            error = np.abs(predicted - reference)
        
        error[~mask] = 0
        
        return DepthVisualizer.colorize_depth(error, cmap=cmap)


class PoseVisualizer:
    """Visualize pose estimation results."""
    
    @staticmethod
    def draw_skeleton(image: np.ndarray, keypoints: List,
                     connections: List[Tuple[int, int]] = None,
                     confidence_threshold: float = 0.5,
                     thickness: int = 2,
                     radius: int = 5) -> np.ndarray:
        """
        Draw skeleton on image.
        
        Args:
            image: RGB image (H x W x 3)
            keypoints: List of Joint objects
            connections: List of (from, to) indices
            confidence_threshold: Confidence threshold for drawing
            thickness: Line thickness
            radius: Point radius
            
        Returns:
            Image with skeleton drawn
        """
        image = image.copy()
        
        if connections is None:
            # Default MediaPipe connections
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                (24, 26), (26, 28),
            ]
        
        # Draw connections
        for idx1, idx2 in connections:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                kp1 = keypoints[idx1]
                kp2 = keypoints[idx2]
                
                if kp1.confidence >= confidence_threshold and kp2.confidence >= confidence_threshold:
                    p1 = (int(kp1.x), int(kp1.y))
                    p2 = (int(kp2.x), int(kp2.y))
                    
                    # Check bounds
                    if (0 <= p1[0] < image.shape[1] and 0 <= p1[1] < image.shape[0] and
                        0 <= p2[0] < image.shape[1] and 0 <= p2[1] < image.shape[0]):
                        cv2.line(image, p1, p2, (0, 255, 0), thickness)
        
        # Draw keypoints
        for keypoint in keypoints:
            if keypoint.confidence >= confidence_threshold:
                p = (int(keypoint.x), int(keypoint.y))
                
                if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                    color = (0, 255, 0) if keypoint.confidence >= confidence_threshold else (0, 0, 255)
                    cv2.circle(image, p, radius, color, -1)
        
        return image
    
    @staticmethod
    def draw_depth_aware_skeleton(image: np.ndarray, depth: np.ndarray,
                                  keypoints: List,
                                  connections: List[Tuple[int, int]] = None,
                                  confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Draw skeleton with depth-based coloring.
        
        Args:
            image: RGB image
            depth: Depth map
            keypoints: Joint list
            connections: Skeleton connections
            confidence_threshold: Confidence threshold
            
        Returns:
            Image with depth-aware skeleton
        """
        image = image.copy()
        
        if connections is None:
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                (24, 26), (26, 28),
            ]
        
        # Get depth range
        valid_depth = depth[depth > 0]
        if len(valid_depth) == 0:
            return PoseVisualizer.draw_skeleton(image, keypoints, connections, confidence_threshold)
        
        min_depth = valid_depth.min()
        max_depth = valid_depth.max()
        
        # Draw connections with depth coloring
        for idx1, idx2 in connections:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                kp1 = keypoints[idx1]
                kp2 = keypoints[idx2]
                
                if kp1.confidence >= confidence_threshold and kp2.confidence >= confidence_threshold:
                    p1 = (int(kp1.x), int(kp1.y))
                    p2 = (int(kp2.x), int(kp2.y))
                    
                    if (0 <= p1[0] < image.shape[1] and 0 <= p1[1] < image.shape[0] and
                        0 <= p2[0] < image.shape[1] and 0 <= p2[1] < image.shape[0]):
                        
                        # Get depth values
                        d1 = depth[p1[1], p1[0]] if depth[p1[1], p1[0]] > 0 else 0
                        d2 = depth[p2[1], p2[0]] if depth[p2[1], p2[0]] > 0 else 0
                        
                        # Color based on average depth
                        avg_d = (d1 + d2) / 2
                        if avg_d > 0:
                            color_val = (avg_d - min_depth) / (max_depth - min_depth + 1e-8)
                            color = plt.cm.jet(color_val)[:3]
                            color = tuple(int(c * 255) for c in color)
                        else:
                            color = (0, 255, 0)
                        
                        cv2.line(image, p1, p2, color, 2)
        
        # Draw keypoints
        for keypoint in keypoints:
            if keypoint.confidence >= confidence_threshold:
                p = (int(keypoint.x), int(keypoint.y))
                
                if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                    d = depth[p[1], p[0]]
                    if d > 0:
                        color_val = (d - min_depth) / (max_depth - min_depth + 1e-8)
                        color = plt.cm.jet(color_val)[:3]
                        color = tuple(int(c * 255) for c in color)
                    else:
                        color = (0, 0, 255)
                    
                    cv2.circle(image, p, 5, color, -1)
        
        return image


class VideoComparison:
    """Create comparison videos."""
    
    @staticmethod
    def create_comparison_video(rgb_frames: List[np.ndarray],
                               depth_frames1: List[np.ndarray],
                               depth_frames2: List[np.ndarray],
                               output_path: str,
                               fps: float = 30,
                               depth_frames3: Optional[List[np.ndarray]] = None):
        """
        Create side-by-side comparison video.
        
        Args:
            rgb_frames: List of RGB frames
            depth_frames1: List of depth frames 1
            depth_frames2: List of depth frames 2
            output_path: Output video path
            fps: Output FPS
            depth_frames3: Optional third depth frames
        """
        logger.info(f"Creating comparison video: {output_path}")
        
        # Create writer
        h, w = rgb_frames[0].shape[:2]
        
        num_cols = 3 if depth_frames3 is None else 4
        out_w = w * num_cols
        out_h = h
        
        writer = imageio.get_writer(output_path, fps=fps)
        
        for i, (rgb, d1, d2) in enumerate(zip(rgb_frames, depth_frames1, depth_frames2)):
            # Colorize depths
            d1_color = DepthVisualizer.colorize_depth(d1)
            d2_color = DepthVisualizer.colorize_depth(d2)
            
            # Resize to match RGB
            if rgb.shape[:2] != d1_color.shape[:2]:
                d1_color = cv2.resize(d1_color, (w, h), interpolation=cv2.INTER_LINEAR)
                d2_color = cv2.resize(d2_color, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Concatenate
            row = np.hstack([rgb, d1_color, d2_color])
            
            if depth_frames3 is not None:
                d3_color = DepthVisualizer.colorize_depth(depth_frames3[i])
                if rgb.shape[:2] != d3_color.shape[:2]:
                    d3_color = cv2.resize(d3_color, (w, h), interpolation=cv2.INTER_LINEAR)
                row = np.hstack([row, d3_color])
            
            writer.append_data(row)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(rgb_frames)} frames")
        
        writer.close()
        logger.info(f"Comparison video saved: {output_path}")


def save_depth_visualization(depth: np.ndarray, output_path: str,
                             cmap: str = 'viridis'):
    """Save depth map visualization."""
    colored = DepthVisualizer.colorize_depth(depth, cmap=cmap)
    
    # Convert to BGR for OpenCV
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, colored_bgr)
    logger.info(f"Saved depth visualization: {output_path}")


def save_comparison_grid(rgb: np.ndarray, depths: List[np.ndarray],
                        output_path: str, titles: List[str] = None):
    """Save comparison grid image."""
    images = [rgb] + [DepthVisualizer.colorize_depth(d) for d in depths]
    
    if titles is None:
        titles = ['RGB'] + [f'Depth {i}' for i in range(len(depths))]
    
    # Create grid
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 4, 4))
    
    if len(images) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison grid: {output_path}")
