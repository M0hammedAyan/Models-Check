"""
Video loading and frame extraction from MP4, AVI, and RealSense .bag files.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Generator
import os

logger = logging.getLogger(__name__)


class VideoLoader:
    """Load frames from video files (MP4, AVI, etc.)."""
    
    def __init__(self, video_path: str, target_fps: Optional[int] = None):
        """
        Initialize video loader.
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS (will downsample if original > target)
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.target_fps = target_fps or self.fps
        self.frame_skip = max(1, int(self.fps / self.target_fps))
        self.actual_fps = self.fps / self.frame_skip
        
        logger.info(f"Loaded video: {video_path}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps} -> {self.actual_fps}")
        logger.info(f"  Total frames: {self.frame_count}")
    
    def __len__(self) -> int:
        """Get total number of frames to be extracted."""
        return self.frame_count // self.frame_skip
    
    def get_frames(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Generator:
        """
        Yield frames from video.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None = all)
            
        Yields:
            (frame, frame_index, timestamp)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame * self.frame_skip)
        
        if end_frame is None:
            end_frame = self.frame_count
        
        frame_index = start_frame
        
        while frame_index < end_frame:
            ret = True
            for _ in range(self.frame_skip):
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_index / self.actual_fps
            
            yield frame, frame_index, timestamp
            frame_index += 1
    
    def get_frame(self, frame_index: int) -> Tuple[np.ndarray, float]:
        """Get specific frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index * self.frame_skip)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"Cannot read frame {frame_index}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = frame_index / self.actual_fps
        
        return frame, timestamp
    
    def close(self):
        """Release video capture."""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class RealSenseLoader:
    """Load frames from RealSense .bag recordings."""
    
    def __init__(self, bag_path: str, target_fps: Optional[int] = None, 
                 align_to_rgb: bool = True):
        """
        Initialize RealSense loader.
        
        Args:
            bag_path: Path to .bag file
            target_fps: Target FPS
            align_to_rgb: Align depth to RGB frame
        """
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError("pyrealsense2 not installed. Install with: pip install pyrealsense2")
        
        self.rs = rs
        self.bag_path = bag_path
        self.target_fps = target_fps
        self.align_to_rgb = align_to_rgb
        
        # Create pipeline for playing .bag file
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_path, repeat_playback=False)
        
        # Configure streams
        config.enable_stream(rs.stream.color, rs.format.rgb8)
        config.enable_stream(rs.stream.depth, rs.format.z16)
        
        profile = self.pipeline.start(config)
        
        # Get stream info
        color_stream = profile.get_stream(rs.stream.color)
        depth_stream = profile.get_stream(rs.stream.depth)
        
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        logger.info(f"Loaded RealSense .bag file: {bag_path}")
        logger.info(f"  Depth scale: {self.depth_scale}")
        logger.info(f"  Color intrinsics: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
        logger.info(f"  Depth intrinsics: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
        
        # Setup alignment if requested
        if align_to_rgb:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None
        
        self.frame_count = 0
        self.fps = target_fps or 30  # Default FPS for bag files
        
    def __len__(self) -> int:
        """Get approximate frame count (not exact for .bag files)."""
        return self.frame_count
    
    def get_frames(self, max_frames: Optional[int] = None) -> Generator:
        """
        Yield RGB-D frame pairs from .bag file.
        
        Args:
            max_frames: Maximum frames to extract
            
        Yields:
            (rgb_frame, depth_frame, frame_index, timestamp)
        """
        frame_index = 0
        
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                
                if self.align:
                    frames = self.align.process(frames)
                
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    break
                
                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
                
                timestamp = frames.get_timestamp() / 1000.0  # Convert to seconds
                
                yield rgb, depth, frame_index, timestamp
                
                frame_index += 1
                self.frame_count = frame_index
                
                if max_frames and frame_index >= max_frames:
                    break
        
        except Exception as e:
            logger.warning(f"Reached end of .bag file: {e}")
    
    def get_depth_scale(self) -> float:
        """Get depth scale factor."""
        return self.depth_scale
    
    def close(self):
        """Release pipeline."""
        self.pipeline.stop()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class FrameExtractor:
    """Extract and process frames from various sources."""
    
    @staticmethod
    def extract_video_frames(video_path: str, output_dir: str, 
                            target_fps: Optional[int] = None,
                            file_format: str = 'png') -> list:
        """
        Extract frames from video and save to disk.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            target_fps: Target FPS
            file_format: Output format (png or jpg)
            
        Returns:
            List of output frame paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        
        with VideoLoader(video_path, target_fps=target_fps) as loader:
            for frame, frame_idx, timestamp in loader.get_frames():
                frame_file = output_dir / f"frame_{frame_idx:06d}.{file_format}"
                
                # Convert RGB to BGR for cv2.imwrite
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_file), frame_bgr)
                frame_paths.append(str(frame_file))
                
                if (frame_idx + 1) % 100 == 0:
                    logger.info(f"Extracted {frame_idx + 1} frames")
        
        logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths
    
    @staticmethod
    def extract_bag_frames(bag_path: str, output_rgb_dir: str, output_depth_dir: str,
                          max_frames: Optional[int] = None) -> Tuple[list, list]:
        """
        Extract RGB and depth frames from RealSense .bag file.
        
        Args:
            bag_path: Path to .bag file
            output_rgb_dir: Output directory for RGB frames
            output_depth_dir: Output directory for depth frames
            max_frames: Maximum frames to extract
            
        Returns:
            (rgb_frame_paths, depth_frame_paths)
        """
        output_rgb_dir = Path(output_rgb_dir)
        output_depth_dir = Path(output_depth_dir)
        output_rgb_dir.mkdir(parents=True, exist_ok=True)
        output_depth_dir.mkdir(parents=True, exist_ok=True)
        
        rgb_paths = []
        depth_paths = []
        
        with RealSenseLoader(bag_path, align_to_rgb=True) as loader:
            for rgb, depth, frame_idx, timestamp in loader.get_frames(max_frames=max_frames):
                rgb_file = output_rgb_dir / f"frame_rgb_{frame_idx:06d}.png"
                depth_file = output_depth_dir / f"frame_depth_{frame_idx:06d}.png"
                
                # Save RGB
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(rgb_file), rgb_bgr)
                
                # Save depth as 16-bit PNG
                depth_uint16 = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(depth_file), depth_uint16)
                
                rgb_paths.append(str(rgb_file))
                depth_paths.append(str(depth_file))
                
                if (frame_idx + 1) % 100 == 0:
                    logger.info(f"Extracted {frame_idx + 1} RGB-D frame pairs")
        
        logger.info(f"Extracted {len(rgb_paths)} RGB-D pairs to {output_rgb_dir} and {output_depth_dir}")
        return rgb_paths, depth_paths


def get_media_info(path: str) -> dict:
    """Get information about media file."""
    info = {
        'path': path,
        'exists': os.path.exists(path),
        'size_mb': os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0,
    }
    
    if path.endswith('.bag'):
        info['type'] = 'realsense_bag'
    elif path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        info['type'] = 'video'
        try:
            with VideoLoader(path) as loader:
                info['fps'] = loader.fps
                info['frame_count'] = loader.frame_count
                info['resolution'] = (loader.width, loader.height)
                info['duration_sec'] = loader.frame_count / loader.fps
        except Exception as e:
            logger.error(f"Error reading video info: {e}")
    
    return info
