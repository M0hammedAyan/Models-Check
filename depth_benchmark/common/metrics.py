"""
Metrics for evaluating depth estimation and pose accuracy.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class DepthMetrics:
    """Compute depth estimation metrics."""
    
    @staticmethod
    def mae(predicted: np.ndarray, reference: np.ndarray, 
           mask: Optional[np.ndarray] = None) -> float:
        """Mean Absolute Error."""
        if mask is None:
            mask = (reference > 0) & (~np.isnan(predicted))
        
        if mask.sum() == 0:
            return np.nan
        
        error = np.abs(predicted[mask] - reference[mask])
        return float(error.mean())
    
    @staticmethod
    def rmse(predicted: np.ndarray, reference: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """Root Mean Squared Error."""
        if mask is None:
            mask = (reference > 0) & (~np.isnan(predicted))
        
        if mask.sum() == 0:
            return np.nan
        
        error = (predicted[mask] - reference[mask]) ** 2
        return float(np.sqrt(error.mean()))
    
    @staticmethod
    def abs_rel_error(predicted: np.ndarray, reference: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> float:
        """Absolute Relative Error."""
        if mask is None:
            mask = (reference > 0) & (~np.isnan(predicted))
        
        if mask.sum() == 0:
            return np.nan
        
        rel_error = np.abs(predicted[mask] - reference[mask]) / reference[mask]
        return float(rel_error.mean())
    
    @staticmethod
    def sq_rel_error(predicted: np.ndarray, reference: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> float:
        """Squared Relative Error."""
        if mask is None:
            mask = (reference > 0) & (~np.isnan(predicted))
        
        if mask.sum() == 0:
            return np.nan
        
        rel_error = ((predicted[mask] - reference[mask]) ** 2) / reference[mask]
        return float(rel_error.mean())
    
    @staticmethod
    def silog_error(predicted: np.ndarray, reference: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> float:
        """Scale-Invariant Log Error."""
        if mask is None:
            mask = (reference > 0) & (~np.isnan(predicted))
        
        if mask.sum() == 0:
            return np.nan
        
        log_pred = np.log(predicted[mask] + 1e-8)
        log_ref = np.log(reference[mask] + 1e-8)
        
        diff = log_pred - log_ref
        mean_diff = diff.mean()
        
        silog = np.sqrt((diff ** 2).mean() - mean_diff ** 2)
        return float(silog)
    
    @staticmethod
    def delta_accuracy(predicted: np.ndarray, reference: np.ndarray,
                      thresholds: list = None,
                      mask: Optional[np.ndarray] = None) -> dict:
        """
        Delta accuracy at multiple thresholds.
        
        Args:
            predicted: Predicted depth
            reference: Reference depth
            thresholds: List of delta values [1.25, 1.25^2, 1.25^3]
            mask: Validity mask
            
        Returns:
            Dictionary with accuracy for each threshold
        """
        if thresholds is None:
            thresholds = [1.25, 1.25**2, 1.25**3]
        
        if mask is None:
            mask = (reference > 0) & (~np.isnan(predicted))
        
        if mask.sum() == 0:
            return {f'delta_{t}': np.nan for t in thresholds}
        
        pred_valid = predicted[mask]
        ref_valid = reference[mask]
        
        results = {}
        for threshold in thresholds:
            max_ratio = np.maximum(pred_valid / ref_valid, ref_valid / pred_valid)
            accuracy = (max_ratio < threshold).sum() / len(max_ratio)
            results[f'delta_{threshold:.4f}'] = float(accuracy)
        
        return results
    
    @staticmethod
    def correlation(predicted: np.ndarray, reference: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> float:
        """Pearson correlation coefficient."""
        if mask is None:
            mask = (reference > 0) & (~np.isnan(predicted))
        
        if mask.sum() < 2:
            return np.nan
        
        corr, _ = pearsonr(predicted[mask], reference[mask])
        return float(corr)
    
    @staticmethod
    def compute_all_metrics(predicted: np.ndarray, reference: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> dict:
        """Compute all depth metrics."""
        metrics = {
            'mae': DepthMetrics.mae(predicted, reference, mask),
            'rmse': DepthMetrics.rmse(predicted, reference, mask),
            'abs_rel': DepthMetrics.abs_rel_error(predicted, reference, mask),
            'sq_rel': DepthMetrics.sq_rel_error(predicted, reference, mask),
            'silog': DepthMetrics.silog_error(predicted, reference, mask),
            'correlation': DepthMetrics.correlation(predicted, reference, mask),
        }
        
        # Add delta accuracies
        delta_metrics = DepthMetrics.delta_accuracy(predicted, reference, mask=mask)
        metrics.update(delta_metrics)
        
        return metrics


class TemporalMetrics:
    """Compute temporal consistency metrics."""
    
    @staticmethod
    def temporal_consistency(depth_sequence: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute frame-to-frame depth consistency.
        
        Args:
            depth_sequence: Sequence of depth maps (T x H x W)
            mask: Validity mask
            
        Returns:
            Mean squared temporal gradient
        """
        if len(depth_sequence) < 2:
            return np.nan
        
        temporal_diffs = []
        
        for t in range(len(depth_sequence) - 1):
            curr = depth_sequence[t]
            next_frame = depth_sequence[t + 1]
            
            if mask is not None:
                valid = mask[t] & mask[t + 1]
            else:
                valid = (curr > 0) & (next_frame > 0)
            
            if valid.sum() > 0:
                diff = (curr[valid] - next_frame[valid]) ** 2
                temporal_diffs.append(diff.mean())
        
        if len(temporal_diffs) == 0:
            return np.nan
        
        return float(np.mean(temporal_diffs))
    
    @staticmethod
    def flickering_score(depth_sequence: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> float:
        """
        Compute flickering score (higher = more flicker).
        
        Based on variance of temporal differences.
        """
        if len(depth_sequence) < 3:
            return np.nan
        
        temporal_diffs = []
        
        for t in range(len(depth_sequence) - 1):
            curr = depth_sequence[t]
            next_frame = depth_sequence[t + 1]
            
            if mask is not None:
                valid = mask[t] & mask[t + 1]
            else:
                valid = (curr > 0) & (next_frame > 0)
            
            if valid.sum() > 0:
                diff = np.abs(curr[valid] - next_frame[valid])
                temporal_diffs.append(diff)
        
        if len(temporal_diffs) < 2:
            return np.nan
        
        temporal_diffs = np.concatenate(temporal_diffs)
        return float(temporal_diffs.std())
    
    @staticmethod
    def motion_stability(depth_sequence: np.ndarray,
                       motion_mask: Optional[np.ndarray] = None) -> float:
        """
        Compute motion stability.
        
        Args:
            depth_sequence: Sequence of depth maps (T x H x W)
            motion_mask: Mask of moving regions (T x H x W)
            
        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(depth_sequence) < 2:
            return np.nan
        
        consistency_scores = []
        
        for t in range(len(depth_sequence) - 1):
            curr = depth_sequence[t]
            next_frame = depth_sequence[t + 1]
            
            if motion_mask is not None:
                static_mask = (~motion_mask[t]) & (~motion_mask[t + 1])
            else:
                static_mask = (curr > 0) & (next_frame > 0)
            
            if static_mask.sum() > 0:
                diff = np.abs(curr[static_mask] - next_frame[static_mask])
                # Higher score if difference is smaller
                score = np.exp(-diff.mean())
                consistency_scores.append(score)
        
        if len(consistency_scores) == 0:
            return np.nan
        
        return float(np.mean(consistency_scores))


class PoseMetrics:
    """Compute pose estimation metrics."""
    
    @staticmethod
    def joint_depth_error(predicted_joints_3d: np.ndarray,
                         reference_joints_3d: np.ndarray,
                         valid_mask: Optional[np.ndarray] = None) -> dict:
        """
        Compute per-joint depth error.
        
        Args:
            predicted_joints_3d: Predicted joint positions (N_joints x 3)
            reference_joints_3d: Reference joint positions (N_joints x 3)
            valid_mask: Validity mask for joints
            
        Returns:
            Dictionary with errors per joint
        """
        if valid_mask is None:
            valid_mask = np.ones(len(predicted_joints_3d), dtype=bool)
        
        errors = {}
        joint_names = [f"joint_{i}" for i in range(len(predicted_joints_3d))]
        
        for i, joint_name in enumerate(joint_names):
            if valid_mask[i]:
                error = np.linalg.norm(predicted_joints_3d[i] - reference_joints_3d[i])
                errors[joint_name] = float(error)
            else:
                errors[joint_name] = np.nan
        
        errors['mean'] = float(np.nanmean(list(errors.values())))
        return errors
    
    @staticmethod
    def skeleton_stability(skeleton_sequence: np.ndarray) -> float:
        """
        Compute skeleton stability over time.
        
        Args:
            skeleton_sequence: Skeleton positions over time (T x N_joints x 3)
            
        Returns:
            Stability score
        """
        if len(skeleton_sequence) < 2:
            return np.nan
        
        temporal_diffs = []
        
        for t in range(len(skeleton_sequence) - 1):
            diff = np.linalg.norm(
                skeleton_sequence[t] - skeleton_sequence[t + 1],
                axis=1
            )
            temporal_diffs.append(diff.mean())
        
        return float(np.mean(temporal_diffs))
    
    @staticmethod
    def pose_consistency(pose_sequence: np.ndarray,
                        reference_pose: Optional[np.ndarray] = None) -> float:
        """
        Compute pose consistency.
        
        Args:
            pose_sequence: Pose sequence (T x N_joints x 3)
            reference_pose: Optional reference pose
            
        Returns:
            Consistency score
        """
        if reference_pose is None:
            reference_pose = pose_sequence.mean(axis=0)
        
        distances = []
        
        for pose in pose_sequence:
            dist = np.linalg.norm(pose - reference_pose)
            distances.append(dist)
        
        # Lower variance = higher consistency
        consistency = np.exp(-np.var(distances))
        return float(consistency)


class ErrorMap:
    """Generate error visualizations."""
    
    @staticmethod
    def compute_error_map(predicted: np.ndarray, reference: np.ndarray,
                         error_type: str = 'abs') -> np.ndarray:
        """
        Compute error map.
        
        Args:
            predicted: Predicted depth
            reference: Reference depth
            error_type: 'abs', 'rel', 'log'
            
        Returns:
            Error map
        """
        mask = reference > 0
        
        if error_type == 'abs':
            error = np.abs(predicted[mask] - reference[mask])
        elif error_type == 'rel':
            error = np.abs(predicted[mask] - reference[mask]) / (reference[mask] + 1e-8)
        elif error_type == 'log':
            error = np.abs(np.log(predicted[mask] + 1e-8) - np.log(reference[mask] + 1e-8))
        else:
            error = np.abs(predicted[mask] - reference[mask])
        
        error_map = np.full_like(predicted, np.nan)
        error_map[mask] = error
        
        return error_map
    
    @staticmethod
    def get_error_statistics(error_map: np.ndarray) -> dict:
        """Get statistics from error map."""
        valid = error_map[~np.isnan(error_map)]
        
        if len(valid) == 0:
            return {}
        
        return {
            'min': float(valid.min()),
            'max': float(valid.max()),
            'mean': float(valid.mean()),
            'median': float(np.median(valid)),
            'std': float(valid.std()),
            'q95': float(np.percentile(valid, 95)),
        }
