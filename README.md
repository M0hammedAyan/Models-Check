# Depth Estimation Benchmark: Monocular vs RealSense

A production-grade benchmarking framework for comparing monocular depth estimation models against Intel RealSense depth sensing, with pose-aware analysis and temporal refinement.

## Features

✓ **8 State-of-the-Art Depth Models**
- MiDaS v3
- Depth Anything v1
- Depth Anything v2
- ZoeDepth
- DPT-Large
- Depth Pro
- LeReS
- BTS

✓ **Advanced Processing**
- DepthCrafter temporal refinement for all models
- Mixed precision (FP16) inference
- Tiled inference for 4K resolution
- CUDA acceleration with efficient memory handling

✓ **RealSense Integration**
- .bag file support with synchronized RGB-D extraction
- Depth alignment and scale matching
- Reference comparison metrics

✓ **Comprehensive Metrics**
- Depth: MAE, RMSE, Abs Rel, SILog, Delta Accuracy
- Temporal: Consistency, Flickering, Motion Stability
- Pose: Joint depth error, skeleton stability, temporal consistency

✓ **Pose Estimation**
- MediaPipe Pose detection
- Depth-aware skeleton visualization
- Body-part depth consistency analysis
- Joint-wise depth comparison

✓ **Visualizations**
- Side-by-side RGB/depth/error comparisons
- Depth heatmaps and error maps
- Pose overlays with depth coloring
- Temporal consistency graphs

## Project Structure

```
depth_benchmark/
├── common/                          # Shared utilities
│   ├── config.py                   # Configuration management
│   ├── utils.py                    # General utilities
│   ├── video_loader.py             # Video/RealSense .bag loading
│   ├── frame_extractor.py          # Frame extraction
│   ├── depth_alignment.py          # Depth alignment
│   ├── metrics.py                  # Metric computation
│   ├── pose_estimation.py          # MediaPipe integration
│   ├── temporal_refinement.py      # DepthCrafter refinement
│   └── visualization.py            # Visualization utilities
│
├── models/                          # Depth model runners
│   ├── base_model.py               # Base model class
│   ├── midas_runner.py
│   ├── depth_anything_runner.py
│   ├── depth_anything_v2_runner.py
│   ├── zoedepth_runner.py
│   ├── dpt_large_runner.py
│   ├── depth_pro_runner.py
│   ├── leres_runner.py
│   └── bts_runner.py
│
├── benchmark/                      # Benchmarking pipeline
│   ├── benchmark_pipeline.py       # Main pipeline
│   ├── compare_models.py           # Model comparison utilities
│   ├── generate_csv.py             # CSV export
│   └── generate_report.py          # Report generation
│
├── notebooks/                      # Jupyter notebooks
│   ├── evaluation.ipynb            # Full evaluation pipeline
│   └── visualization.ipynb         # Results visualization
│
├── outputs/                        # Results (auto-created)
│   ├── raw_depth/
│   ├── refined_depth/
│   ├── metrics/
│   ├── visualizations/
│   └── pose_results/
│
├── config.yaml                     # Main configuration
├── main.py                         # Entry point
└── requirements.txt                # Dependencies
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/M0hammedAyan/Models-Check.git
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Install Model-Specific Dependencies

```bash
# Depth Anything
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -e .
cd ..

# Depth Anything V2
pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git

# ZoeDepth
pip install git+https://github.com/isl-org/ZoeDepth.git

# Depth Pro (if available)
pip install git+https://github.com/apple/depth-pro.git

# RealSense (for .bag support)
pip install pyrealsense2
```

### 5. Optional: Install Optional Packages

```bash
# FLOPs calculation
pip install fvcore

# Development tools
pip install black pylint pytest
```

## Quick Start

### 1. Prepare Data

Organize test data in the following structure:

```
test_data/
├── Ncam/
│   ├── video1.mp4
│   └── video2.mp4
└── Realsense/
    ├── recording1.bag
    └── recording2.bag
```

### 2. Update Configuration

Edit `config.yaml`:

```yaml
data:
  video_path: ../test_data/Ncam/video.mp4  # or set via CLI
  bag_path: ../test_data/Realsense/recording.bag
  
output_dir: ./outputs
device: cuda
```

### 3. Run Benchmark

```bash
# Full benchmark with all models
python main.py --config config.yaml --input-video ../test_data/Ncam/video.mp4

# With RealSense .bag
python main.py --config config.yaml --input-bag ../test_data/Realsense/recording.bag

# Specific models only
python main.py --config config.yaml --models midas depth_anything dpt_large

# On CPU
python main.py --config config.yaml --device cpu

# Skip pose estimation
python main.py --config config.yaml --skip-pose

# Skip temporal refinement
python main.py --config config.yaml --skip-temporal-refinement
```

### Raspberry Pi 5 Safe Mode

On Raspberry Pi 5, run one model at a time with CPU-only inference and optional pose/refinement disabled:

```bash
python main.py --config config.yaml \
  --input-video ../test_data/Ncam/video.mp4 \
  --device cpu \
  --model midas \
  --skip-pose \
  --skip-temporal-refinement
```

You can repeat the same command later with `--model depth_anything`, `--model zoedepth`, or any other single model you want to test.

### 4. View Results

```bash
# Check outputs
ls -la outputs/

# Open visualization notebook
jupyter notebook notebooks/visualization.ipynb

# View generated reports
cat outputs/reports/model_rankings.txt
cat outputs/metrics/model_comparison.csv
```

## Usage Examples

### Example 1: Benchmark Single Video Against RealSense

```bash
python main.py \
  --config config.yaml \
  --input-video test_data/Ncam/person.mp4 \
  --input-bag test_data/Realsense/person.bag \
  --output-dir outputs/person_benchmark
```

### Example 2: Compare Refined vs Non-Refined

```bash
# With refinement (default)
python main.py --config config.yaml --input-video video.mp4 --output-dir outputs/refined

# Without refinement
python main.py --config config.yaml --input-video video.mp4 --output-dir outputs/non_refined --skip-temporal-refinement
```

### Example 3: Specific Model Comparison

```bash
python main.py \
  --config config.yaml \
  --input-video test_data/Ncam/4k_video.mp4 \
  --models midas depth_anything depth_anything_v2 depth_pro \
  --device cuda
```

### Example 4: Programmatic Usage (Python)

```python
from common import BenchmarkConfig
from benchmark import BenchmarkPipeline
from models import ModelFactory

# Load config
config = BenchmarkConfig.from_yaml('config.yaml')

# Create pipeline
pipeline = BenchmarkPipeline(config)
pipeline.load_models()

# Process video
results, rgb_frames = pipeline.process_video('test_data/Ncam/video.mp4')

# Compute metrics
metrics = pipeline.compute_metrics(
    results['models']['midas']['depths'],
    reference_depths
)

print(metrics)
```

## API Reference

### Core Classes

#### `BenchmarkConfig`
Configuration management with YAML/JSON support.

```python
from common import BenchmarkConfig

config = BenchmarkConfig.from_yaml('config.yaml')
config.models['midas'].use_cuda = True
config.to_yaml('config_modified.yaml')
```

#### `BenchmarkPipeline`
Main benchmarking pipeline.

```python
from benchmark import BenchmarkPipeline

pipeline = BenchmarkPipeline(config)
pipeline.load_models()
results, rgb_frames = pipeline.process_video(video_path)
```

#### `ModelFactory`
Create depth models on-the-fly.

```python
from models import ModelFactory

model = ModelFactory.create_model('midas', device='cuda', use_fp16=True)
depth = model.predict(rgb_image)

# List available models
print(ModelFactory.list_models())
```

#### `DepthMetrics`
Compute depth comparison metrics.

```python
from common import DepthMetrics

metrics = DepthMetrics.compute_all_metrics(predicted, reference)
print(metrics)
# {'mae': ..., 'rmse': ..., 'abs_rel': ..., 'silog': ..., 'delta_1.25': ...}
```

#### `MediaPipePose`
Pose estimation and analysis.

```python
from common import MediaPipePose, PoseProcessor

pose_detector = MediaPipePose()
pose_result = pose_detector.detect(rgb_image)

# Get joint positions
joints = PoseProcessor.extract_joint_positions(pose_result, [12, 14, 16])
```

### Utilities

```python
from common import (
    VideoLoader, RealSenseLoader,
    DepthAligner, DepthVisualizer,
    apply_temporal_refinement
)

# Load video
with VideoLoader(path) as loader:
    for frame, idx, timestamp in loader.get_frames():
        pass

# Load RealSense .bag
with RealSenseLoader(bag_path) as loader:
    for rgb, depth, idx, timestamp in loader.get_frames():
        pass

# Align depths
aligner = DepthAligner()
aligner.compute_alignment(predicted, reference)
aligned = aligner.align(predicted)

# Visualize
colored = DepthVisualizer.colorize_depth(depth, cmap='viridis')

# Temporal refinement
refined = apply_temporal_refinement(depth_sequence, method='gaussian')
```

## Output Structure

```
outputs/
├── raw_depth/              # Unrefined depth maps
├── refined_depth/          # Temporally refined depth maps
├── metrics/
│   ├── benchmark_results.json
│   ├── model_comparison.csv
│   └── summary.json
├── visualizations/
│   ├── metrics_comparison.png
│   ├── rgb_depth_0000.png
│   ├── comparison_video.mp4
│   └── pose_overlay_0000.png
├── pose_results/
│   ├── pose_joints.json
│   ├── skeleton_depth_analysis.csv
│   └── consistency_scores.json
└── reports/
    ├── model_rankings.txt
    └── summary.json
```

## Metrics Explained

### Depth Metrics

- **MAE**: Mean Absolute Error (meters)
- **RMSE**: Root Mean Squared Error (meters)
- **Abs Rel**: Absolute Relative Error (unitless ratio)
- **SILog**: Scale-Invariant Logarithmic Error
- **δ < 1.25**: Percentage of pixels within 1.25x of ground truth

### Temporal Metrics

- **Consistency**: Frame-to-frame depth stability
- **Flickering Score**: Temporal depth variation (lower = better)
- **Motion Stability**: Skeleton stability during motion

### Pose Metrics

- **Joint Error**: Per-joint depth discrepancy (meters)
- **Skeleton Stability**: Temporal skeleton motion smoothness
- **Temporal Consistency**: Pose prediction consistency over time

## Performance Notes

### GPU Memory Requirements

- MiDaS/DPT: ~2GB
- Depth Anything: ~3GB
- Depth Pro: ~4GB
- All models with batch inference: Add 1GB per additional batch

### Speed Benchmarks (RTX 3090, 1080p)

- MiDaS: ~45 FPS
- Depth Anything: ~30 FPS
- DPT-Large: ~25 FPS
- Depth Pro: ~15 FPS

### Optimization Tips

1. **Reduce Resolution**: Set `resolution: [960, 540]`
2. **Batch Processing**: Set `batch_size: 4` (if GPU memory allows)
3. **CPU Inference**: Use `device: cpu` for testing
4. **Skip Optional**: Use `--skip-pose` and `--skip-temporal-refinement`

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python main.py --config config.yaml --models midas

# Use CPU
python main.py --config config.yaml --device cpu

# Reduce resolution in config.yaml
```

### Missing Model Weights

Models download weights automatically on first run. Ensure internet connection.

```bash
# Pre-download weights
python -c "from models import ModelFactory; ModelFactory.create_model('midas')"
```

### RealSense .bag Issues

```bash
# Install pyrealsense2
pip install pyrealsense2

# Test with sample .bag
python -c "from common import RealSenseLoader; loader = RealSenseLoader('test.bag')"
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{depth_benchmark_2024,
  title={Depth Estimation Benchmark: Monocular vs RealSense},
  author={Research Team},
  year={2024},
  howpublished={\url{https://github.com/...}}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

## Support

For issues and questions:
- Check GitHub issues
- Review documentation
- Create detailed bug reports

## References

- [MiDaS](https://github.com/isl-org/MiDaS)
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
- [ZoeDepth](https://github.com/isl-org/ZoeDepth)
- [Depth Pro](https://github.com/apple/depth-pro)
- [MediaPipe](https://mediapipe.dev/)
- [RealSense SDK](https://github.com/IntelRealSense/librealsense)

---

**Last Updated**: January 2024  
**Maintainer**: Research Team  
**Status**: Production Ready
