# Usage Examples

Quick reference for common benchmark scenarios.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Model Selection](#model-selection)
3. [Advanced Scenarios](#advanced-scenarios)
4. [Data Processing](#data-processing)
5. [Output Analysis](#output-analysis)

## Basic Usage

### Run on Single Video (All Models)

```bash
python main.py --config config.yaml \
  --input-video test_data/Ncam/my_video.mp4
```

### Run on RealSense .bag File

```bash
python main.py --config config.yaml \
  --input-bag test_data/Realsense/my_recording.bag
```

### Quick Test (Single Model, Small Resolution)

```bash
# Edit config.yaml first:
# - Set resolution: [960, 540]
# - Set models: only {midas: {...}}

python main.py --config config.yaml --input-video test_data/Ncam/video.mp4
```

## Model Selection

### Run Specific Models Only

```bash
python main.py --config config.yaml \
  --input-video test_data/Ncam/video.mp4 \
  --models midas depth_anything dpt_large
```

### Benchmark MiDaS Family

```bash
python main.py --config config.yaml \
  --input-video test_data/Ncam/video.mp4 \
  --models midas dpt_large
```

### Compare Depth Anything Versions

```bash
python main.py --config config.yaml \
  --input-video test_data/Ncam/video.mp4 \
  --models depth_anything depth_anything_v2
```

## Advanced Scenarios

### Refined vs Non-Refined Comparison

```bash
# Run with refinement (default)
python main.py --config config.yaml \
  --input-video test_data/Ncam/video.mp4 \
  --output-dir outputs/refined

# Run without refinement
python main.py --config config.yaml \
  --input-video test_data/Ncam/video.mp4 \
  --output-dir outputs/non_refined \
  --skip-temporal-refinement

# Compare results with: notebooks/visualization.ipynb
```

### CPU-Only Benchmark (for Testing)

```bash
python main.py --config config.yaml \
  --input-video test_data/Ncam/small_video.mp4 \
  --device cpu \
  --models midas
```

### Skip Optional Components

```bash
# Skip pose estimation
python main.py --config config.yaml \
  --input-video test_data/Ncam/video.mp4 \
  --skip-pose

# Skip temporal refinement
python main.py --config config.yaml \
  --input-video test_data/Ncam/video.mp4 \
  --skip-temporal-refinement

# Skip both
python main.py --config config.yaml \
  --input-video test_data/Ncam/video.mp4 \
  --skip-pose \
  --skip-temporal-refinement
```

### High-Resolution Processing (4K)

```bash
# First, modify config.yaml:
# data:
#   resolution: [2160, 3840]  # 4K
# processing:
#   enable_tiling_4k: true
#   tile_size: 512

python main.py --config config.yaml \
  --input-video test_data/Ncam/4k_video.mp4 \
  --models depth_pro depth_anything_v2
```

### Batch Processing (Multiple Videos)

```bash
# Create batch script
for video in test_data/Ncam/*.mp4; do
  echo "Processing: $video"
  python main.py --config config.yaml \
    --input-video "$video" \
    --output-dir "outputs/$(basename $video .mp4)"
done
```

### Custom Configuration

```bash
# Create custom config
cp config.yaml config_custom.yaml

# Edit config_custom.yaml for your needs

# Run with custom config
python main.py --config config_custom.yaml \
  --input-video test_data/Ncam/video.mp4
```

## Data Processing

### Extract Frames Only (No Inference)

```python
from common import VideoLoader

with VideoLoader('test_data/Ncam/video.mp4', target_fps=30) as loader:
    for frame, idx, timestamp in loader.get_frames():
        print(f"Frame {idx}: {timestamp:.2f}s, Shape: {frame.shape}")
        # Process frame...
```

### Load RealSense .bag and Save Frames

```python
from common import RealSenseLoader, FrameExtractor

FrameExtractor.extract_bag_frames(
    'test_data/Realsense/recording.bag',
    'output_rgb/',
    'output_depth/',
    max_frames=500
)
```

### Compute Metrics Manually

```python
from common import DepthMetrics
import numpy as np

# Mock depths for example
predicted = np.random.rand(480, 640)
reference = np.random.rand(480, 640)

# Compute all metrics
metrics = DepthMetrics.compute_all_metrics(predicted, reference)
print(metrics)
```

## Output Analysis

### View Results in Jupyter

```bash
jupyter notebook notebooks/visualization.ipynb

# Or directly in VS Code:
# Open notebooks/visualization.ipynb
```

### Parse CSV Results

```python
import pandas as pd

df = pd.read_csv('outputs/metrics/model_comparison.csv')
print(df)

# Best model for each metric
print("\nBest Models:")
for col in ['mae_mean', 'rmse_mean', 'abs_rel_mean', 'silog_mean']:
    best_idx = df[col].idxmin()
    print(f"  {col}: {df.loc[best_idx, 'Model']} ({df.loc[best_idx, col]:.4f})")
```

### Load and Analyze JSON Results

```python
import json

with open('outputs/metrics/benchmark_results_results.json') as f:
    results = json.load(f)

# Print model names
print("Models evaluated:", [k for k in results.keys() if k != 'pose'])

# Print metrics for first model
model = list(results.keys())[0]
print(f"\nMetrics for {model}:")
print(json.dumps(results[model], indent=2))
```

### Generate Custom Reports

```python
from benchmark import CSVExporter, BenchmarkSummaryWriter
import json

# Load results
with open('outputs/metrics/benchmark_results_results.json') as f:
    results = json.load(f)

# Export CSV
csv_path = 'outputs/custom_results.csv'
CSVExporter.export_model_comparison(results, csv_path)

# Write text report
report_path = 'outputs/custom_report.txt'
BenchmarkSummaryWriter.write_text_report(results, report_path)

# Write markdown report
markdown_path = 'outputs/custom_report.md'
BenchmarkSummaryWriter.write_markdown_report(results, markdown_path)
```

## Programmatic Usage

### Full Pipeline in Python

```python
from common import BenchmarkConfig
from benchmark import BenchmarkPipeline, CSVExporter, ReportGenerator

# Load configuration
config = BenchmarkConfig.from_yaml('config.yaml')
config.data.video_path = 'test_data/Ncam/video.mp4'

# Create and run pipeline
pipeline = BenchmarkPipeline(config)
pipeline.load_models()

# Process video
results, rgb_frames = pipeline.process_video(config.data.video_path)

# Compute metrics for each model
all_results = {}
for model_name, depths in results['models'].items():
    metrics = pipeline.compute_metrics(depths['depths'], depths['depths'])
    all_results[model_name] = metrics

# Export results
CSVExporter.export_model_comparison(all_results, 'results.csv')
ReportGenerator('reports').generate_all_reports(all_results)

print("✓ Done!")
```

### Run Single Model Inference

```python
from models import ModelFactory
import cv2

# Create model
model = ModelFactory.create_model('midas', device='cuda', use_fp16=True)

# Load image
image = cv2.imread('test_data/Ncam/frame.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict depth
depth = model.predict(image_rgb)

print(f"Depth map shape: {depth.shape}")
print(f"Depth range: {depth.min():.4f} - {depth.max():.4f}")
```

### Compare Models Side-by-Side

```python
from models import ModelFactory
from common import DepthVisualizer
import cv2
import numpy as np

# Load image
image = cv2.imread('test_data/Ncam/frame.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run models
results = {}
for model_name in ['midas', 'depth_anything', 'dpt_large']:
    model = ModelFactory.create_model(model_name, device='cuda')
    results[model_name] = model.predict(image_rgb)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('RGB Image')

for idx, (name, depth) in enumerate(results.items(), 1):
    ax = axes.flat[idx]
    colored = DepthVisualizer.colorize_depth(depth)
    ax.imshow(colored)
    ax.set_title(f'{name}')

plt.tight_layout()
plt.savefig('model_comparison.png')
```

### Real-Time Depth Streaming

```python
from models import ModelFactory
from common import DepthVisualizer
import cv2

# Setup camera
cap = cv2.VideoCapture(0)

# Load model
model = ModelFactory.create_model('midas', device='cuda')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predict depth
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth = model.predict(frame_rgb)
    
    # Visualize
    depth_colored = DepthVisualizer.colorize_depth(depth)
    depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    
    # Display
    combined = cv2.hconcat([frame, depth_bgr])
    cv2.imshow('RGB | Depth', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Performance Optimization

### Profile Inference Speed

```python
from models import ModelFactory, TiledInference
import time
import numpy as np

# Create model
model = ModelFactory.create_model('midas', device='cuda')

# Benchmark
image = np.random.rand(1080, 1920, 3).astype(np.uint8)

start = time.time()
for _ in range(100):
    depth = model.predict(image)
elapsed = time.time() - start

fps = 100 / elapsed
print(f"Average FPS: {fps:.2f}")
```

### Memory-Efficient Processing

```python
# Process large video in chunks
from common import VideoLoader

with VideoLoader('large_video.mp4') as loader:
    for frame, idx, _ in loader.get_frames():
        # Process one frame at a time
        # Don't load entire video into memory
        pass
```

---

**Last Updated**: January 2024
