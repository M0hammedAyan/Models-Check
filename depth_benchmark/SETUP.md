# Complete Setup Guide: Depth Estimation Benchmark

This guide covers **everything** needed to set up this project on a fresh PC from scratch. After following all steps, the only pending item is placing a video file and running the benchmark.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Prerequisites Installation](#2-prerequisites-installation)
3. [Clone This Project](#3-clone-this-project)
4. [Python Environment Setup](#4-python-environment-setup)
5. [Install Core Python Dependencies](#5-install-core-python-dependencies)
6. [Clone External Model Repositories](#6-clone-external-model-repositories)
7. [Install Model-Specific Dependencies](#7-install-model-specific-dependencies)
8. [Download Model Weights / Checkpoints](#8-download-model-weights--checkpoints)
9. [Directory Structure Verification](#9-directory-structure-verification)
10. [Test the Installation](#10-test-the-installation)
11. [Configuration Guide](#11-configuration-guide)
12. [Running the Pipeline](#12-running-the-pipeline)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux (Ubuntu 20.04+), Windows 10+, macOS 12+ | Ubuntu 22.04 |
| **Python** | 3.8 | 3.10 |
| **RAM** | 16 GB | 32 GB |
| **Disk Space** | 50 GB free | 100 GB free (for model weights + repos) |
| **GPU** | NVIDIA GPU with 6 GB VRAM | NVIDIA RTX 3090 / 4090 (24 GB) |
| **CUDA** | 11.8 | 12.1 |
| **Internet** | Required for first-time model weight downloads | Broadband |

---

## 2. Prerequisites Installation

### 2.1 CUDA & NVIDIA Driver (GPU systems only)

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not installed, download from:
# https://www.nvidia.com/download/index.aspx
# Or via package manager:
# Ubuntu/Debian
sudo apt update && sudo apt install nvidia-driver-545 nvidia-utils-545

# Install CUDA toolkit (if PyTorch doesn't find CUDA)
# https://developer.nvidia.com/cuda-downloads
# Ubuntu 22.04 example:
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### 2.2 System Packages

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y git wget curl build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 ffmpeg libhdf5-dev

# macOS (Homebrew)
brew install git wget cmake ffmpeg libomp

# Windows - Install via:
# - Git: https://git-scm.com/download/win
# - CMake: https://cmake.org/download/
# - FFmpeg: https://ffmpeg.org/download.html
# - Build Tools for Visual Studio: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
```

### 2.3 Verify Python

```bash
python3 --version   # Must be 3.8+
python3 -m pip --version
```

If Python is missing, install it:
```bash
# Ubuntu
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# macOS
brew install python@3.10

# Windows
# Download from https://www.python.org/downloads/
```

---

## 3. Clone This Project

```bash
# Choose a location on your machine
mkdir -p ~/projects
cd ~/projects

# Clone this repository (the benchmark framework)
git clone https://github.com/M0hammedAyan/Models-Check.git
cd Models-Check/depth_benchmark
```

---

## 4. Python Environment Setup

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate

# Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
# Windows (cmd):
# venv\Scripts\activate.bat

# Verify activation
which python   # Linux/macOS - should show inside venv/
where python   # Windows
```

---

## 5. Install Core Python Dependencies

```bash
# Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first (choose your CUDA version or CPU)
# CUDA 12.1 (RTX 30xx/40xx):
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older GPUs):
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU):
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install the core requirements
pip install -r requirements.txt
```

### Alternative: Full requirements list (if requirements.txt is unavailable)

```bash
pip install \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    timm>=0.9.0 \
    opencv-python>=4.8.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    scipy>=1.10.0 \
    pyyaml>=6.0 \
    tqdm>=4.66.0 \
    imageio>=2.31.0 \
    pillow>=10.0.0 \
    mediapipe>=0.10.0 \
    huggingface_hub>=0.19.0 \
    jupyter>=1.0.0 \
    ipython>=8.0.0 \
    pytest>=7.0.0
```

### Verify core installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import cv2; print(f'OpenCV: {cv2.__version__}')
import numpy; print(f'NumPy: {numpy.__version__}')
import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')
print('Core dependencies OK')
"
```

---

## 6. Clone External Model Repositories

Create the directory for external model repos and clone every repository the project depends on.

**Why**: The model runners (`models/*.py`) import from these cloned repos at runtime via `sys.path.insert`.

```bash
# Stay inside depth_benchmark/
mkdir -p upstream_repos
cd upstream_repos

# ── Core / Foundational Models ──────────────────────────

# Depth Anything v1 (for depth_anything runner)
git clone https://github.com/LiheYoung/Depth-Anything.git

# Depth Anything v2 (for depth_anything_v2 runner)
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

# ZoeDepth (for zoedepth runner)
git clone https://github.com/isl-org/ZoeDepth.git

# Apple Depth Pro (for depth_pro runner)
git clone https://github.com/apple/ml-depth-pro.git

# BTS (From Big to Small) (for bts runner)
git clone https://github.com/cleinc/bts.git

# AdaBins (for adabins runner)
git clone https://github.com/shariqfarooq123/AdaBins.git

# ── Metric Models ──────────────────────────────────────

# UniDepth (for unidepth runner)
git clone https://github.com/lpiccinelli-eth/UniDepth.git

# Metric3D (for metric3d runner)
git clone https://github.com/YvanYin/Metric3D.git

# IEBins (for iebins runner)
git clone https://github.com/ShuweiShao/IEBins.git

# ── Diffusion / Flow-Matching Models ────────────────────

# DepthFM (for depthfm runner) - flow matching depth
git clone https://github.com/CompVis/depth-fm.git

# Lotus (for lotus runner) - diffusion depth
git clone https://github.com/EnVision-Research/Lotus.git

# ECoDepth (for ecodepth runner)
git clone https://github.com/Aradhye2002/EcoDepth.git

# ── Tile / High-Resolution Models ───────────────────────

# PatchFusion (for patchfusion runner)
git clone https://github.com/zhyever/PatchFusion.git

# ── Toolbox-based Models ────────────────────────────────

# Monocular Depth Estimation Toolbox (for binsformer runner)
git clone https://github.com/ZJULearning/Monocular-Depth-Estimation-Toolbox.git

# ── Optional: LeReS / AnyDepth ──────────────────────────

# AnyDepth (fallback for leres runner)
git clone https://github.com/AIGeeksGroup/AnyDepth.git

# ── Optional: RealSense support ─────────────────────────

# librealsense (for pyrealsense2 with .bag files)
git clone https://github.com/IntelRealSense/librealsense.git

# Return to depth_benchmark/
cd ..
```

---

## 7. Install Model-Specific Dependencies

Some model repos need to be `pip install -e .`'d or have extra pip packages.

### 7.1 Install via pip (editable mode)

```bash
cd upstream_repos

# Depth Anything v1
cd Depth-Anything && pip install -e . && cd ..

# ZoeDepth
cd ZoeDepth && pip install -e . && cd ..

# Depth Pro
cd ml-depth-pro && pip install -e . && cd ..

# UniDepth
cd UniDepth && pip install -e . && cd ..

# DepthFM
cd depth-fm && pip install -e . && cd ..

# Lotus
cd Lotus && pip install -e . && cd ..

# ECoDepth
cd EcoDepth && pip install -e . && cd ..

cd ..
```

### 7.2 Install Depth Anything v2

```bash
# Option A: Editable from cloned repo
cd upstream_repos/Depth-Anything-V2 && pip install -e . && cd ../..

# Option B: Direct pip install (if clone fails)
# pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git
```

### 7.3 Install BinsFormer dependencies (MMSegmentation toolbox)

```bash
# Install mmcv-full and mmsegmentation
pip install -U openmim
mim install mmcv-full
pip install mmsegmentation

# Then the toolbox itself
cd upstream_repos/Monocular-Depth-Estimation-Toolbox
pip install -e .
cd ../..
```

### 7.4 Install RealSense support (optional, for .bag files)

```bash
# Python wrapper
pip install pyrealsense2

# On Windows, if pyrealsense2 wheel fails, download from:
# https://github.com/IntelRealSense/librealsense/releases
# Then: pip install pyrealsense2-<version>.whl
```

### 7.5 Install HuggingFace Hub (for model weight downloads)

```bash
pip install huggingface_hub
# Login (optional, needed for gated models)
huggingface-cli login
```

---

## 8. Download Model Weights / Checkpoints

Most models auto-download weights on first run. Some require manual download.

### 8.1 Weights that auto-download (no action needed)

These models download weights to `~/.cache/torch/hub/checkpoints/` automatically on first use:

| Model | Source | Cache file |
|-------|--------|-----------|
| **MiDaS** | torch.hub / timm | Auto |
| **DPT-Large** | torch.hub / timm | Auto |
| **Depth Anything v1** | GitHub Releases | `depth_anything_vits14.pth`, `vitb14`, `vitl14` |
| **Depth Anything v2** | HuggingFace Hub | `depth_anything_v2_vits.pth`, `vitb`, `vitl` |
| **ZoeDepth** | GitHub Releases | `ZoeD_M12_NK.pt` / `ZoeD_M12_K.pt` |
| **Depth Pro** | Local checkpoint | Must be placed at `checkpoints/depth_pro.pt` (see 8.2) |
| **AdaBins** | GitHub Releases | `AdaBins_nyu-256-*.pth` / `AdaBins_kitti-256-*.pth` |
| **UniDepth** | HuggingFace Hub | Auto via `from_pretrained` |
| **DepthFM** | URL download | `depthfm-v1.ckpt` |
| **Lotus** | HuggingFace Hub | Auto via `from_pretrained` |
| **ECoDepth** | Auto-download | `weights_indoor.ckpt` / `weights_outdoor.ckpt` |
| **Metric3D** | torch.hub | Auto |

### 8.2 Manual checkpoint placement

Create the checkpoints directory:

```bash
mkdir -p checkpoints
```

**Depth Pro** checkpoint (required):
```bash
# Download from: https://github.com/apple/ml-depth-pro
# Place the .pt file at:
# checkpoints/depth_pro.pt

# Or download via script:
wget -P checkpoints/ https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
```

**IEBins** checkpoints (manual, Google Drive):
```
NYU-SwinL: https://drive.google.com/file/d/14Rn-vxvpXO2EXRaWqCPmh2JufvOurwtl
KITTI-SwinL: https://drive.google.com/file/d/1xaVLDq7zJ-C2GtFvABolSUtK7gzvNQNd

Place these in: ~/.cache/torch/hub/checkpoints/
```

**BinsFormer** checkpoints (manual, Google Drive):
```
NYU-SwinL: https://drive.google.com/file/d/1j1FmtXKSOD5e6HWBBd_3cwI2M11Jd_nB
NYU-SwinT: https://drive.google.com/file/d/1tcWx_BQBNJHpP5-RUWGWjpVRfeUiUMzJ

Place these in: ~/.cache/torch/hub/checkpoints/
```

### 8.3 Pre-download all auto weights (optional, saves time later)

```bash
python -c "
from models import ModelFactory

# Test-load each model (weights download on first call)
models_to_test = [
    'midas', 'dpt_large', 'depth_anything', 'depth_anything_v2',
    'zoedepth', 'adabins', 'unidepth',
]

for name in models_to_test:
    try:
        print(f'Pre-loading {name}...')
        model = ModelFactory.create_model(name, device='cpu', use_fp16=False)
        print(f'  OK')
    except Exception as e:
        print(f'  SKIP ({e})')
"
```

---

## 9. Directory Structure Verification

After all steps, your `depth_benchmark/` directory should look like this:

```
depth_benchmark/
├── common/                          # Shared utilities (9 files)
│   ├── __init__.py
│   ├── config.py
│   ├── depth_alignment.py
│   ├── metrics.py
│   ├── pose_estimation.py
│   ├── temporal_refinement.py
│   ├── utils.py
│   ├── video_loader.py
│   └── visualization.py
├── models/                          # Model runners (17 files + __init__)
│   ├── __init__.py
│   ├── base_model.py
│   ├── adabins_runner.py
│   ├── binsformer_runner.py
│   ├── bts_runner.py
│   ├── depth_anything_runner.py
│   ├── depth_anything_v2_runner.py
│   ├── depth_pro_runner.py
│   ├── depthfm_runner.py
│   ├── dpt_large_runner.py
│   ├── ecodepth_runner.py
│   ├── iebins_runner.py
│   ├── leres_runner.py
│   ├── lotus_runner.py
│   ├── metric3d_runner.py
│   ├── midas_runner.py
│   ├── patchfusion_runner.py
│   ├── unidepth_runner.py
│   └── zoedepth_runner.py
├── benchmark/                       # Benchmark pipeline (4 files)
│   ├── __init__.py
│   ├── benchmark_pipeline.py
│   ├── compare_models.py
│   ├── generate_csv.py
│   └── generate_report.py
├── notebooks/                       # Jupyter notebooks
│   ├── evaluation.ipynb
│   └── visualization.ipynb
├── upstream_repos/                  # Cloned model repositories (17 dirs)
│   ├── AdaBins/
│   ├── AnyDepth/
│   ├── bts/
│   ├── Depth-Anything/
│   ├── Depth-Anything-V2/
│   ├── depth-fm/
│   ├── EcoDepth/
│   ├── IEBins/
│   ├── librealsense/
│   ├── Lotus/
│   ├── Metric3D/
│   ├── ml-depth-pro/
│   ├── Monocular-Depth-Estimation-Toolbox/
│   ├── PatchFusion/
│   ├── UniDepth/
│   └── ZoeDepth/
├── checkpoints/                     # Manually downloaded weights
│   └── depth_pro.pt                 # (optional, auto if not present)
├── outputs/                         # Auto-created on first run
│   ├── raw_depth/
│   ├── refined_depth/
│   ├── metrics/
│   ├── visualizations/
│   ├── pose_results/
│   └── reports/
├── main.py                          # CLI entry point
├── example_run.py                   # Python usage examples
├── config.yaml                      # Main configuration file
├── requirements.txt                 # Core pip dependencies
├── SETUP.md                         # This file
├── README.md                        # Documentation
├── INSTALLATION.md                  # Installation guide
├── USAGE_EXAMPLES.md                # Usage reference
└── venv/                            # Virtual environment
```

Verify with:
```bash
# List all nested dirs one level deep
ls -d */

# Check upstream_repos count (should be ~17)
ls upstream_repos/ | wc -l
```

---

## 10. Test the Installation

### 10.1 Test imports

```bash
python -c "
from common import BenchmarkConfig, VideoLoader, DepthMetrics, MediaPipePose
from models import ModelFactory
from benchmark import BenchmarkPipeline
print('All imports OK')
print(f'Available models: {ModelFactory.list_models()}')
"
```

### 10.2 Run end-to-end test with example video

```bash
# Check if the example video exists
ls -la upstream_repos/Depth-Anything/assets/examples_video/

# Run the test_e2e script (uses 2 models: midas + dpt_large)
python test_e2e.py

# Expected output:
# Processing video...
# Processed N frames
# midas: N depth maps, shape=(H, W)
# dpt_large: N depth maps, shape=(H, W)
# SUCCESS: End-to-end pipeline works!
```

### 10.3 Run quick benchmark on example video

```bash
python main.py --config config.yaml \
    --models midas dpt_large \
    --device cuda \
    --skip-pose \
    --skip-temporal-refinement \
    --output-dir outputs/quick_test

# Check results
ls outputs/quick_test/metrics/
cat outputs/quick_test/metrics/model_comparison.csv
```

---

## 11. Configuration Guide

Edit `config.yaml` to match your hardware and needs.

### 11.1 Key configuration options

```yaml
# ── Data ──────────────────────────────────────────────────
data:
  video_path: null          # Set this to your video file path
  bag_path: null            # Set this for RealSense .bag input
  output_fps: 1             # Frames per second to extract
                            #   1 = process 1 frame/sec (fast, good for 5-min videos)
                            #   30 = process all frames (slow, high accuracy)
  resolution: [540, 960]    # [height, width] for processing
                            #   [540, 960]   = 540p (fast)
                            #   [1080, 1920] = 1080p (balanced)
                            #   [2160, 3840] = 4K   (slow, needs tiling)

# ── Processing ────────────────────────────────────────────
processing:
  apply_temporal_refinement: false   # Set true for smoother depth (slower)
  temporal_window_size: 5
  batch_size: 1                      # Keep at 1 for RTX 3050 6GB
  enable_tiling_4k: false            # Set true only for 4K video
  tile_size: 512
  use_fp16: true                     # Enable FP16 for memory saving

# ── Pose ──────────────────────────────────────────────────
pose:
  enable_pose: false                 # Set true to analyze body pose
  pose_model: "mediapipe"
  min_confidence: 0.5

# ── Models ────────────────────────────────────────────────
# Uncomment only the models you want to run.
# Fewer models = faster processing.
models:
  midas:
    model_type: midas
    input_size: [384, 384]
    use_cuda: true
    mixed_precision: true
  dpt_large:
    model_type: dpt_large
    input_size: [384, 384]
    use_cuda: true
    mixed_precision: true
  depth_anything:
    model_type: depth_anything
    input_size: [518, 518]
    use_cuda: true
    mixed_precision: true
  depth_anything_v2:
    model_type: depth_anything_v2
    input_size: [518, 518]
    use_cuda: true
    mixed_precision: true
  zoedepth:
    model_type: zoedepth
    input_size: [384, 384]
    use_cuda: true
    mixed_precision: false    # ZoeDepth doesn't support FP16
  depth_pro:
    model_type: depth_pro
    input_size: [1024, 768]
    use_cuda: true
    mixed_precision: true
  # Add more models as needed (adabins, bts, unidepth, metric3d, depthfm,
  # lotus, iebins, ecodepth, binsformer, patchfusion)

# ── System ────────────────────────────────────────────────
output_dir: ./outputs
device: cuda                       # "cuda" or "cpu"
num_workers: 4
seed: 42
```

### 11.2 Recommended configurations by hardware

**RTX 3050 6GB (current config):**
```yaml
output_fps: 1
resolution: [540, 960]
apply_temporal_refinement: false
enable_pose: false
use_fp16: true
# Use 2-4 models maximum
```

**RTX 3090/4090 24GB:**
```yaml
output_fps: 30
resolution: [1080, 1920]
apply_temporal_refinement: true
enable_pose: true
use_fp16: true
# Can run all 12+ models
```

**CPU only:**
```yaml
output_fps: 1
resolution: [270, 480]
apply_temporal_refinement: false
enable_pose: false
use_fp16: false
device: cpu
# Use 1 model at a time (midas is fastest)
```

---

## 12. Running the Pipeline

### 12.1 Prepare your video

```bash
# Create test data directories
mkdir -p test_data/Ncam
mkdir -p test_data/Realsense

# Place your video file
# Option 1: Copy your video
cp /path/to/your/video.mp4 test_data/Ncam/

# Option 2: Use the built-in example video
# The default config already points to:
#   upstream_repos/Depth-Anything/assets/examples_video/davis_dolphins.mp4
```

### 12.2 Update config.yaml

Set `data.video_path` to your video file path.

### 12.3 Run the benchmark

```bash
# Activate environment (if not already)
source venv/bin/activate   # Linux/macOS
# .\venv\Scripts\Activate.ps1   # Windows

# Run with all configured models
python main.py --config config.yaml

# Run with specific models only (faster)
python main.py --config config.yaml \
    --models midas depth_anything dpt_large

# Run on CPU
python main.py --config config.yaml \
    --device cpu

# Skip pose and temporal refinement (fastest)
python main.py --config config.yaml \
    --skip-pose \
    --skip-temporal-refinement

# Run with specific video (override config)
python main.py --config config.yaml \
    --input-video test_data/Ncam/my_video.mp4 \
    --output-dir outputs/my_video_results
```

### 12.4 View results

```bash
# List outputs
ls -la outputs/

# Check metrics CSV
cat outputs/metrics/model_comparison.csv

# Check JSON results
cat outputs/metrics/benchmark_results_results.json

# View in Jupyter
jupyter notebook notebooks/visualization.ipynb
```

---

## 13. Troubleshooting

### 13.1 CUDA out of memory

```python
# Symptom: torch.cuda.OutOfMemoryError
# Fix 1: Reduce number of models
python main.py --models midas --config config.yaml

# Fix 2: Reduce resolution (in config.yaml)
#   resolution: [270, 480]

# Fix 3: Enable FP16 (already on by default)
#   use_fp16: true

# Fix 4: Reduce input_size in model config
#   midas: { input_size: [256, 256] }
```

### 13.2 ModuleNotFoundError for model runners

```python
# Symptom: "No module named 'depth_anything'"
# Fix: The upstream_repo wasn't cloned or installed properly

# Check if repo exists
ls upstream_repos/Depth-Anything/

# Install in editable mode
cd upstream_repos/Depth-Anything && pip install -e .
```

### 13.3 Model weight download failures

```python
# Symptom: RuntimeError about missing checkpoint file
# Fix: Manual download

# For Depth Anything weights:
wget -P ~/.cache/torch/hub/checkpoints/ \
    https://github.com/LiheYoung/Depth-Anything/releases/download/v1.0/depth_anything_vitl14.pth

# For other models, check the runner file for the download URL
# grep -r "download_url\|hf_hub_download" models/*.py
```

### 13.4 MediaPipe not working

```bash
pip install --force-reinstall mediapipe==0.10.0
```

### 13.5 pyrealsense2 installation fails

```bash
# Windows: Download pre-built wheel from:
# https://github.com/IntelRealSense/librealsense/releases
pip install pyrealsense2-*.whl

# Linux
pip install pyrealsense2
```

### 13.6 OpenCV errors

```bash
# On Linux, if cv2.imshow fails:
pip install opencv-python-headless
```

### 13.7 Memory errors during video loading

```bash
# Reduce output_fps in config.yaml to 1 (process 1 frame/sec)
# Or process shorter videos
```

---

## Quick Start One-Liner (after full setup)

```bash
source venv/bin/activate && \
python main.py --config config.yaml \
    --input-video /path/to/your/video.mp4 \
    --models midas depth_anything \
    --skip-pose --skip-temporal-refinement \
    --output-dir outputs/quick_run
```

---

## Verification Checklist

After completing all steps, verify each point:

- [ ] Python 3.8+ installed
- [ ] GPU drivers + CUDA installed (`nvidia-smi` works)
- [ ] Virtual environment created and activated
- [ ] PyTorch installed with CUDA (`torch.cuda.is_available() == True`)
- [ ] All pip packages from `requirements.txt` installed
- [ ] All 17 `upstream_repos/` cloned
- [ ] Model-editable installs done (`pip install -e .`)
- [ ] Depth Pro checkpoint at `checkpoints/depth_pro.pt`
- [ ] `test_e2e.py` runs without errors
- [ ] `config.yaml` edited for your hardware
- [ ] Input video file ready
