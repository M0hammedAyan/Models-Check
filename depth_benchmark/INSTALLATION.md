# Installation Guide

## Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- 16GB+ RAM (32GB recommended for 4K processing)
- 50GB+ disk space for model weights

## Step-by-Step Installation

### 1. Environment Setup

```bash
# Navigate to project directory
cd ~/projet1.0/model\ testing/depth_benchmark

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core packages
pip install -r requirements.txt
```

### 3. Install Model-Specific Dependencies

#### Option A: Complete Installation (All Models)

```bash
# Create a temporary directory for installations
mkdir -p ~/tmp/models
cd ~/tmp/models

# Depth Anything
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -e .
cd ..

# Depth Anything V2
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -e .
cd ..

# ZoeDepth
git clone https://github.com/isl-org/ZoeDepth
cd ZoeDepth
pip install -e .
cd ..

# Depth Pro (requires registration)
# pip install git+https://github.com/apple/depth-pro.git

# Return to project directory
cd ~/projet1.0/model\ testing/depth_benchmark
```

#### Option B: Minimal Installation (Core Models Only)

```bash
# Install timm for MiDaS and DPT
pip install timm

# Install MediaPipe for pose estimation
pip install mediapipe

# Optional: RealSense for .bag file support
pip install pyrealsense2
```

### 4. Verify Installation

```bash
# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test project imports
python -c "from common import BenchmarkConfig; print('✓ Common imports OK')"
python -c "from models import ModelFactory; print('✓ Models OK')"
python -c "from benchmark import BenchmarkPipeline; print('✓ Benchmark OK')"

# List available models
python -c "from models import ModelFactory; print('Available:', ModelFactory.list_models())"
```

## GPU Setup (Optional)

### CUDA Installation

For NVIDIA GPU acceleration:

```bash
# Check NVIDIA GPU
nvidia-smi

# Install CUDA toolkit (if needed)
# Visit: https://developer.nvidia.com/cuda-downloads

# Verify CUDA with PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Memory Optimization

If you have limited GPU memory, modify `config.yaml`:

```yaml
processing:
  batch_size: 1              # Reduce from default
  use_fp16: true            # Enable FP16 for memory savings
  tile_size: 256            # Reduce tile size for 4K

models:
  # Use smaller model variants
  midas:
    model_type: midas
    input_size: [256, 256]  # Reduce input size
```

## Docker Setup (Optional)

### Build Docker Image

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

CMD ["python", "main.py", "--config", "config.yaml"]
EOF

# Build image
docker build -t depth-benchmark:latest .
```

### Run Docker Container

```bash
docker run --gpus all -v $(pwd):/app depth-benchmark:latest \
  python main.py --config config.yaml
```

## Troubleshooting

### Issue: ModuleNotFoundError for model-specific packages

**Solution**: Install the specific model's dependencies:

```bash
# For Depth Anything
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything && pip install -e .

# For ZoeDepth
pip install git+https://github.com/isl-org/ZoeDepth.git
```

### Issue: CUDA Out of Memory

**Solution**: Reduce memory usage:

```bash
# Edit config.yaml
processing:
  batch_size: 1
  use_fp16: true
  tile_size: 256

models:
  midas:
    input_size: [256, 256]
```

### Issue: pyrealsense2 Installation Fails

**Solution**: Install from conda or pre-built wheels:

```bash
# Via conda
conda install pyrealsense2

# Or pre-built wheel (visit: https://github.com/IntelRealSense/librealsense/releases)
pip install pyrealsense2-*.whl
```

### Issue: MediaPipe Not Working

**Solution**: Reinstall with specific version:

```bash
pip install --force-reinstall mediapipe==0.10.0
```

## Performance Benchmarks

Expected performance on various hardware:

### GPU Performance (1080p input)

| GPU | Model | Speed (FPS) | Memory (GB) |
|-----|-------|------------|-----------|
| RTX 3090 | MiDaS | 45 | 2.0 |
| RTX 3090 | Depth Anything | 30 | 3.0 |
| RTX 3090 | DPT-Large | 25 | 2.5 |
| RTX 4090 | MiDaS | 80 | 2.0 |
| RTX 4090 | Depth Anything | 55 | 3.0 |

### CPU Performance (1080p input)

| CPU | Model | Speed (FPS) | Memory (GB) |
|-----|-------|------------|-----------|
| i7-13700K | MiDaS | 3 | 4.0 |
| i7-13700K | Depth Anything | 1.5 | 6.0 |

## Next Steps

After installation:

1. **Update Configuration**: Edit `config.yaml` for your setup
2. **Test Installation**: Run `python main.py --config config.yaml --models midas`
3. **Prepare Data**: Organize test videos in `test_data/` folder
4. **Run Benchmark**: Execute full pipeline
5. **View Results**: Check `outputs/` directory

## Support

For installation issues:

1. Check the README.md for additional context
2. Review GitHub issues for similar problems
3. Create detailed bug report with:
   - Python version
   - PyTorch version
   - CUDA version
   - Error message
   - Steps to reproduce

---

**Last Updated**: January 2024
