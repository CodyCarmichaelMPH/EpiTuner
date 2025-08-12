# GPU Setup for EpiTuner

## Quick GPU Fix

If you have issues with PyTorch GPU support:

```bash
# Install correct PyTorch with GPU support
python install_gpu_pytorch.py

# Verify GPU is working
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## For RTX Quadro 4000 (8GB VRAM)

1. **Install NVIDIA drivers** (if not already installed)
2. **Run the GPU fix**: `python install_gpu_pytorch.py`
3. **Test training**: Use QLoRA mode for memory efficiency

## Minimal Changes Made

- Fixed PyTorch version compatibility check in `scripts/train.py`
- Added `install_gpu_pytorch.py` for easy GPU PyTorch installation
- Existing QLoRA functionality maintained [[memory:5868892]]
- All operations remain local [[memory:5868719]]

The existing EpiTuner code already supports:
- ✅ QLoRA training with BitsAndBytes
- ✅ Memory-efficient training
- ✅ Local model selection
- ✅ GUI interface

Just needs proper GPU PyTorch installation.
