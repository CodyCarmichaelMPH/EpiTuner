# EpiTuner SFT - Enhanced Fine-Tuning System

A comprehensive fine-tuning system with QLoRA/LoRA support for medical case evaluation and beyond.

## 🚀 Quick Setup (GPU Required)

### Step 1: GPU Setup (CRITICAL)
```bash
# First, ensure GPU PyTorch is installed
python install_gpu.py

# Verify GPU is working
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Complete Setup
```bash
# Run full setup
python setup.py

# OR manual setup:
pip install -r requirements.txt
python setup.py
```

## 💻 System Requirements

### Required
- **Python 3.8+**
- **NVIDIA GPU with 6GB+ VRAM** (for practical training)
- **CUDA 11.8 or 12.1** (latest drivers recommended)

### Recommended Hardware
- **RTX 3080/4070 (10-12GB)**: Excellent for most models
- **RTX 4080/4090 (16-24GB)**: Can handle large models
- **RTX 3060/4060 (8GB)**: Good for smaller models with QLoRA

### Minimum Hardware
- **GTX 1660 Ti (6GB)**: Basic training with tiny models only
- **RTX 2060 (6GB)**: Small model training possible

## ⚠️ CUDA Installation

If you don't have CUDA:

1. **Install NVIDIA Drivers**: https://www.nvidia.com/drivers
2. **Install CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
3. **Verify**: Run `nvidia-smi` in terminal

## 🎯 GPU Memory Guide

| GPU VRAM | Recommended Models | Training Mode |
|----------|-------------------|---------------|
| 24GB+ | Llama-2-7B, Mistral-7B | LoRA/QLoRA |
| 12-16GB | Qwen-2.5-3B, Phi-2 | LoRA/QLoRA |
| 8-10GB | TinyLlama, DialoGPT-Medium | QLoRA only |
| 6GB | TinyLlama only | QLoRA only |
| <6GB | Not recommended | CPU (very slow) |

## 📦 What Gets Installed

### Core Dependencies
- **PyTorch** (GPU version with CUDA support)
- **Transformers** (Hugging Face)
- **PEFT** (Parameter Efficient Fine-Tuning)
- **BitsAndBytes** (Quantization)
- **Accelerate** (Distributed training)

### Data & Evaluation
- **Datasets** (Data loading)
- **Pandas/Numpy** (Data processing)
- **ROUGE/NLTK** (Evaluation metrics)

### UI & Monitoring
- **Streamlit** (Web interface)
- **TensorBoard** (Training monitoring)
- **Plotly** (Visualizations)

## 🔧 Troubleshooting

### Common Issues

**"CUDA not available"**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall GPU PyTorch
python install_gpu.py
```

**"Out of memory"**
```bash
# Use smaller models or QLoRA mode
# Check GPU memory: nvidia-smi
```

**"ModuleNotFoundError"**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### GPU Not Detected
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Reinstall PyTorch: `python install_gpu.py`

### Low VRAM Issues
1. Use QLoRA mode (4-bit quantization)
2. Choose smaller models (TinyLlama, Phi-2)
3. Reduce batch size in config

## 🎮 Supported GPUs

### Excellent (12GB+)
- RTX 4070 Ti, 4080, 4090
- RTX 3080, 3080 Ti, 3090
- Tesla V100, A100

### Good (8-10GB)
- RTX 4070, 3070, 3070 Ti
- RTX 2080 Ti

### Basic (6-8GB)
- RTX 4060, 3060, 2070
- GTX 1660 Ti (6GB variant)

### Not Recommended (<6GB)
- GTX 1060, 1050 Ti
- Any CPU-only setup

## 🚀 After Setup

Once setup is complete:

1. **Test GPU**: Verify CUDA works
2. **Run GUI**: Start the web interface
3. **Load Data**: Upload your CSV or use samples
4. **Train Model**: Choose QLoRA for memory efficiency
5. **Evaluate**: Test your trained model

## 📚 Next Steps

- **GUI Mode**: Run `streamlit run app.py`
- **CLI Mode**: Use command line tools
- **Custom Data**: Prepare your CSV files
- **Model Selection**: Choose appropriate base models

## 🆘 Support

If you encounter issues:

1. **Check GPU**: Run `python install_gpu.py`
2. **Validate Setup**: Run `python setup.py`
3. **Check Logs**: Look for error messages
4. **Hardware**: Verify your GPU meets requirements

---

**⚡ Ready to fine-tune? Your GPU-accelerated training system awaits!**