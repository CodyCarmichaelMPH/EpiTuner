# 🪟 EpiTuner Windows Setup Guide

## Prerequisites for Windows 10/11

### 1. NVIDIA GPU Setup

**Supported Consumer GPUs:**
- **Excellent**: RTX 4060 Ti (16GB), RTX 3060 (12GB), RTX 4070+
- **Good**: RTX 4060 (8GB), RTX 3060 (8GB), RTX 3070
- **Basic**: GTX 1660 Ti (6GB), RTX 2060 (6GB)
- **Minimal**: GTX 1660 (6GB), GTX 1650 (4GB) - TinyLlama only

**Install NVIDIA Drivers:**
1. Download latest drivers from [nvidia.com/drivers](https://nvidia.com/drivers)
2. Install and restart your computer
3. Verify installation:
   ```cmd
   nvidia-smi
   ```

### 2. Python Installation

**Download Python 3.11:**
1. Go to [python.org/downloads](https://python.org/downloads)
2. Download Python 3.11.x (64-bit)
3. **Important**: Check "Add Python to PATH" during installation
4. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### 3. Git Installation (Optional)
Download from [git-scm.com](https://git-scm.com) if you want to clone the repository.

## Quick Setup

### Option 1: Automated Setup
```cmd
# Run the automated setup script
python setup.py
```

### Option 2: Manual Setup
```cmd
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir configs data outputs adapters chat_templates scripts workflows sample_data

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Consumer GPU Optimization

### For 4-6GB VRAM (GTX 1660 Ti, RTX 3060 8GB)

**Recommended Models:**
- TinyLlama (1.1B) - Safe choice
- DialoGPT-medium (345M) - Very conservative

**Configuration:**
```yaml
# Use configs/config_consumer_gpu.yaml
model:
  max_seq_len: 128
train:
  batch_size: 1
  gradient_accumulation_steps: 8
  num_epochs: 1
tuning:
  lora_r: 4
  lora_alpha: 8
```

### For 8-12GB VRAM (RTX 3060 12GB, RTX 4060 Ti)

**Recommended Models:**
- Phi-2 (2.7B) - Good balance
- Qwen-1.8B - Efficient alternative
- TinyLlama - Fast iteration

**Configuration:**
```yaml
# Use configs/config_base.yaml with modifications
model:
  max_seq_len: 256
train:
  batch_size: 1
  gradient_accumulation_steps: 4
  num_epochs: 2
tuning:
  lora_r: 8
  lora_alpha: 16
```

## Windows-Specific Tips

### Performance Optimization

**Before Training:**
1. Close Chrome/Edge browsers (GPU memory hogs)
2. Close games and GPU-intensive applications
3. Check GPU usage in Task Manager
4. Set Windows to High Performance mode

**Monitor During Training:**
- Open Task Manager → Performance → GPU
- Watch GPU memory usage
- Normal: 70-90% utilization
- Warning: >95% utilization (may cause OOM)

### Common Windows Issues

**Issue**: Python not found
```cmd
# Fix: Add Python to PATH
setx PATH "%PATH%;C:\Users\[YOUR_USERNAME]\AppData\Local\Programs\Python\Python311"
```

**Issue**: CUDA not available
```cmd
# Check NVIDIA driver
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

**Issue**: Multiprocessing errors
```yaml
# In config file, set:
train:
  dataloader_num_workers: 0
```

**Issue**: Out of memory
1. Use `configs/config_consumer_gpu.yaml`
2. Close other applications
3. Reduce `lora_r` to 4
4. Use TinyLlama model

## Ollama Installation (Windows)

### Download and Install
1. Visit [ollama.ai](https://ollama.ai)
2. Download Windows installer
3. Install and restart

### Pull Recommended Models
```cmd
# For 4-6GB cards
ollama pull tinyllama

# For 8GB+ cards  
ollama pull phi3
ollama pull llama3.2:1b

# For 12GB+ cards
ollama pull mistral
```

### Verify Ollama
```cmd
ollama list
ollama run tinyllama "Hello, how are you?"
```

## Testing Your Setup

### Quick Test
```cmd
# Run the automated setup check
python setup.py

# If everything passes, start EpiTuner
python -m streamlit run app.py
```

### Training Test
```cmd
# Test with sample data
make train DATA=sample_data/medical_sample.csv MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 TOPIC="test classification" OUTPUT=outputs/test_model CONFIG=configs/config_consumer_gpu.yaml
```

## Performance Expectations

### Training Times (Sample 10 Records)

**GTX 1660 Ti (6GB):**
- TinyLlama: ~3-5 minutes
- Larger models: Not recommended

**RTX 3060 (8GB):**
- TinyLlama: ~2-3 minutes  
- Phi-2: ~5-8 minutes

**RTX 4060 (8GB):**
- TinyLlama: ~1-2 minutes
- Phi-2: ~3-5 minutes

**RTX 4060 Ti (16GB):**
- TinyLlama: ~1 minute
- Phi-2: ~2-3 minutes
- Larger models: ~5-10 minutes

## Troubleshooting

### Training Fails with OOM
1. **Step 1**: Use `configs/config_consumer_gpu.yaml`
2. **Step 2**: Close all other applications
3. **Step 3**: Reduce LoRA rank to 4
4. **Step 4**: Use TinyLlama model
5. **Step 5**: Reduce max_seq_len to 64

### Slow Training
1. **Check GPU usage** in Task Manager
2. **Verify CUDA** is being used (not CPU)
3. **Close browser tabs** and other apps
4. **Use smaller batch sizes** but more accumulation

### Application Won't Start
```cmd
# Check Python
python --version

# Check Streamlit
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit

# Run with verbose output
streamlit run app.py --logger.level debug
```

## Next Steps

1. **✅ Setup Complete**: Run `make run-app`
2. **📊 Upload Data**: Use the sample CSV or your own
3. **🤖 Select Model**: Choose based on your GPU memory
4. **🚀 Start Training**: Begin with 1 epoch for testing
5. **👨‍⚕️ Review Results**: Use expert feedback to improve

## Support

For Windows-specific issues:
1. Check this guide first
2. Verify GPU drivers and Python installation  
3. Test with sample data before using real data
4. Start with TinyLlama model for initial testing

**GPU Memory Issues?** Always start with the consumer GPU config and work your way up to larger models as you get comfortable with the system.
