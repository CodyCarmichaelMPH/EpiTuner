# EpiTuner Troubleshooting Guide

## Common Issues and Solutions

### 🔧 NVIDIA GPU Issues

#### "NVIDIA GPU found but PyTorch cannot access CUDA"
**Problem:** NVIDIA GPU detected but PyTorch is CPU-only version.

**Solution:**
```powershell
.\setup.ps1 upgrade-torch
```
This installs CUDA-enabled PyTorch version.

#### "8 bit optimizer is not available on your device, only available on CUDA for now"
**Problem:** BitsAndBytes can't access CUDA for 4-bit quantization.

**Status:** ✅ **This is normal and handled automatically**

**What happens:**
- Training automatically falls back to standard LoRA
- GPU training still works (faster than CPU)
- Model quality remains excellent
- Uses more VRAM but works on 6GB+ GPUs

**No action needed** - this is expected behavior on some systems.

### 🖥️ Hardware Detection

#### Intel Graphics Detected Instead of NVIDIA
**Problem:** System has NVIDIA GPU but shows Intel graphics.

**Causes:**
1. PyTorch CPU-only version installed
2. NVIDIA GPU is secondary (common in laptops)
3. CUDA drivers not properly configured

**Solution:**
```powershell
.\setup.ps1 upgrade-torch  # Install CUDA-enabled PyTorch
.\setup.ps1 check-deps    # Verify detection after upgrade
```

### 📁 Model Detection

#### Medical Phi Models Not Found
**Problem:** Custom medical models not mapping correctly.

**Supported variants:**
- `phi`, `phi3`, `phi4` (any variant)
- `medical-phi`, `phi-medical`
- `clinical-phi`, `health-phi`
- Any model with "medical", "clinical", "health" keywords

**All map to appropriate Microsoft Phi models for training.**

### 🚀 Performance

#### Training Very Slow
**Check GPU status:**
```powershell
.\setup.ps1 check-deps
```

**Expected speeds:**
- **NVIDIA GPU**: 5-15 minutes per epoch
- **Intel/AMD GPU**: 15-30 minutes per epoch  
- **CPU only**: 1-3 hours per epoch

**If CPU-only but you have NVIDIA:** Run `.\setup.ps1 upgrade-torch`

### 🔧 Windows-Specific

#### "streamlit is not recognized as the name of a cmdlet"
**Solution:** Use the PowerShell script:
```powershell
.\setup.ps1 run-app
```
This uses `python -m streamlit` which is more reliable.

#### PowerShell Execution Policy Error
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 📦 Installation Issues

#### Package Installation Fails
**Try:**
```powershell
.\setup.ps1 install    # Retry installation
```

**If still failing:**
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### PyTorch Version Conflicts
**Check current version:**
```powershell
python -c "import torch; print(torch.__version__)"
```

**If not 2.2.0-2.7.1 range:**
```powershell
.\setup.ps1 upgrade-torch
```

## 🎯 Expected Behavior

### ✅ Normal Warnings (Safe to Ignore)
- "8 bit optimizer not available" - Falls back to standard LoRA
- "Falling back to standard LoRA" - Normal compatibility handling
- "BitsAndBytes not available" - Uses standard LoRA instead

### ❌ Real Issues (Need Fixing)
- "CUDA out of memory" - Use smaller models or reduce batch size
- "No models found in Ollama" - Install models with `ollama pull`
- "PyTorch not installed" - Run `.\setup.ps1 install`

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check system compatibility:**
   ```powershell
   .\setup.ps1 check-deps
   .\setup.ps1 check-ollama
   ```

2. **Try fresh installation:**
   ```powershell
   git clone https://github.com/CodyCarmichaelMPH/EpiTuner.git
   cd EpiTuner
   .\setup.ps1 install
   .\setup.ps1 setup
   ```

3. **Verify hardware setup:**
   - NVIDIA GPU: Run `.\setup.ps1 upgrade-torch`
   - Ollama models: Run `ollama list` to verify models are installed

The system is designed to be robust and handle most compatibility issues automatically.
