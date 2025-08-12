#!/usr/bin/env python3
"""
Minimal GPU PyTorch Installation Fix for EpiTuner
Just installs the correct PyTorch version with GPU support
"""

import subprocess
import sys

def install_gpu_pytorch():
    """Install PyTorch with GPU support"""
    print("🔥 Installing PyTorch with GPU support...")
    
    # Uninstall CPU version first
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                      capture_output=True)
    except:
        pass
    
    # Install GPU version (CUDA 12.1 is most compatible)
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✅ GPU PyTorch installed")
        
        # Test it
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        except:
            print("❌ Import test failed")
            
    else:
        print("❌ Installation failed")

if __name__ == "__main__":
    install_gpu_pytorch()
