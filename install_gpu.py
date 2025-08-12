#!/usr/bin/env python3
"""
GPU PyTorch Installation Script for EpiTuner SFT

This script ensures proper GPU-enabled PyTorch installation with CUDA support.
Run this before the main setup if you have GPU issues.
"""

import subprocess
import sys
import platform


def detect_cuda_version():
    """Detect installed CUDA version"""
    print("🔍 Detecting CUDA version...")
    
    try:
        # Try nvidia-smi first
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"✅ CUDA {cuda_version} detected via nvidia-smi")
                    return cuda_version
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    
    # Try nvcc as backup
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    # Extract version like "release 11.8, V11.8.89"
                    version_part = line.split('release')[1].split(',')[0].strip()
                    print(f"✅ CUDA {version_part} detected via nvcc")
                    return version_part
    except FileNotFoundError:
        print("❌ nvcc not found")
    
    return None


def install_pytorch_gpu():
    """Install PyTorch with GPU support based on detected CUDA version"""
    cuda_version = detect_cuda_version()
    
    if not cuda_version:
        print("❌ No CUDA detected!")
        print("\n🔧 Troubleshooting:")
        print("   1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("   2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("   3. Verify with: nvidia-smi")
        print("\n⚠️  Will install CPU version for now (very slow training!)")
        
        install_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio"
        ]
    else:
        print(f"🚀 Installing PyTorch for CUDA {cuda_version}")
        
        # Map CUDA versions to appropriate PyTorch installations
        if cuda_version.startswith('12.4'):
            index_url = "https://download.pytorch.org/whl/cu124"
        elif cuda_version.startswith('12.1'):
            index_url = "https://download.pytorch.org/whl/cu121" 
        elif cuda_version.startswith('11.8'):
            index_url = "https://download.pytorch.org/whl/cu118"
        else:
            # Default to latest supported CUDA version
            print(f"⚠️  CUDA {cuda_version} not directly supported, using CUDA 12.1 PyTorch")
            index_url = "https://download.pytorch.org/whl/cu121"
        
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url
        ]
    
    print(f"📦 Running: {' '.join(install_cmd)}")
    
    try:
        subprocess.run(install_cmd, check=True)
        print("✅ PyTorch installation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False


def verify_installation():
    """Verify PyTorch GPU installation"""
    print("\n🧪 Verifying PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU devices: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Test basic GPU operation
            try:
                x = torch.randn(10, 10).cuda()
                y = torch.randn(10, 10).cuda()
                z = torch.mm(x, y)
                print("✅ GPU computation test passed")
                return True
            except Exception as e:
                print(f"❌ GPU computation test failed: {e}")
                return False
        else:
            print("❌ CUDA not available in PyTorch")
            print("   This means you have CPU-only PyTorch installed")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False


def install_bitsandbytes():
    """Install BitsAndBytes for quantization"""
    print("\n🔧 Installing BitsAndBytes for quantization...")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  Skipping BitsAndBytes - CUDA not available")
            return False
        
        # Install BitsAndBytes
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "bitsandbytes>=0.41.0"
        ], check=True)
        
        # Test import
        import bitsandbytes as bnb
        print(f"✅ BitsAndBytes {bnb.__version__} installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ BitsAndBytes installation failed: {e}")
        print("   Will use standard LoRA without quantization")
        return False
    except ImportError as e:
        print(f"❌ BitsAndBytes import failed: {e}")
        return False


def main():
    """Main installation function"""
    print("🔥 GPU PyTorch Installation for EpiTuner SFT")
    print("=" * 60)
    
    if platform.system() != "Windows":
        print("⚠️  This script is optimized for Windows")
        print("   For Linux/Mac, use: pip install torch torchvision torchaudio --index-url <cuda-url>")
    
    # Uninstall existing PyTorch to avoid conflicts
    print("\n🧹 Removing existing PyTorch installations...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", 
            "torch", "torchvision", "torchaudio", "-y"
        ], check=False)  # Don't fail if not installed
    except:
        pass
    
    # Install PyTorch with GPU support
    if not install_pytorch_gpu():
        print("\n❌ PyTorch installation failed")
        sys.exit(1)
    
    # Verify installation
    if verify_installation():
        print("\n🎉 GPU PyTorch installation successful!")
        
        # Install BitsAndBytes
        install_bitsandbytes()
        
        print("\n✅ Ready for GPU fine-tuning!")
        print("\n🚀 Next steps:")
        print("   1. Run: python setup.py")
        print("   2. Follow the setup instructions")
        
    else:
        print("\n❌ GPU verification failed")
        print("\n🔧 Troubleshooting:")
        print("   1. Check NVIDIA drivers are installed")
        print("   2. Verify CUDA toolkit installation")
        print("   3. Restart your terminal/IDE")
        print("   4. Try running: nvidia-smi")


if __name__ == "__main__":
    main()
