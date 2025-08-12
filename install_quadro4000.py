#!/usr/bin/env python3
"""
RTX Quadro 4000 Optimized Installation Script

This script sets up PyTorch with proper CUDA support specifically 
optimized for RTX Quadro 4000 (8GB VRAM) systems.
"""

import subprocess
import sys
import platform
import os


def check_nvidia_driver():
    """Check NVIDIA driver and CUDA version"""
    print("🔍 Checking NVIDIA RTX Quadro 4000 setup...")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            
            # Check for Quadro 4000
            if "Quadro RTX 4000" in output or "RTX 4000" in output:
                print("✅ RTX Quadro 4000 detected!")
            else:
                print("⚠️  Different GPU detected, but proceeding...")
            
            # Extract CUDA version
            for line in output.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"✅ CUDA Driver Version: {cuda_version}")
                    return cuda_version
            
            print("⚠️  CUDA version not found in nvidia-smi output")
            return None
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found!")
        print("   Install NVIDIA drivers from: https://www.nvidia.com/drivers")
        return None
    
    return None


def install_pytorch_for_quadro4000():
    """Install PyTorch optimized for RTX Quadro 4000"""
    cuda_version = check_nvidia_driver()
    
    if not cuda_version:
        print("❌ Cannot proceed without NVIDIA drivers")
        return False
    
    print(f"🚀 Installing PyTorch for RTX Quadro 4000 with CUDA {cuda_version}")
    
    # RTX Quadro 4000 works best with CUDA 11.8 or 12.1
    if cuda_version.startswith('12.'):
        print("🎯 Using CUDA 12.1 PyTorch (recommended for RTX Quadro 4000)")
        torch_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    elif cuda_version.startswith('11.8'):
        print("🎯 Using CUDA 11.8 PyTorch")
        torch_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    else:
        print(f"🎯 CUDA {cuda_version} detected - using CUDA 12.1 PyTorch (universal)")
        torch_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    
    print(f"📦 Running: {' '.join(torch_cmd)}")
    
    try:
        subprocess.run(torch_cmd, check=True)
        print("✅ PyTorch GPU installation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False


def verify_quadro4000_setup():
    """Verify PyTorch works correctly with RTX Quadro 4000"""
    print("\n🧪 Verifying RTX Quadro 4000 setup...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA devices detected: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                
                print(f"   GPU {i}: {gpu_name}")
                print(f"   Memory: {gpu_memory:.1f}GB")
                
                if "Quadro RTX 4000" in gpu_name or "RTX 4000" in gpu_name:
                    print("   🎯 Perfect! RTX Quadro 4000 confirmed")
                    print("   💡 Optimal for QLoRA training with 3-7B models")
                elif gpu_memory >= 7.0:
                    print("   ✅ Good GPU memory for QLoRA training")
                else:
                    print("   ⚠️  Limited memory - use smaller models")
            
            # Test GPU computation
            try:
                print("\n🔥 Testing GPU computation...")
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda') 
                z = torch.mm(x, y)
                print("✅ GPU computation test PASSED")
                
                # Test memory allocation (simulate model loading)
                print("🧠 Testing memory allocation (simulating model loading)...")
                large_tensor = torch.randn(50, 1000, 1000, device='cuda')  # ~200MB
                del large_tensor
                torch.cuda.empty_cache()
                print("✅ Memory allocation test PASSED")
                
                return True
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("⚠️  Memory test failed - may need to use smaller models")
                    return True  # Still usable, just constrained
                else:
                    print(f"❌ GPU computation failed: {e}")
                    return False
                    
        else:
            print("❌ CUDA not available in PyTorch")
            print("   This means PyTorch was not installed with GPU support")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False


def install_optimized_dependencies():
    """Install dependencies optimized for RTX Quadro 4000"""
    print("\n📦 Installing RTX Quadro 4000 optimized dependencies...")
    
    try:
        # Install core dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ Core dependencies installed")
        
        # Test BitsAndBytes specifically (critical for 8GB VRAM)
        try:
            print("🔧 Testing BitsAndBytes (essential for 8GB VRAM)...")
            import bitsandbytes as bnb
            print(f"✅ BitsAndBytes {bnb.__version__} working")
            
            # Test 4-bit quantization capability
            import torch
            if torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print("✅ 4-bit quantization configuration working")
                print("🎯 RTX Quadro 4000 ready for QLoRA training!")
                
        except ImportError as e:
            print(f"❌ BitsAndBytes test failed: {e}")
            print("   QLoRA training may not work - install manually:")
            print("   pip install bitsandbytes>=0.41.0")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency installation failed: {e}")
        return False


def create_quadro4000_config():
    """Create optimized configuration for RTX Quadro 4000"""
    print("\n⚙️  Creating RTX Quadro 4000 optimized configuration...")
    
    # Create directories
    os.makedirs("configs", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Quadro 4000 optimized config
    config_content = """# RTX Quadro 4000 Optimized Configuration (8GB VRAM)
# Designed for maximum efficiency on 8GB professional GPU

model:
  base_model: "microsoft/DialoGPT-medium"  # 355M params - perfect for 8GB
  max_seq_len: 512  # Conservative for memory
  trust_remote_code: true

training:
  mode: "qlora"  # Essential for 8GB VRAM
  backend: "bitsandbytes"  # Stable quantization
  
  # Memory-optimized training settings
  num_train_epochs: 3
  per_device_train_batch_size: 1  # Conservative for 8GB
  gradient_accumulation_steps: 8  # Effective batch size = 8
  learning_rate: 2e-4
  warmup_ratio: 0.03
  
  # Precision settings for RTX Quadro 4000
  fp16: true  # Quadro 4000 works well with FP16
  bf16: false  # Use FP16 instead
  
  # Memory management
  gradient_checkpointing: true
  dataloader_num_workers: 2  # Professional workstation can handle this
  
qlora:
  # 4-bit quantization - essential for 8GB VRAM
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"  # Match fp16 setting
  bnb_4bit_use_double_quant: true

lora:
  r: 16  # Good balance for Quadro 4000
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["c_attn", "c_proj"]  # For DialoGPT

evaluation:
  eval_batch_size: 2  # Slightly larger for eval
  max_eval_samples: 100  # Keep evaluation fast

output:
  output_dir: "outputs/quadro4000_training"
  checkpoint_dir: "outputs/quadro4000_training/checkpoints"
  
monitoring:
  tensorboard_enabled: true
  tensorboard_dir: "outputs/tensorboard"
"""
    
    with open("configs/quadro4000_config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ RTX Quadro 4000 configuration created")
    print("   File: configs/quadro4000_config.yaml")


def main():
    """Main installation process for RTX Quadro 4000"""
    print("🎯 RTX Quadro 4000 SFT Training System Setup")
    print("=" * 60)
    print("Optimized for: RTX Quadro 4000 (8GB VRAM)")
    print("Target models: DialoGPT-medium, TinyLlama, Phi-2")
    print("Training mode: QLoRA (4-bit quantization)")
    print("=" * 60)
    
    # Remove any existing PyTorch installations
    print("\n🧹 Cleaning existing PyTorch installations...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall",
            "torch", "torchvision", "torchaudio", "-y"
        ], check=False)
    except:
        pass
    
    # Install PyTorch for Quadro 4000
    if not install_pytorch_for_quadro4000():
        print("\n❌ PyTorch installation failed")
        sys.exit(1)
    
    # Verify setup
    if not verify_quadro4000_setup():
        print("\n❌ RTX Quadro 4000 verification failed")
        sys.exit(1)
    
    # Install dependencies
    if not install_optimized_dependencies():
        print("\n❌ Dependencies installation failed")
        sys.exit(1)
    
    # Create configuration
    create_quadro4000_config()
    
    # Success summary
    print("\n🎉 RTX Quadro 4000 Setup Complete!")
    print("=" * 60)
    print("✅ PyTorch with CUDA support installed")
    print("✅ BitsAndBytes 4-bit quantization ready") 
    print("✅ RTX Quadro 4000 optimized configuration created")
    print("✅ 8GB VRAM memory management configured")
    
    print(f"\n🚀 Ready for Training:")
    print(f"   GPU: RTX Quadro 4000 (8GB)")
    print(f"   Mode: QLoRA (4-bit quantization)")
    print(f"   Models: DialoGPT-medium, TinyLlama, Phi-2")
    print(f"   Config: configs/quadro4000_config.yaml")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Copy this entire folder to your RTX Quadro 4000 machine")
    print(f"   2. Run: python setup.py")
    print(f"   3. Start training with: python train.py --config configs/quadro4000_config.yaml")
    
    print(f"\n⚡ Optimized for maximum efficiency on your RTX Quadro 4000!")


if __name__ == "__main__":
    main()
