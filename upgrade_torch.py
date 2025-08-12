#!/usr/bin/env python3
"""
Upgrade script to update PyTorch to compatible version range (2.2.0 to 2.7.1)
This ensures compatibility with BitsAndBytes and other dependencies.
"""

import subprocess
import sys
import importlib.util

def check_torch_version():
    """Check current PyTorch version"""
    try:
        import torch
        current_version = torch.__version__
        print(f"Current PyTorch version: {current_version}")
        
        # Parse version
        version_parts = current_version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 2 or (major == 2 and minor < 2):
            print("❌ PyTorch version too old for BitsAndBytes compatibility")
            return False
        elif major > 2 or (major == 2 and minor > 7):
            print("⚠️  PyTorch version newer than tested range")
            return True
        else:
            print("✅ PyTorch version is compatible")
            return True
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def upgrade_torch():
    """Upgrade PyTorch to compatible version"""
    print("\nUpgrading PyTorch to compatible version...")
    
    # Check what GPUs are detected by system (not just PyTorch)
    nvidia_detected = False
    intel_gpu_detected = False
    amd_gpu_detected = False
    
    try:
        import subprocess
        # Check for NVIDIA
        result = subprocess.run(['powershell', '-Command', 
            'Get-WmiObject -Class Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"} | Select-Object Name'], 
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            nvidia_detected = True
            print("NVIDIA GPU detected by system - installing CUDA version")
            
        # Check for Intel GPU (not just CPU)
        result = subprocess.run(['powershell', '-Command', 
            'Get-WmiObject -Class Win32_VideoController | Where-Object {$_.Name -like "*Intel*" -and $_.Name -notlike "*CPU*"} | Select-Object Name'], 
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            intel_gpu_detected = True
            print("Intel GPU detected by system")
            
        # Check for AMD GPU
        result = subprocess.run(['powershell', '-Command', 
            'Get-WmiObject -Class Win32_VideoController | Where-Object {$_.Name -like "*AMD*" -or $_.Name -like "*Radeon*"} | Select-Object Name'], 
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            amd_gpu_detected = True
            print("AMD GPU detected by system")
    except:
        pass
    
    # Check if CUDA is available to current PyTorch
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("CUDA already available - installing updated CUDA version")
    except ImportError:
        pass
    
    # Determine which PyTorch to install based on detected hardware
    if nvidia_detected or cuda_available:
        print("Installing CUDA-enabled PyTorch for NVIDIA GPU support")
        cmd = [sys.executable, "-m", "pip", "install", "torch>=2.2.0,<=2.7.1", "torchvision>=0.17.0", "torchaudio>=2.2.0", "--index-url", "https://download.pytorch.org/whl/cu121"]
    elif intel_gpu_detected:
        print("Installing PyTorch with Intel GPU support (Intel Extension for PyTorch)")
        # First install regular PyTorch, then Intel extension
        cmd = [sys.executable, "-m", "pip", "install", "torch>=2.2.0,<=2.7.1", "torchvision>=0.17.0", "torchaudio>=2.2.0"]
    elif amd_gpu_detected:
        print("Installing PyTorch with ROCm support for AMD GPU")
        cmd = [sys.executable, "-m", "pip", "install", "torch>=2.2.0,<=2.7.1", "torchvision>=0.17.0", "torchaudio>=2.2.0", "--index-url", "https://download.pytorch.org/whl/rocm5.7"]
    else:
        print("No discrete GPU detected - installing standard PyTorch")
        cmd = [sys.executable, "-m", "pip", "install", "torch>=2.2.0,<=2.7.1", "torchvision>=0.17.0", "torchaudio>=2.2.0"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch upgrade successful")
        
        # Install Intel GPU extension if Intel GPU detected
        if intel_gpu_detected:
            print("Installing Intel Extension for PyTorch...")
            intel_cmd = [sys.executable, "-m", "pip", "install", "intel-extension-for-pytorch"]
            try:
                subprocess.run(intel_cmd, check=True, capture_output=True, text=True)
                print("✅ Intel GPU extension installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Intel GPU extension installation failed, but PyTorch will still work: {e}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch upgrade failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_dependencies():
    """Check if key dependencies work with new PyTorch"""
    deps = ["transformers", "bitsandbytes", "peft", "accelerate"]
    
    print("\nChecking dependency compatibility...")
    for dep in deps:
        try:
            spec = importlib.util.find_spec(dep)
            if spec is not None:
                module = importlib.import_module(dep)
                if hasattr(module, '__version__'):
                    print(f"✅ {dep}: {module.__version__}")
                else:
                    print(f"✅ {dep}: installed")
            else:
                print(f"⚠️  {dep}: not installed")
        except Exception as e:
            print(f"❌ {dep}: error - {e}")

def main():
    print("EpiTuner PyTorch Compatibility Checker")
    print("=" * 40)
    
    # Check current version
    is_compatible = check_torch_version()
    
    if not is_compatible:
        response = input("\nWould you like to upgrade PyTorch? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            success = upgrade_torch()
            if success:
                print("\nRechecking PyTorch version...")
                check_torch_version()
        else:
            print("Skipping upgrade. Note: You may encounter BitsAndBytes compatibility issues.")
    
    # Check other dependencies
    check_dependencies()
    
    print("\n" + "=" * 40)
    print("Compatibility check complete!")
    print("You can now run: .\\setup.ps1 run-app")

if __name__ == "__main__":
    main()
