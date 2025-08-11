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
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA detected - installing CUDA version")
            cmd = [sys.executable, "-m", "pip", "install", "torch>=2.2.0,<=2.7.1", "torchvision>=0.17.0", "torchaudio>=2.2.0", "--index-url", "https://download.pytorch.org/whl/cu121"]
        else:
            print("No CUDA - installing CPU version")
            cmd = [sys.executable, "-m", "pip", "install", "torch>=2.2.0,<=2.7.1", "torchvision>=0.17.0", "torchaudio>=2.2.0", "--index-url", "https://download.pytorch.org/whl/cpu"]
    except ImportError:
        print("PyTorch not installed - installing CPU version by default")
        cmd = [sys.executable, "-m", "pip", "install", "torch>=2.2.0,<=2.7.1", "torchvision>=0.17.0", "torchaudio>=2.2.0", "--index-url", "https://download.pytorch.org/whl/cpu"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch upgrade successful")
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
