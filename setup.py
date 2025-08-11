#!/usr/bin/env python3
"""
EpiTuner Setup Script
Quick setup and validation for EpiTuner
"""

import subprocess
import sys
import os
from pathlib import Path
import pkg_resources


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_cuda():
    """Check CUDA availability and provide consumer GPU guidance"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✅ CUDA available: {gpu_name}")
            print(f"   VRAM: {vram_gb:.1f}GB")
            
            # Provide specific guidance based on GPU
            if vram_gb >= 12:
                print(f"   🚀 Excellent for LoRA training! Can handle most models.")
                print(f"   💡 Recommended: Phi-2, Mistral-7B")
            elif vram_gb >= 8:
                print(f"   🎯 Good for LoRA training with medium models.")
                print(f"   💡 Recommended: Phi-2, Qwen-1.8B")
            elif vram_gb >= 6:
                print(f"   ⚡ Basic LoRA training possible with small models.")
                print(f"   💡 Recommended: TinyLlama, DialoGPT-medium")
            elif vram_gb >= 4:
                print(f"   ⚠️  Very limited - only tiny models possible.")
                print(f"   💡 Recommended: TinyLlama only")
            else:
                print(f"   ❌ Insufficient VRAM for practical training.")
                
            # Check for common consumer GPUs
            if any(gpu in gpu_name.lower() for gpu in ['gtx', 'rtx']):
                print(f"   🎮 Consumer GPU detected - optimized configs available!")
            
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (very slow training)")
            print("   💡 Consider getting an NVIDIA GPU for practical training")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check CUDA")
        return False


def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            
            # Check for available models
            models_result = subprocess.run(["ollama", "list"], 
                                         capture_output=True, text=True)
            if models_result.returncode == 0:
                models = models_result.stdout.strip().split('\n')[1:]  # Skip header
                if models and models[0].strip():
                    print(f"✅ Available models: {len(models)}")
                    for model in models[:3]:  # Show first 3
                        if model.strip():
                            print(f"   - {model.split()[0]}")
                else:
                    print("⚠️  No Ollama models found")
                    print("   Try: ollama pull llama2")
            return True
        else:
            print("❌ Ollama not found")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found")
        print("   Install from: https://ollama.ai")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "configs", "data", "outputs", "adapters", 
        "chat_templates", "scripts", "workflows", "sample_data"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Directory structure created")


def validate_sample_data():
    """Validate sample data exists and is readable"""
    sample_path = Path("sample_data/medical_sample.csv")
    
    if sample_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(sample_path)
            
            required_cols = ['C_Biosense_ID', 'ChiefComplaintOrig', 'Expert Rating']
            missing_cols = set(required_cols) - set(df.columns)
            
            if not missing_cols:
                print(f"✅ Sample data valid ({len(df)} records)")
                return True
            else:
                print(f"❌ Sample data missing columns: {missing_cols}")
                return False
                
        except Exception as e:
            print(f"❌ Error reading sample data: {e}")
            return False
    else:
        print("❌ Sample data not found")
        return False


def run_quick_test():
    """Run a quick validation test"""
    print("\n🧪 Running quick validation test...")
    
    try:
        # Test imports
        import torch
        import transformers
        import streamlit
        import pandas as pd
        import numpy as np
        
        print("✅ All core imports successful")
        
        # Test sample data loading
        if validate_sample_data():
            print("✅ Sample data validation passed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🏥 EpiTuner Setup")
    print("=" * 50)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    # Check pip
    if not check_pip():
        all_good = False
    
    if not all_good:
        print("\n❌ Basic requirements not met. Please fix the above issues.")
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing Dependencies")
    print("-" * 30)
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Setting Up Directory Structure")
    print("-" * 30)
    create_directories()
    
    # Check additional components
    print("\n🔧 Checking System Components")
    print("-" * 30)
    
    cuda_available = check_cuda()
    ollama_available = check_ollama()
    
    # Run tests
    test_passed = run_quick_test()
    
    # Summary
    print("\n📋 Setup Summary")
    print("=" * 50)
    
    if test_passed:
        print("✅ EpiTuner setup completed successfully!")
        print("\n🚀 Next Steps:")
        print("   1. Run: make run-app")
        print("   2. Open browser to: http://localhost:8501")
        print("   3. Upload your CSV data or use the sample data")
        
        if not cuda_available:
            print("\n⚠️  Note: No CUDA detected - training will be slower on CPU")
        
        if not ollama_available:
            print("\n⚠️  Note: Ollama not found - install from https://ollama.ai for model integration")
    
    else:
        print("❌ Setup encountered issues")
        print("\n🔧 Troubleshooting:")
        print("   1. Check that all dependencies installed correctly")
        print("   2. Verify Python version >= 3.8")
        print("   3. Try running: pip install -r requirements.txt")
    
    print("\n📚 For help, see README.md or run: make help")


if __name__ == "__main__":
    main()
