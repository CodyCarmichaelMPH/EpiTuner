#!/usr/bin/env python3
"""
EpiTuner Setup Script
Installation and setup for EpiTuner LoRA fine-tuning tool
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_git():
    """Check if git is available"""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git detected")
            return True
        else:
            print("❌ Git not found")
            return False
    except FileNotFoundError:
        print("❌ Git not found")
        return False

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ VRAM: {memory_gb:.1f}GB")
            return True
        else:
            print("⚠️ No CUDA GPU detected - training will be CPU-only (very slow)")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed - GPU check skipped")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "outputs",
        "data",
        "adapters"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")

def check_ollama():
    """Check if Ollama is installed"""
    print("\n🤖 Checking Ollama installation...")
    
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama detected")
            return True
        else:
            print("⚠️ Ollama not found")
            return False
    except FileNotFoundError:
        print("⚠️ Ollama not found")
        print("💡 Install Ollama from https://ollama.ai for local model support")
        return False

def main():
    """Main setup function"""
    print("🚀 EpiTuner Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check git
    check_git()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed - could not install dependencies")
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Check Ollama
    check_ollama()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Install Ollama from https://ollama.ai")
    print("2. Download models: ollama pull phi3")
    print("3. Start EpiTuner: streamlit run app.py")
    print("\nFor help, see README.md")

if __name__ == "__main__":
    main()
