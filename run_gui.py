#!/usr/bin/env python3
"""
EpiTuner GUI Launcher

This script launches the Streamlit-based GUI for the EpiTuner suite.
It sets up the environment and provides a user-friendly interface for all operations.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['streamlit', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Setup the environment for the GUI."""
    # Add scripts directory to Python path
    scripts_dir = Path(__file__).parent / 'scripts'
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    
    # Add gui directory to Python path
    gui_dir = Path(__file__).parent / 'gui'
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    print("Environment setup complete")

def run_streamlit():
    """Run the Streamlit GUI."""
    gui_file = Path(__file__).parent / 'gui' / 'epituner_gui.py'
    
    if not gui_file.exists():
        print(f"GUI file not found: {gui_file}")
        return False
    
    print("Starting EpiTuner GUI...")
    print("Opening in your default web browser...")
    print("If browser doesn't open automatically, go to: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', str(gui_file),
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 GUI stopped by user")
    except Exception as e:
        print(f"Error running GUI: {e}")
        return False
    
    return True

def main():
    """Main function to launch the GUI."""
    print("EpiTuner - Ollama Fine-Tuning and Evaluation Suite")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Run the GUI
    success = run_streamlit()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 