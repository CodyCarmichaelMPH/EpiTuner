# EpiTuner Windows Setup Script
# PowerShell script to replace make commands for Windows users

param(
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "EpiTuner - LoRA Fine-tuning for Medical Data" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "  install       Install Python dependencies"
    Write-Host "  setup         Create necessary directories"
    Write-Host "  clean         Clean up generated files"
    Write-Host "  run-app       Start Streamlit application"
    Write-Host "  check-deps    Check system dependencies"
    Write-Host "  check-ollama  Check Ollama installation and models"
    Write-Host "  upgrade-torch Check and upgrade PyTorch to compatible version"
    Write-Host ""
    Write-Host "Quick start:" -ForegroundColor Green
    Write-Host "  .\setup.ps1 install"
    Write-Host "  .\setup.ps1 setup"
    Write-Host "  .\setup.ps1 run-app"
    Write-Host ""
    Write-Host "For advanced training features, use the Streamlit GUI" -ForegroundColor Gray
}

function Install-Dependencies {
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Dependencies installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Error installing dependencies!" -ForegroundColor Red
        exit 1
    }
}

function Setup-Directories {
    Write-Host "Creating directory structure..." -ForegroundColor Yellow
    $directories = @("configs", "data", "outputs", "adapters", "chat_templates", "scripts", "workflows", "sample_data")
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
            Write-Host "Created: $dir" -ForegroundColor Gray
        }
    }
    Write-Host "Directory structure created!" -ForegroundColor Green
}

function Clean-Files {
    Write-Host "Cleaning up generated files..." -ForegroundColor Yellow
    
    # Remove training outputs
    Get-ChildItem -Path "outputs" -Filter "training_*" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    
    # Remove prediction outputs
    Get-ChildItem -Path "outputs" -Filter "predictions_*" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    
    # Remove evaluation outputs
    Get-ChildItem -Path "outputs" -Filter "evaluation_*" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    
    # Remove Python cache
    if (Test-Path "__pycache__") {
        Remove-Item "__pycache__" -Recurse -Force
    }
    
    if (Test-Path ".streamlit") {
        Remove-Item ".streamlit" -Recurse -Force
    }
    
    # Remove .pyc files
    Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Path . -Recurse -Filter "*.pyo" -ErrorAction SilentlyContinue | Remove-Item -Force
    
    Write-Host "Cleanup complete!" -ForegroundColor Green
}

function Start-App {
    Write-Host "Starting EpiTuner application..." -ForegroundColor Yellow
    
    # Check if streamlit is available
    $streamlitCheck = python -c "import streamlit" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Streamlit not found!" -ForegroundColor Red
        Write-Host "Please run: .\setup.ps1 install" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Open your browser to: http://localhost:8501" -ForegroundColor Cyan
    python -m streamlit run app.py
}

function Check-Dependencies {
    Write-Host "Checking system dependencies..." -ForegroundColor Yellow
    
    # Check Python
    $pythonCheck = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Python: $pythonCheck" -ForegroundColor Green
    } else {
        Write-Host "✗ Python not found!" -ForegroundColor Red
        return
    }
    
    # Check pip
    $pipCheck = pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ pip available" -ForegroundColor Green
    } else {
        Write-Host "✗ pip not found!" -ForegroundColor Red
        return
    }
    
    # Check Python packages
    Write-Host "Checking Python packages..." -ForegroundColor Yellow
    
    python -c "import torch; print('torch: OK')" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "✓ torch: OK" -ForegroundColor Green } else { Write-Host "✗ torch not installed!" -ForegroundColor Red }
    
    python -c "import transformers; print('transformers: OK')" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "✓ transformers: OK" -ForegroundColor Green } else { Write-Host "✗ transformers not installed!" -ForegroundColor Red }
    
    python -c "import streamlit; print('streamlit: OK')" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "✓ streamlit: OK" -ForegroundColor Green } else { Write-Host "✗ streamlit not installed!" -ForegroundColor Red }
    
    python -c "import pandas; print('pandas: OK')" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "✓ pandas: OK" -ForegroundColor Green } else { Write-Host "✗ pandas not installed!" -ForegroundColor Red }
    
    python -c "import numpy; print('numpy: OK')" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "✓ numpy: OK" -ForegroundColor Green } else { Write-Host "✗ numpy not installed!" -ForegroundColor Red }
    
    Write-Host "Dependency check complete!" -ForegroundColor Green
}

function Check-Ollama {
    Write-Host "Checking Ollama installation..." -ForegroundColor Yellow
    
    $ollamaCheck = ollama --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Ollama: $ollamaCheck" -ForegroundColor Green
        
        Write-Host "Available Ollama models:" -ForegroundColor Cyan
        ollama list
    } else {
        Write-Host "✗ Ollama not found! Install from https://ollama.ai" -ForegroundColor Red
        Write-Host "After installing Ollama, try:" -ForegroundColor Yellow
        Write-Host "  ollama pull llama3.2:1b    # Small model for limited hardware"
        Write-Host "  ollama pull phi3           # Good general model"
    }
}

function Upgrade-Torch {
    Write-Host "Checking PyTorch compatibility..." -ForegroundColor Yellow
    python upgrade_torch.py
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Dependencies }
    "setup" { Setup-Directories }
    "clean" { Clean-Files }
    "run-app" { Start-App }
    "check-deps" { Check-Dependencies }
    "check-ollama" { Check-Ollama }
    "upgrade-torch" { Upgrade-Torch }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Use '.\setup.ps1 help' for available commands" -ForegroundColor Yellow
    }
}