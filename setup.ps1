# EpiTuner Windows Setup Script
# PowerShell script to replace make commands for Windows users

param(
    [string]$Command = "help",
    [string]$Data = "",
    [string]$Model = "",
    [string]$Topic = "",
    [string]$Output = "",
    [string]$Config = "",
    [string]$Predictions = "",
    [string]$GroundTruth = "",
    [string]$OutputDir = ""
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
    Write-Host "  train         Train LoRA model"
    Write-Host "  infer         Run inference"
    Write-Host "  eval          Evaluate model"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\setup.ps1 install"
    Write-Host "  .\setup.ps1 run-app"
    Write-Host "  .\setup.ps1 train -Data 'sample_data\medical_sample.csv' -Model 'microsoft/phi-2' -Topic 'motor vehicle collisions' -Output 'outputs\test_model'"
    Write-Host ""
    Write-Host "Quick start:" -ForegroundColor Magenta
    Write-Host "  .\setup.ps1 install"
    Write-Host "  .\setup.ps1 setup"
    Write-Host "  .\setup.ps1 run-app"
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
    Write-Host "Open your browser to: http://localhost:8501" -ForegroundColor Cyan
    streamlit run app.py
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

function Start-Training {
    if ([string]::IsNullOrEmpty($Data) -or [string]::IsNullOrEmpty($Model) -or [string]::IsNullOrEmpty($Topic) -or [string]::IsNullOrEmpty($Output)) {
        Write-Host "Usage: .\setup.ps1 train -Data <csv_file> -Model <model_name> -Topic <classification_topic> -Output <output_dir>" -ForegroundColor Red
        Write-Host "Example: .\setup.ps1 train -Data 'sample_data\medical_sample.csv' -Model 'microsoft/phi-2' -Topic 'motor vehicle collisions' -Output 'outputs\mvc_model'" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Training LoRA model..." -ForegroundColor Yellow
    if (!(Test-Path $Output)) {
        New-Item -ItemType Directory -Force -Path $Output | Out-Null
    }
    
    python scripts/train.py --config configs/config_base.yaml --data $Data --model $Model --topic $Topic --output $Output
}

function Start-Inference {
    if ([string]::IsNullOrEmpty($Model) -or [string]::IsNullOrEmpty($Config) -or [string]::IsNullOrEmpty($Data) -or [string]::IsNullOrEmpty($Topic) -or [string]::IsNullOrEmpty($Output)) {
        Write-Host "Usage: .\setup.ps1 infer -Model <model_path> -Config <config_path> -Data <csv_file> -Topic <topic> -Output <output_json>" -ForegroundColor Red
        Write-Host "Example: .\setup.ps1 infer -Model 'outputs\mvc_model' -Config 'configs\config_base.yaml' -Data 'sample_data\medical_sample.csv' -Topic 'motor vehicle collisions' -Output 'outputs\predictions.json'" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Running inference..." -ForegroundColor Yellow
    python scripts/inference.py --model $Model --config $Config --data $Data --topic $Topic --output $Output
}

function Start-Evaluation {
    if ([string]::IsNullOrEmpty($Predictions) -or [string]::IsNullOrEmpty($GroundTruth) -or [string]::IsNullOrEmpty($OutputDir)) {
        Write-Host "Usage: .\setup.ps1 eval -Predictions <predictions_json> -GroundTruth <ground_truth_csv> -OutputDir <output_directory>" -ForegroundColor Red
        Write-Host "Example: .\setup.ps1 eval -Predictions 'outputs\predictions.json' -GroundTruth 'sample_data\medical_sample.csv' -OutputDir 'outputs\evaluation'" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Evaluating model..." -ForegroundColor Yellow
    if (!(Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    }
    
    python scripts/evaluate.py --predictions $Predictions --ground_truth $GroundTruth --output_dir $OutputDir
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
    "train" { Start-Training }
    "infer" { Start-Inference }
    "eval" { Start-Evaluation }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Use '.\setup.ps1 help' for available commands" -ForegroundColor Yellow
    }
}