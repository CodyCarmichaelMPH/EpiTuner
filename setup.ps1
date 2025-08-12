# EpiTuner Windows Setup Script
# Run this script in PowerShell to set up EpiTuner on Windows

Write-Host "🚀 EpiTuner Windows Setup" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "⚠️  This script may need administrator privileges for some operations" -ForegroundColor Yellow
}

# Check Python version
Write-Host "`n🐍 Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python detected: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python not found" -ForegroundColor Red
        Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Python not found" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Check pip
Write-Host "`n📦 Checking pip..." -ForegroundColor Cyan
try {
    $pipVersion = pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ pip detected: $pipVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ pip not found" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ pip not found" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "`n⬆️  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "`n📦 Installing Python dependencies..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ requirements.txt not found" -ForegroundColor Red
    exit 1
}

# Check GPU
Write-Host "`n🔍 Checking GPU availability..." -ForegroundColor Cyan
try {
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ GPU check completed" -ForegroundColor Green
    } else {
        Write-Host "⚠️  GPU check failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not check GPU" -ForegroundColor Yellow
}

# Create directories
Write-Host "`n📁 Creating directories..." -ForegroundColor Cyan
$directories = @("outputs", "data", "adapters")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "✅ Created $dir/" -ForegroundColor Green
    } else {
        Write-Host "✅ $dir/ already exists" -ForegroundColor Green
    }
}

# Check Ollama
Write-Host "`n🤖 Checking Ollama..." -ForegroundColor Cyan
try {
    $ollamaVersion = ollama --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Ollama detected: $ollamaVersion" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Ollama not found" -ForegroundColor Yellow
        Write-Host "Install Ollama from https://ollama.ai for local model support" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Ollama not found" -ForegroundColor Yellow
    Write-Host "Install Ollama from https://ollama.ai for local model support" -ForegroundColor Yellow
}

Write-Host "`n" -ForegroundColor White
Write-Host "================================" -ForegroundColor Green
Write-Host "✅ Setup completed successfully!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "1. Install Ollama from https://ollama.ai" -ForegroundColor White
Write-Host "2. Download models: ollama pull phi3" -ForegroundColor White
Write-Host "3. Start EpiTuner: streamlit run app.py" -ForegroundColor White
Write-Host "`nFor help, see README.md" -ForegroundColor White
