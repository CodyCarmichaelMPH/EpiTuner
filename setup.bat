@echo off
REM EpiTuner Windows Setup Script (Batch version)
REM Simple batch commands for Windows users

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="setup" goto setup
if "%1"=="clean" goto clean
if "%1"=="run-app" goto run-app
if "%1"=="check-deps" goto check-deps
if "%1"=="check-ollama" goto check-ollama

echo Unknown command: %1
echo Use 'setup.bat help' for available commands
goto end

:help
echo EpiTuner - LoRA Fine-tuning for Medical Data
echo.
echo Available commands:
echo   install       Install Python dependencies
echo   setup         Create necessary directories  
echo   clean         Clean up generated files
echo   run-app       Start Streamlit application
echo   check-deps    Check system dependencies
echo   check-ollama  Check Ollama installation and models
echo.
echo Examples:
echo   setup.bat install
echo   setup.bat run-app
echo.
echo For advanced features, use: setup.ps1
goto end

:install
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies!
    goto end
)
echo Dependencies installed successfully!
goto end

:setup
echo Creating directory structure...
if not exist "configs" mkdir configs
if not exist "data" mkdir data  
if not exist "outputs" mkdir outputs
if not exist "adapters" mkdir adapters
if not exist "chat_templates" mkdir chat_templates
if not exist "scripts" mkdir scripts
if not exist "workflows" mkdir workflows
if not exist "sample_data" mkdir sample_data
echo Directory structure created!
goto end

:clean
echo Cleaning up generated files...
if exist "outputs\training_*" rmdir /s /q "outputs\training_*" 2>nul
if exist "outputs\predictions_*" rmdir /s /q "outputs\predictions_*" 2>nul
if exist "outputs\evaluation_*" rmdir /s /q "outputs\evaluation_*" 2>nul
if exist "__pycache__" rmdir /s /q "__pycache__" 2>nul
if exist ".streamlit" rmdir /s /q ".streamlit" 2>nul
del /s /q "*.pyc" 2>nul
del /s /q "*.pyo" 2>nul
echo Cleanup complete!
goto end

:run-app
echo Starting EpiTuner application...
echo Open your browser to: http://localhost:8501
streamlit run app.py
goto end

:check-deps
echo Checking system dependencies...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found!
    goto end
)
echo Python: OK

pip --version >nul 2>&1  
if %errorlevel% neq 0 (
    echo pip not found!
    goto end
)
echo pip: OK

echo Checking Python packages...
python -c "import torch; print('PyTorch: OK')" 2>nul || echo PyTorch: NOT INSTALLED
python -c "import transformers; print('Transformers: OK')" 2>nul || echo Transformers: NOT INSTALLED  
python -c "import streamlit; print('Streamlit: OK')" 2>nul || echo Streamlit: NOT INSTALLED
python -c "import pandas; print('Pandas: OK')" 2>nul || echo Pandas: NOT INSTALLED
goto end

:check-ollama
echo Checking Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama not found! Install from https://ollama.ai
    echo After installing Ollama, try:
    echo   ollama pull llama3.2:1b    # Small model for limited hardware
    echo   ollama pull phi3           # Good general model
    goto end
)
echo Ollama: OK
echo Available Ollama models:
ollama list
goto end

:end
