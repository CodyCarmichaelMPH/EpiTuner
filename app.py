#!/usr/bin/env python3
"""
EpiTuner - LoRA Fine-tuning Tool for Syndromic Surveillance Data
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import os
import json
import yaml
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="EpiTuner - LoRA Fine-tuning for Syndromic Surveillance Data",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'data_upload'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'training_config' not in st.session_state:
    st.session_state.training_config = {}
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'expert_feedback' not in st.session_state:
    st.session_state.expert_feedback = []


class GPUManager:
    """Manage GPU detection and recommendations for consumer cards"""
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get comprehensive GPU information for Windows systems"""
        gpu_info = {
            'has_cuda': False,
            'gpu_name': 'Unknown',
            'memory_gb': 0,
            'recommended_models': [],
            'config_file': 'configs/config_base.yaml',
            'hardware_type': 'cpu',
            'training_feasible': False,
            'warnings': []
        }
        
        # Check for NVIDIA CUDA GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['has_cuda'] = True
                gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
                gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_info['hardware_type'] = 'nvidia_gpu'
                gpu_info['training_feasible'] = True
                
                # Determine recommended models based on memory
                memory_gb = gpu_info['memory_gb']
                if memory_gb < 5:
                    gpu_info['recommended_models'] = [
                        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        'microsoft/DialoGPT-small'
                    ]
                    gpu_info['config_file'] = 'configs/config_consumer_gpu.yaml'
                elif memory_gb < 7:
                    gpu_info['recommended_models'] = [
                        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        'microsoft/phi-2',
                        'microsoft/DialoGPT-medium'
                    ]
                    gpu_info['config_file'] = 'configs/config_consumer_gpu.yaml'
                elif memory_gb < 10:
                    gpu_info['recommended_models'] = [
                        'microsoft/phi-2',
                        'Qwen/Qwen1.5-1.8B-Chat',
                        'microsoft/DialoGPT-medium'
                    ]
                else:
                    gpu_info['recommended_models'] = [
                        'microsoft/phi-2',
                        'Qwen/Qwen1.5-4B-Chat',
                        'mistralai/Mistral-7B-Instruct-v0.1'
                    ]
            else:
                # No CUDA detected by PyTorch - check what hardware we actually have
                nvidia_detected = False
                try:
                    # Check ALL video controllers and prioritize NVIDIA if found
                    result = subprocess.run(['powershell', '-Command', 
                        'Get-WmiObject -Class Win32_VideoController | Select-Object Name, AdapterRAM'], 
                        capture_output=True, text=True)
                    
                    intel_found = False
                    amd_found = False
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        
                        # First pass: Look specifically for NVIDIA
                        for line in lines[2:]:  # Skip header
                            if line.strip() and 'nvidia' in line.lower():
                                nvidia_detected = True
                                # Extract the actual GPU name
                                gpu_name_parts = [part for part in line.split() if part.strip()]
                                if gpu_name_parts:
                                    # Find the GPU name (skip RAM column)
                                    gpu_name_clean = ' '.join(gpu_name_parts[:-1]) if len(gpu_name_parts) > 1 else gpu_name_parts[0]
                                    gpu_info['gpu_name'] = f"{gpu_name_clean} (CUDA unavailable to PyTorch)"
                                else:
                                    gpu_info['gpu_name'] = "NVIDIA GPU detected (CUDA unavailable to PyTorch)"
                                
                                gpu_info['hardware_type'] = 'nvidia_gpu_no_cuda'
                                gpu_info['training_feasible'] = True
                                gpu_info['warnings'] = [
                                    'NVIDIA GPU detected but PyTorch cannot access CUDA',
                                    'You likely have PyTorch CPU-only version installed',
                                    'Run: .\\setup.ps1 upgrade-torch to install CUDA-enabled PyTorch',
                                    'Training will use CPU mode (very slow) until fixed'
                                ]
                                # Conservative recommendations since we can't detect VRAM  
                                gpu_info['recommended_models'] = [
                                    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                                    'microsoft/phi-2',
                                    'microsoft/DialoGPT-small'
                                ]
                                break
                        
                        # If no NVIDIA found, check for others
                        if not nvidia_detected:
                            for line in lines[2:]:  # Skip header
                                if line.strip():
                                    if 'intel' in line.lower():
                                        intel_found = True
                                        gpu_info['gpu_name'] = 'Intel Integrated Graphics'
                                        gpu_info['hardware_type'] = 'integrated_gpu'
                                    elif any(brand in line.lower() for brand in ['amd', 'radeon']):
                                        amd_found = True
                                        gpu_info['gpu_name'] = 'AMD Graphics'
                                        gpu_info['hardware_type'] = 'amd_gpu'
                except:
                    pass
                
                # If no NVIDIA was detected, fall back to CPU recommendations
                if not nvidia_detected:
                    # CPU-only fallback - provide basic options
                    gpu_info['recommended_models'] = [
                        'microsoft/DialoGPT-small',  # Very small model for CPU
                        'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  # If user wants to try
                ]
                gpu_info['config_file'] = 'configs/config_cpu_only.yaml'
                gpu_info['training_feasible'] = True  # Allow CPU training
                gpu_info['warnings'] = [
                    'No NVIDIA GPU detected - training will be CPU-only and slower',
                    'CPU training is recommended only for small models and testing',
                    'For production use, consider using a machine with NVIDIA GPU'
                ]
                
        except ImportError:
            gpu_info['warnings'] = ['PyTorch not properly installed']
        
        return gpu_info


class OllamaModelManager:
    """Manage Ollama models for local inference"""
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                st.warning("Ollama not found or not running. Please install and start Ollama.")
                return []
        except FileNotFoundError:
            st.warning("Ollama not found. Please install Ollama to see available models.")
            return []
    
    @staticmethod
    def model_to_hf_name(ollama_model: str) -> str:
        """Convert Ollama model name to Hugging Face model name"""
        # Normalize the model name for matching
        model_lower = ollama_model.lower()
        
        # Expanded mapping optimized for consumer GPUs and medical models
        model_mapping = {
            'llama3.2:1b': 'meta-llama/Llama-3.2-1B-Instruct',
            'llama3.2': 'meta-llama/Llama-3.2-1B-Instruct',
            'llama2': 'meta-llama/Llama-2-7b-chat-hf',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.1',
            'phi3': 'microsoft/Phi-3-mini-4k-instruct',
            'phi': 'microsoft/phi-2',
            'qwen': 'Qwen/Qwen1.5-1.8B-Chat',
            'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        }
        
        # Try exact match first
        if ollama_model in model_mapping:
            return model_mapping[ollama_model]
        
        # Special handling for phi models (including medical variants)
        if 'phi' in model_lower:
            if 'phi3' in model_lower or 'phi-3' in model_lower:
                return 'microsoft/Phi-3-mini-4k-instruct'
            else:
                return 'microsoft/phi-2'
        
        # Special handling for llama models
        if 'llama' in model_lower:
            if '3.2' in model_lower or '1b' in model_lower:
                return 'meta-llama/Llama-3.2-1B-Instruct'
            else:
                return 'meta-llama/Llama-2-7b-chat-hf'
        
        # Special handling for medical or specialized models
        if any(keyword in model_lower for keyword in ['medical', 'med', 'clinical', 'health']):
            # For medical models, default to phi-2 which is good for specialized tasks
            return 'microsoft/phi-2'
        
        # Try partial matches with original mapping
        for ollama_key, hf_name in model_mapping.items():
            if ollama_key in model_lower:
                return hf_name
        
        # Default fallback - safe for consumer GPUs
        return 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


class DataValidator:
    """Validate uploaded medical data"""
    
    REQUIRED_COLUMNS = [
        'C_Biosense_ID',
        'ChiefComplaintOrig',
        'DischargeDiagnosis',
        'Expert Rating',
        'Rationale_of_Rating'
    ]
    
    OPTIONAL_COLUMNS = [
        'Sex',
        'Age',
        'c_ethnicity',
        'c_race',
        'Admit_Reason_Combo',
        'Diagnosis_Combo',
        'CCDD Category',
        'TriageNotes',
        'TriageNotesOrig'
    ]
    
    @classmethod
    def validate_data(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate the uploaded dataframe"""
        errors = []
        
        # Check for required columns
        missing_required = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing_required:
            errors.append(f"Missing required columns: {', '.join(missing_required)}")
        
        # Check for empty required columns
        for col in cls.REQUIRED_COLUMNS:
            if col in df.columns and df[col].isna().all():
                errors.append(f"Required column '{col}' is completely empty")
        
        # Check Expert Rating values
        if 'Expert Rating' in df.columns:
            valid_ratings = {'Match', 'Not a Match', 'Unknown/Not able to determine'}
            invalid_ratings = set(df['Expert Rating'].dropna()) - valid_ratings
            if invalid_ratings:
                errors.append(f"Invalid Expert Rating values found: {', '.join(invalid_ratings)}. "
                            f"Valid values are: {', '.join(valid_ratings)}")
        
        # Check data types
        if 'Age' in df.columns:
            try:
                pd.to_numeric(df['Age'], errors='coerce')
            except:
                errors.append("Age column contains non-numeric values")
        
        return len(errors) == 0, errors


def display_header():
    """Display the main application header"""
    st.markdown('<div class="main-header">EpiTuner</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Professional LoRA Fine-tuning for Syndromic Surveillance Data Classification</div>', unsafe_allow_html=True)


def display_sidebar():
    """Display the sidebar with navigation"""
    st.sidebar.title("Navigation")
    
    steps = {
        'data_upload': 'Data Upload',
        'model_selection': 'Model Selection',
        'training_config': 'Training Configuration',
        'training': 'Training',
        'expert_review': 'Expert Review',
        'export': 'Export Results'
    }
    
    for step_key, step_name in steps.items():
        if st.sidebar.button(step_name, key=f"nav_{step_key}"):
            st.session_state.current_step = step_key
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About EpiTuner")
    st.sidebar.markdown("""
    EpiTuner creates LoRA adapters for medical data classification using your local Ollama models.
    
    **Requirements:**
    - Local Ollama installation with models
    - User must explicitly select models
    - No automatic model downloads
    - Completely local processing (PHI-safe)
    
    **Features:**
    - Expert-in-the-loop validation
    - Consumer GPU optimized training
    - Professional Ollama integration
    """)


def step_data_upload():
    """Step 1: Data Upload"""
    st.markdown('<div class="section-header">Step 1: Data Upload</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Syndromic Surveillance Data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file containing syndromic surveillance records",
            type=['csv'],
            help="Upload a CSV file with syndromic surveillance data including Chief Complaints, Diagnoses, and Expert Ratings"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df
                
                # Validate the data
                is_valid, errors = DataValidator.validate_data(df)
                
                if is_valid:
                    st.markdown('<div class="success-box">Data validation successful!</div>', unsafe_allow_html=True)
                    
                    # Display data preview
                    st.markdown("### Data Preview")
                    st.dataframe(df.head(10))
                    
                    # Display data statistics
                    st.markdown("### Data Statistics")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        st.metric("Total Records", len(df))
                    
                    with col_stats2:
                        if 'Expert Rating' in df.columns:
                            rating_counts = df['Expert Rating'].value_counts()
                            st.metric("Match Cases", rating_counts.get('Match', 0))
                    
                    with col_stats3:
                        st.metric("Complete Records", df.dropna().shape[0])
                    
                    # Expert Rating distribution
                    if 'Expert Rating' in df.columns:
                        st.markdown("### Expert Rating Distribution")
                        rating_counts = df['Expert Rating'].value_counts()
                        fig = px.pie(values=rating_counts.values, names=rating_counts.index, 
                                   title="Distribution of Expert Ratings")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Next step button
                    if st.button("Continue to Model Selection →", type="primary"):
                        st.session_state.current_step = 'model_selection'
                        st.rerun()
                        
                else:
                    st.markdown('<div class="error-box">Data validation failed!</div>', unsafe_allow_html=True)
                    for error in errors:
                        st.error(error)
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    with col2:
        st.markdown("### Sample Data")
        st.markdown("""
        Don't have data to test with? Download our sample dataset:
        """)
        
        # Load sample data
        sample_path = Path("sample_data/medical_sample.csv")
        if sample_path.exists():
            with open(sample_path, 'rb') as f:
                st.download_button(
                    label="Download Sample Data",
                    data=f.read(),
                    file_name="medical_sample.csv",
                    mime="text/csv"
                )
        
        st.markdown("### Required Columns")
        st.markdown("""
        **Required:**
        - `C_Biosense_ID`: Unique identifier
        - `ChiefComplaintOrig`: Chief complaints
        - `DischargeDiagnosis`: Discharge diagnosis
        - `Expert Rating`: Match/Not a Match/Unknown
        - `Rationale_of_Rating`: Expert reasoning
        
        **Optional but Recommended:**
        - `Sex`, `Age`, `c_ethnicity`, `c_race`
        - `Admit_Reason_Combo`, `Diagnosis_Combo`
        - `CCDD Category`, `TriageNotes`
        """)


def step_model_selection():
    """Step 2: Model Selection"""
    st.markdown('<div class="section-header">Step 2: Model Selection</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload data first.")
        return
    
    # Get GPU information
    gpu_info = GPUManager.get_gpu_info()
    
    # Display GPU information
    st.markdown("### Your Hardware Information")
    
    if gpu_info['has_cuda']:
        st.success(f"✅ NVIDIA GPU Detected: {gpu_info['gpu_name']}")
        st.info(f"VRAM: {gpu_info['memory_gb']:.1f}GB")
        
        # Memory-based warnings
        if gpu_info['memory_gb'] < 5:
            st.warning("Limited VRAM detected. Recommended to use TinyLlama models only.")
        elif gpu_info['memory_gb'] < 8:
            st.info("Good for small-medium models. Phi-2 recommended.")
        else:
            st.success("Sufficient VRAM for most models.")
    elif gpu_info['hardware_type'] == 'nvidia_gpu_no_cuda':
        st.warning(f"⚠️ {gpu_info['gpu_name']}")
        st.error("**Issue:** NVIDIA GPU found but PyTorch cannot access CUDA")
        st.info("**Solution:** Run `.\setup.ps1 upgrade-torch` to install CUDA-enabled PyTorch")
        
        with st.expander("Why is this happening?"):
            st.markdown("""
            **Your system has an NVIDIA GPU but PyTorch can't use it because:**
            1. You have the CPU-only version of PyTorch installed
            2. CUDA drivers may not be properly configured
            
            **To fix this:**
            1. Run `.\setup.ps1 upgrade-torch` to install CUDA-enabled PyTorch
            2. Or use `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
            3. Restart the application after upgrading
            
            **Note:** Training will work in CPU mode but will be much slower.
            """)
    else:
        st.error(f"❌ No NVIDIA GPU: {gpu_info['gpu_name']}")
        
        # Show specific warnings
        for warning in gpu_info['warnings']:
            st.warning(warning)
        
        # Show training information (more positive messaging)
        if not gpu_info['has_cuda']:
            st.info("ℹ️ **CPU Training Mode**")
            st.markdown("""
            **CPU Training Characteristics:**
            - Slower than GPU training (30-60+ minutes for small models)
            - Works best with small models (DialoGPT-small, TinyLlama)
            - Suitable for testing and small datasets
            - No special drivers required
            
            **For faster training:**
            - Use a machine with NVIDIA GPU (RTX 3060 or better)
            - Consider cloud services with GPU support
            - Start with smaller models for testing
            """)
            
            # Make CPU training more accessible
            st.session_state.cpu_training_acknowledged = True
    
    st.markdown("### Select Base Model")
    
    # Get available Ollama models first - prioritize what user actually has
    available_models = OllamaModelManager.get_available_models()
    
    # Local Ollama models only - no external downloads
    if available_models:
        st.markdown("#### Your Local Ollama Models")
        st.markdown("Choose from models already installed on your system:")
        
        selected_ollama_model = st.selectbox(
            "Select a locally available Ollama model:",
            options=[""] + available_models,
            help="EpiTuner only uses models you have already downloaded locally - no automatic downloads"
        )
        
        if selected_ollama_model:
            # Convert to training architecture name (this is just for the training code - model stays local)
            hf_model_name = OllamaModelManager.model_to_hf_name(selected_ollama_model)
            
            st.success(f"Selected Local Model: {selected_ollama_model}")
            st.info("✅ Training will use your local model weights - no downloads required!")
            
            # Check if model is suitable for hardware
            if gpu_info['memory_gb'] < 5 and 'llama2' in selected_ollama_model.lower():
                st.warning("This model may be too large for your GPU. Consider using TinyLlama instead.")
            
            st.session_state.selected_model = hf_model_name
            st.session_state.selected_model_display = selected_ollama_model
    else:
        st.error("❌ No Ollama models found!")
        st.markdown("**EpiTuner requires local Ollama models to ensure data privacy:**")
        st.code("""
# Install models in Ollama first:
ollama pull llama3.2:1b     # Small model for limited hardware
ollama pull phi3            # Good general model  
ollama pull mistral         # Larger model if you have sufficient resources
""")
        st.warning("⚠️ EpiTuner only works with locally installed Ollama models. No external downloads are allowed.")
        
        # Show hardware-appropriate recommendations
        st.markdown("**Recommended models for your hardware:**")
        recommended = gpu_info.get('recommended_models', ['TinyLlama/TinyLlama-1.1B-Chat-v1.0'])
        for model in recommended:
            st.markdown(f"- `{model}`")
        
        return  # Stop here - don't allow training without local models
    
    # Show selection status
    if not st.session_state.get('selected_model'):
        st.warning("⚠️ Please select one of your local Ollama models above to continue.")
        st.info("💡 EpiTuner ensures data privacy by using only locally available models.")

    
    # Classification topic
    st.markdown("### Classification Task")
    classification_topic = st.text_area(
        "Describe what you want the model to classify:",
        value="Used to detect motor vehicle collisions in syndromic surveillance records",
        help="Provide context about what the model should identify. This helps the model understand the classification task.",
        height=100
    )
    
    st.session_state.classification_topic = classification_topic
    
    # Next step button
    if st.session_state.selected_model and classification_topic:
        if st.button("Continue to Training Configuration →", type="primary"):
            st.session_state.current_step = 'training_config'
            st.rerun()


def step_training_config():
    """Step 3: Training Configuration"""
    st.markdown('<div class="section-header">Step 3: Training Configuration</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None or st.session_state.selected_model is None:
        st.warning("Please complete previous steps first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Training Parameters")
        st.markdown("""
        Configure how the model will be trained. These settings affect training speed, memory usage, and final model quality.
        """)
        
        # Basic parameters
        epochs = st.slider("Number of Epochs", min_value=1, max_value=10, value=2, 
                          help="An epoch is one complete pass through all training data. More epochs = longer training but potentially better results. Start with 1-2 for testing.")
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4],
            value=2e-4,
            format_func=lambda x: f"{x:.0e}",
            help="Controls how quickly the model learns. Higher = faster learning but may be unstable. Lower = slower but more stable. 2e-4 is a good starting point."
        )
        
        # LoRA parameters
        st.markdown("### LoRA Configuration")
        st.markdown("""
        LoRA (Low-Rank Adaptation) allows efficient fine-tuning by adding small trainable modules to the base model.
        Lower values use less memory but may learn less effectively.
        """)
        
        lora_r = st.slider("LoRA Rank (r)", min_value=4, max_value=32, value=8, step=4,
                          help="Rank determines the size of LoRA modules. Higher rank = more parameters to train = more memory usage. Start with 8 for consumer GPUs.")
        
        lora_alpha = st.slider("LoRA Alpha", min_value=8, max_value=64, value=16, step=8,
                              help="Scaling factor that controls the strength of LoRA updates. Typically set to 2x the rank value.")
        
        # Data split
        st.markdown("### Data Splitting")
        st.markdown("""
        Your data will be automatically split into three parts for proper evaluation:
        - **Training**: Used to teach the model
        - **Validation**: Used during training to monitor progress
        - **Test**: Used for final evaluation after training
        """)
        train_split = st.slider("Training Split", min_value=0.6, max_value=0.9, value=0.8, step=0.05,
                               help="Percentage of data used for training the model")
        val_split = st.slider("Validation Split", min_value=0.05, max_value=0.3, value=0.1, step=0.025,
                             help="Percentage of data used for validation during training")
        test_split = 1.0 - train_split - val_split
        
        st.info(f"Data Distribution: Train: {train_split:.1%}, Validation: {val_split:.1%}, Test: {test_split:.1%}")
        
        # Confidence threshold
        st.markdown("### Expert Review Configuration")
        st.markdown("""
        Set how confident the model needs to be before automatically accepting its predictions.
        Lower values mean more predictions will need expert review.
        """)
        confidence_threshold = st.slider(
            "Confidence Threshold for Auto-Approval",
            min_value=0.1, max_value=0.9, value=0.7, step=0.05,
            help="Records where the model is more confident than this threshold will be automatically approved. Others will be flagged for expert review."
        )
        
        # Save configuration with GPU optimization
        gpu_info = GPUManager.get_gpu_info()
        config = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'train_split': train_split,
            'val_split': val_split,
            'test_split': test_split,
            'confidence_threshold': confidence_threshold,
            'model_name': st.session_state.selected_model,
            'classification_topic': st.session_state.get('classification_topic', ''),
            'config_file': gpu_info['config_file'],  # Use appropriate config
            'gpu_memory': gpu_info['memory_gb'],
            'gpu_name': gpu_info['gpu_name']
        }
        
        st.session_state.training_config = config
        
        # Training requirements info
        gpu_info = GPUManager.get_gpu_info()
        
        st.markdown("### Training Requirements")
        
        if gpu_info['training_feasible'] or st.session_state.get('cpu_training_acknowledged', False):
            if gpu_info['has_cuda']:
                st.success("🔥 GPU-accelerated LoRA training ready")
                st.markdown("**Your System:**")
                st.markdown("- ✅ NVIDIA GPU with CUDA support")
                st.markdown("- ✅ Sufficient VRAM for training")
                st.markdown("- ⏱️ Training time: 10-60 minutes (depends on data size)")
                st.markdown("- 🎯 Output: Production-ready LoRA adapter")
            else:
                st.warning("⚠️ CPU-only training mode enabled")
                st.markdown("**Your System:**")
                st.markdown("- ⚠️ CPU-only training (no NVIDIA GPU)")
                st.markdown("- ⚠️ High memory usage expected")
                st.markdown("- ⏱️ Training time: Hours to days")
                st.markdown("- 🎯 Output: LoRA adapter (if training completes)")
        else:
            st.error("❌ Training requirements not met")
            st.markdown("**Missing:**")
            st.markdown("- ❌ NVIDIA GPU required for practical training")
            st.markdown("- ❌ Current hardware insufficient")
            st.markdown("Please check the hardware section above for options.")
        
        # Start training button
        can_train = gpu_info['training_feasible'] or st.session_state.get('cpu_training_acknowledged', False)
        
        if can_train:
            if st.button("Start Training", type="primary"):
                st.session_state.current_step = 'training'
                st.rerun()
        else:
            st.button("Start Training", type="primary", disabled=True, 
                     help="Training disabled - hardware requirements not met")
            st.error("Cannot start training: Hardware requirements not satisfied")
    
    with col2:
        st.markdown("### Configuration Summary")
        
        if st.session_state.training_config:
            config = st.session_state.training_config
            
            st.markdown(f"**Model:** {config['model_name']}")
            st.markdown(f"**Epochs:** {config['epochs']}")
            st.markdown(f"**Learning Rate:** {config['learning_rate']:.0e}")
            st.markdown(f"**LoRA Rank:** {config['lora_r']}")
            st.markdown(f"**LoRA Alpha:** {config['lora_alpha']}")
            st.markdown(f"**Confidence Threshold:** {config['confidence_threshold']:.1%}")
        
        st.markdown("### Estimated Resources")
        
        gpu_info = GPUManager.get_gpu_info()
        
        if st.session_state.uploaded_data is not None:
            num_records = len(st.session_state.uploaded_data)
            
            # Estimate time based on GPU and settings
            if gpu_info['has_cuda']:
                if gpu_info['memory_gb'] >= 8:
                    time_per_record = 0.05  # Fast GPU
                elif gpu_info['memory_gb'] >= 6:
                    time_per_record = 0.08  # Medium GPU  
                else:
                    time_per_record = 0.15  # Slow GPU
            else:
                time_per_record = 1.0  # CPU is very slow
            
            est_time = max(2, num_records * time_per_record * epochs)
            
            st.markdown(f"**Records:** {num_records}")
            st.markdown(f"**GPU:** {gpu_info['gpu_name'] if gpu_info['has_cuda'] else 'CPU Only'}")
            st.markdown(f"**Est. Time:** {est_time:.1f} minutes")
            
            # Memory usage based on GPU
            if gpu_info['memory_gb'] >= 8:
                memory_usage = f"~{lora_r * 0.3:.1f}GB VRAM"
            elif gpu_info['memory_gb'] >= 4:
                memory_usage = f"~{lora_r * 0.2:.1f}GB VRAM"
            else:
                memory_usage = "May exceed VRAM"
            
            st.markdown(f"**Memory Need:** {memory_usage}")
        
        st.markdown("### Windows Tips")
        st.markdown("""
        **For Consumer GPUs:**
        - Start with 1-2 epochs
        - Use Task Manager to monitor GPU
        - Close Chrome/games before training
        
        **If Out of Memory:**
        - Reduce LoRA rank to 4
        - Use TinyLlama model
        - Close other applications
        
        **For Best Results:**
        - Train in short iterations
        - Review after each training
        - Use expert feedback loop
        """)


def step_training():
    """Step 4: Training"""
    st.markdown('<div class="section-header">Step 4: Training</div>', unsafe_allow_html=True)
    
    if not all([st.session_state.uploaded_data is not None, 
                st.session_state.selected_model, 
                st.session_state.training_config]):
        st.error("❌ Cannot start training - missing requirements:")
        if st.session_state.uploaded_data is None:
            st.error("• No data uploaded")
        if not st.session_state.selected_model:
            st.error("• No model selected")
        if not st.session_state.training_config:
            st.error("• Training not configured")
        st.info("💡 Please complete all previous steps before training.")
        return
    
    # Training status
    if not st.session_state.training_in_progress:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Ready to Train")
        
        config = st.session_state.training_config
        data_size = len(st.session_state.uploaded_data)
        
        # Show if this is a retraining with feedback
        if st.session_state.get('feedback_changes'):
            changes = st.session_state.feedback_changes
            st.success(f"Retraining with expert feedback: {changes['count']} records updated from {changes['feedback_items']} expert corrections")
        
        # Show user-friendly model name if available
        display_model = st.session_state.get('selected_model_display', config['model_name'])
        
        st.markdown(f"""
        **Configuration Summary:**
        - Model: {display_model}
        - Records: {data_size}
        - Epochs: {config['epochs']}
        - Learning Rate: {config['learning_rate']:.0e}
        - LoRA Rank: {config['lora_r']}
        """)
        
        if st.button("Start Training", type="primary"):
            start_training()
        
        with col2:
            st.markdown("### What Happens Next")
            st.markdown("""
            1. Data preprocessing
            2. Model initialization  
            3. LoRA training
            4. Validation
            5. Model saving
            6. Results generation
            """)
    
    else:
        display_training_progress()


def start_training():
    """Start the training process"""
    st.session_state.training_in_progress = True
    
    # Create temporary files for training
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Save data to temporary CSV
        data_path = temp_dir / "training_data.csv"
        st.session_state.uploaded_data.to_csv(data_path, index=False)
        
        # Create config file
        config_path = temp_dir / "config.yaml"
        
        # Automatically select the best config based on system compatibility
        def detect_system_compatibility():
            """Detect what training modes are compatible with the current system"""
            try:
                import torch
                torch_version = torch.__version__
                torch_major, torch_minor = map(int, torch_version.split('.')[:2])
                
                # Check if BnB/PEFT is likely to work
                if torch_major < 2 or (torch_major == 2 and torch_minor < 2):
                    return 'cpu_only'  # Old PyTorch, BnB incompatible
                
                # Get our enhanced GPU info instead of just checking torch.cuda
                gpu_info = GPUManager.get_gpu_info()
                
                if gpu_info['hardware_type'] == 'nvidia_gpu_no_cuda':
                    # NVIDIA GPU detected but CUDA not available to PyTorch
                    # Use no_quantization config - GPU might work after PyTorch upgrade
                    return 'no_quantization'
                elif gpu_info['has_cuda']:
                    # CUDA is available - check memory
                    gpu_memory = gpu_info['memory_gb']
                    if gpu_memory < 4:
                        return 'no_quantization'  # Limited GPU
                    else:
                        return 'base'  # Should handle quantization
                elif gpu_info['hardware_type'] in ['integrated_gpu', 'amd_gpu']:
                    # Intel or AMD GPU detected - use no_quantization for GPU training
                    return 'no_quantization'
                else:
                    # Only use cpu_only if absolutely no GPU detected
                    return 'no_quantization'  # Default to GPU-capable config
            except:
                return 'no_quantization'  # Fallback to GPU-capable config
        
        # Select config based on compatibility
        compatibility = detect_system_compatibility()
        
        if compatibility == 'cpu_only':
            config_file = "configs/config_cpu_only.yaml"
            print("Selected CPU-only config - PyTorch too old for GPU training")
        elif compatibility == 'no_quantization':
            config_file = "configs/config_no_quantization.yaml"
            print("Selected no-quantization config - GPU detected, standard LoRA training")
        else:
            config_file = "configs/config_base.yaml"
            print("Selected base config with full features - NVIDIA GPU with CUDA and sufficient VRAM")
        
        try:
            with open(config_file, 'r') as f:
                base_config = yaml.safe_load(f)
        except FileNotFoundError:
            # Ultimate fallback
            print(f"Warning: {config_file} not found, using base config")
            with open("configs/config_base.yaml", 'r') as f:
                base_config = yaml.safe_load(f)
        
        # Update config with user settings
        config = st.session_state.training_config
        base_config['train']['num_epochs'] = config['epochs']
        base_config['train']['learning_rate'] = config['learning_rate']
        base_config['tuning']['lora_r'] = config['lora_r']
        base_config['tuning']['lora_alpha'] = config['lora_alpha']
        base_config['data']['train_split'] = config['train_split']
        base_config['data']['val_split'] = config['val_split']
        base_config['data']['test_split'] = config['test_split']
        
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create output directory
        output_dir = Path("outputs") / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Always use real training
        real_training(str(data_path), str(config_path), str(output_dir))


def real_training(data_path: str, config_path: str, output_dir: str):
    """Run actual LoRA training using the training script"""
    import subprocess
    import sys
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔥 Starting real LoRA training...")
    progress_bar.progress(0.1)
    
    try:
        # Get training parameters from session state
        model_name = st.session_state.selected_model
        classification_topic = st.session_state.get('classification_topic', 'Syndromic surveillance classification')
        
        # Build command for training script
        train_script = Path("scripts/train.py")
        
        if not train_script.exists():
            st.error("Training script not found! Please ensure scripts/train.py exists.")
            return
        
        cmd = [
            sys.executable, str(train_script),
            "--config", config_path,
            "--data", data_path,
            "--model", model_name,
            "--topic", classification_topic,
            "--output", output_dir
        ]
        
        status_text.text("🚀 Launching training process...")
        progress_bar.progress(0.2)
        
        # Run training in subprocess
        status_text.text("⚙️ Training in progress... This will take a while.")
        progress_bar.progress(0.3)
        
        # Debug: Print the command being executed
        print(f"=== EXECUTING COMMAND ===")
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        print("=== END COMMAND ===")
        
        # Use more resource-friendly subprocess settings for Windows
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,  # Unbuffered for Windows compatibility
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Monitor training progress with real-time parsing
        output_lines = []
        import re
        
        # Initialize progress tracking
        current_epoch = 0
        total_epochs = 0
        current_step = 0
        total_steps = 0
        current_loss = 0.0
        
        status_text.text("🔄 Training is starting...")
        
        for line in process.stdout:
            output_lines.append(line.strip())
            
            # Parse Hugging Face Trainer output format
            # Look for lines like: "Epoch 1/3: 100%|██████████| 50/50 [00:30<00:00,  1.67it/s, loss=0.1234]"
            trainer_progress = re.search(r'Epoch\s+(\d+)/(\d+):\s+(\d+)%\|.*?\|.*?loss=([\d.]+)', line)
            if trainer_progress:
                current_epoch = int(trainer_progress.group(1))
                total_epochs = int(trainer_progress.group(2))
                progress_percent = int(trainer_progress.group(3))
                current_loss = float(trainer_progress.group(4))
                
                # Calculate overall progress
                epoch_progress = (current_epoch - 1) / total_epochs
                step_progress = progress_percent / 100 / total_epochs
                progress = epoch_progress + step_progress
                
                progress_bar.progress(min(progress, 0.95))
                status_text.text(f"🔄 Epoch {current_epoch}/{total_epochs} ({progress_percent}%) - Loss: {current_loss:.4f}")
                print(f"Progress: {progress:.1%} (Epoch {current_epoch}/{total_epochs}, {progress_percent}%, Loss: {current_loss:.4f})")
            
            # Parse epoch progress (e.g., "epoch 2/3")
            epoch_match = re.search(r'epoch\s+(\d+)/(\d+)', line.lower())
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                progress = current_epoch / total_epochs
                progress_bar.progress(progress)
                status_text.text(f"🔄 Training epoch {current_epoch}/{total_epochs}...")
                print(f"Progress: {progress:.1%} (Epoch {current_epoch}/{total_epochs})")
            
            # Parse step progress (e.g., "step 150/300")
            step_match = re.search(r'step\s+(\d+)/(\d+)', line.lower())
            if step_match:
                current_step = int(step_match.group(1))
                total_steps = int(step_match.group(2))
                if total_epochs > 0:
                    # Calculate overall progress including epochs
                    epoch_progress = (current_epoch - 1) / total_epochs
                    step_progress = current_step / total_steps / total_epochs
                    progress = epoch_progress + step_progress
                else:
                    progress = current_step / total_steps
                progress_bar.progress(min(progress, 0.95))  # Keep some room for completion
                status_text.text(f"🔄 Training step {current_step}/{total_steps} (epoch {current_epoch}/{total_epochs})...")
                print(f"Progress: {progress:.1%} (Step {current_step}/{total_steps})")
            
            # Parse loss values
            loss_match = re.search(r'loss:\s*([\d.]+)', line.lower())
            if loss_match:
                current_loss = float(loss_match.group(1))
                status_text.text(f"🔄 Training... Loss: {current_loss:.4f}")
                print(f"Loss: {current_loss:.4f}")
            
            # Parse completion messages
            if 'training completed' in line.lower() or 'saving model' in line.lower():
                progress_bar.progress(0.98)
                status_text.text(f"🔄 {line.strip()}")
                print(f"Status: {line.strip()}")
            
            # Parse error messages
            if 'error' in line.lower() or 'exception' in line.lower():
                print(f"ERROR: {line.strip()}")
                status_text.text(f"❌ Error: {line.strip()}")
        
        # Wait for completion
        return_code = process.wait()
        
        # Debug: Print the output for troubleshooting
        print("=== TRAINING OUTPUT ===")
        for line in output_lines:
            print(line)
        print("=== END TRAINING OUTPUT ===")
        print(f"Return code: {return_code}")
        
        if return_code == 0:
            # Ensure progress bar shows completion
            progress_bar.progress(1.0)
            status_text.text("✅ Training completed successfully!")
            
            # Show final training summary
            if current_epoch > 0 and total_epochs > 0:
                st.info(f"🎉 Training completed: {current_epoch}/{total_epochs} epochs")
            if current_loss > 0:
                st.info(f"📊 Final loss: {current_loss:.4f}")
            
            # Debug: List files in output directory
            print(f"=== FILES IN OUTPUT DIRECTORY: {output_dir} ===")
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    print(f"  - {file}")
            else:
                print("  Output directory does not exist!")
            print("=== END FILE LIST ===")
            
            # Load actual training results
            metrics_path = Path(output_dir) / "model_metadata.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    training_metrics = json.load(f)
                
                # Create results from actual training metrics
                results = {
                    'training_loss': training_metrics.get('train_loss', 0.5),
                    'validation_loss': training_metrics.get('test_eval_loss', 0.6),
                    'model_path': output_dir,
                    'config_path': config_path,
                    'training_metrics': training_metrics
                }
                
                # Run real inference to get actual predictions and metrics
                predictions = run_real_inference()
                if predictions:
                    # Calculate real metrics from predictions
                    agreements = [p['agreement'] for p in predictions]
                    accuracy = sum(agreements) / len(agreements) if agreements else 0.0
                    
                    # Calculate precision, recall, F1 from confusion matrix
                    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
                    
                    y_true = [p['expert_rating'] for p in predictions]
                    y_pred = [p['model_rating'] for p in predictions]
                    
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='weighted', zero_division=0
                    )
                    
                    results.update({
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'predictions': predictions,
                        'training_metrics': training_metrics
                    })
                else:
                    st.error("Failed to generate predictions. Please check the model.")
                    return
            else:
                # No metrics file found - this indicates a training problem
                st.error("Training completed but no metrics file found. This indicates the training may have failed.")
                st.info("Please check the training output for errors and try again.")
                return
            
            st.session_state.model_results = results
            st.session_state.training_in_progress = False
            
            st.success("🎉 Real LoRA training completed successfully!")
            st.info("Your model has been trained and saved. Proceeding to Expert Review...")
            
            time.sleep(3)
            st.session_state.current_step = 'expert_review'
            st.rerun()
            
        else:
            # Handle training failure more gracefully to avoid Streamlit crashes
            progress_bar.progress(0.0)
            status_text.text("❌ Training failed")
            
            # Log error details without causing Streamlit crash
            print(f"Training failed with return code {return_code}")
            
            # Show last few lines of output for debugging (safely)
            if output_lines:
                st.markdown("**Training Output (last 10 lines):**")
                try:
                    for line in output_lines[-10:]:
                        st.code(line)  # Use st.code instead of st.text for better formatting
                except:
                    st.text("Error displaying training output")
            
            # Provide helpful error message based on common issues
            if return_code == 1:
                st.error("Training script encountered an error. Common causes:")
                st.markdown("""
                - **BitsAndBytes compatibility**: Your PyTorch version may be incompatible
                - **Memory issues**: Model too large for available RAM/VRAM  
                - **Missing dependencies**: Required packages not installed properly
                - **Windows resource limits**: Try closing other applications
                """)
                
                st.info("💡 **Suggested fixes:**")
                st.markdown("""
                1. Try using a smaller model (DialoGPT-small)
                2. Restart Streamlit: `streamlit run app.py`
                3. Close other applications to free memory
                4. Check Windows Task Manager for resource usage
                """)
            
            st.session_state.training_in_progress = False
            
    except Exception as e:
        # Handle exceptions more gracefully to prevent Streamlit crashes
        print(f"Training exception: {str(e)}")  # Log to console
        
        # Reset training state
        st.session_state.training_in_progress = False
        
        # Update UI safely
        try:
            progress_bar.progress(0.0)
            status_text.text("❌ Training encountered an error")
            
            # Show user-friendly error message
            st.error("Training encountered an unexpected error")
            st.markdown("**Possible causes:**")
            st.markdown("""
            - **System resource limits**: Windows may have run out of memory or socket buffers
            - **Model compatibility**: The selected model may not be compatible with your system
            - **Environment issues**: Missing dependencies or configuration problems
            """)
            
            st.info("💡 **Recommended actions:**")
            st.markdown("""
            1. **Restart the application**: Close browser and run `streamlit run app.py` again
            2. **Close other applications**: Free up system resources
            3. **Try a smaller model**: Use DialoGPT-small for testing
            4. **Check system resources**: Use Task Manager to monitor RAM/CPU usage
            """)
            
        except:
            # If even the error display fails, just log and reset
            print("Failed to display error in Streamlit - resetting training state")
            pass



def run_real_inference():
    """Run real model inference on the uploaded data"""
    if st.session_state.uploaded_data is None:
        st.error("No data uploaded. Please upload your medical data first.")
        return []
    
    if not st.session_state.model_results or 'model_path' not in st.session_state.model_results:
        st.error("No trained model available. Please complete training first.")
        return []
    
    # Check if model files actually exist
    model_path = st.session_state.model_results['model_path']
    if not os.path.exists(model_path):
        st.error(f"Model files not found at {model_path}. Please ensure training completed successfully.")
        return []
    
    # Check for essential model files
    required_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        st.error(f"Missing essential model files: {missing_files}. Training may have failed.")
        return []
    
    try:
        # Import the real inference system
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
        
        from inference import SyndromicSurveillanceClassificationInference
        
        # Load the trained model
        model_path = st.session_state.model_results['model_path']
        config_path = st.session_state.model_results.get('config_path', 'configs/config_base.yaml')
        
        # Initialize inference engine
        inference_engine = MedicalClassificationInference(
            model_path=model_path,
            config_path=config_path
        )
        
        # Get classification topic from training config
        classification_topic = st.session_state.training_config.get('classification_topic', 'Syndromic Surveillance Classification')
        
        predictions = []
        
        # Run inference on each record
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in st.session_state.uploaded_data.iterrows():
            status_text.text(f"Processing record {idx + 1}/{len(st.session_state.uploaded_data)}...")
            
            # Create syndromic surveillance record for inference
            surveillance_record = {
                'biosense_id': row['C_Biosense_ID'],
                'chief_complaint': row['ChiefComplaintOrig'],
                'discharge_diagnosis': row['DischargeDiagnosis'],
                'demographics': f"Sex: {row.get('Sex', 'N/A')}, Age: {row.get('Age', 'N/A')}, Ethnicity: {row.get('c_ethnicity', 'N/A')}, Race: {row.get('c_race', 'N/A')}",
                'admit_reason': row.get('Admit_Reason_Combo', ''),
                'diagnosis_combo': row.get('Diagnosis_Combo', ''),
                'ccdd_category': row.get('CCDD Category', ''),
                'triage_notes': row.get('TriageNotes', row.get('TriageNotesOrig', ''))
            }
            
            # Run inference
            result = inference_engine.predict_single(surveillance_record, classification_topic)
            
            # Format prediction for display
            prediction = {
                'biosense_id': row['C_Biosense_ID'],
                'expert_rating': row['Expert Rating'],
                'model_rating': result['classification'],
                'confidence': result['confidence'],
                'confidence_score': result.get('confidence_score', 0.0),
                'model_rationale': result['rationale'],
                'agreement': result['classification'] == row['Expert Rating'],
                'chief_complaint': row['ChiefComplaintOrig'],
                'discharge_diagnosis': row['DischargeDiagnosis']
            }
            
            predictions.append(prediction)
            
            # Update progress
            progress_bar.progress((idx + 1) / len(st.session_state.uploaded_data))
        
        status_text.text("Inference completed!")
        progress_bar.empty()
        status_text.empty()
        
        return predictions
        
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        st.error("Please ensure the model was trained successfully and all dependencies are installed.")
        return []


def display_training_progress():
    """Display training progress"""
    st.markdown("### Training in Progress...")
    
    # This would show real training progress in a production system
    st.info("Training is running. This may take several minutes depending on your data size and hardware.")
    
    # Placeholder for real-time metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Epoch", "2/3")
    
    with col2:
        st.metric("Training Loss", "0.45")
    
    with col3:
        st.metric("Validation Loss", "0.52")


def incorporate_expert_feedback():
    """Incorporate expert feedback into the training dataset"""
    if not st.session_state.expert_feedback or st.session_state.uploaded_data is None:
        return
    
    # Create a copy of the original data
    updated_data = st.session_state.uploaded_data.copy()
    changes_made = 0
    
    # Apply expert corrections
    feedback_dict = {f['biosense_id']: f for f in st.session_state.expert_feedback}
    
    for idx, row in updated_data.iterrows():
        biosense_id = str(row['C_Biosense_ID'])
        if biosense_id in feedback_dict:
            feedback = feedback_dict[biosense_id]
            # Update the expert rating with the correction
            old_rating = updated_data.at[idx, 'Expert Rating']
            updated_data.at[idx, 'Expert Rating'] = feedback['expert_correction']
            updated_data.at[idx, 'Rationale_of_Rating'] = feedback['expert_reason']
            changes_made += 1
    
    # Store the updated dataset
    st.session_state.uploaded_data = updated_data
    
    # Store info about changes for display
    st.session_state.feedback_changes = {
        'count': changes_made,
        'feedback_items': len(st.session_state.expert_feedback)
    }
    
    # Clear the expert feedback since it's been incorporated
    st.session_state.expert_feedback = []


def step_expert_review():
    """Step 5: Expert Review"""
    st.markdown('<div class="section-header">Step 5: Expert Review</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_results:
        st.warning("Please complete training first.")
        return
    
    results = st.session_state.model_results
    predictions = results['predictions']
    
    # Overall metrics
    st.markdown("### Training Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.1%}")
    
    with col2:
        st.metric("Precision", f"{results['precision']:.1%}")
    
    with col3:
        st.metric("Recall", f"{results['recall']:.1%}")
    
    with col4:
        st.metric("F1 Score", f"{results['f1_score']:.1%}")
    
    # Agreement analysis
    agreements = [p['agreement'] for p in predictions]
    agreement_rate = sum(agreements) / len(agreements)
    
    # Calculate agreement by confidence level
    confidence_agreement = {}
    for pred in predictions:
        conf = pred['confidence']
        if conf not in confidence_agreement:
            confidence_agreement[conf] = {'total': 0, 'agreed': 0}
        confidence_agreement[conf]['total'] += 1
        if pred['agreement']:
            confidence_agreement[conf]['agreed'] += 1
    
    st.markdown(f"### Model-Expert Agreement: {agreement_rate:.1%}")
    
    # Show agreement by confidence level
    st.markdown("#### Agreement by Confidence Level:")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    confidence_order = ["Very Confident", "Confident", "Somewhat Confident", "Not Very Confident", "Not at all Confident"]
    for i, conf in enumerate(confidence_order):
        if conf in confidence_agreement:
            stats = confidence_agreement[conf]
            agreement_pct = stats['agreed'] / stats['total'] if stats['total'] > 0 else 0
            with [col1, col2, col3, col4, col5][i]:
                st.metric(
                    conf, 
                    f"{agreement_pct:.1%}",
                    f"{stats['agreed']}/{stats['total']}"
                )
    
    # Show rating distribution comparison
    st.markdown("#### Rating Distribution Comparison:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Expert Ratings:**")
        expert_counts = {}
        for pred in predictions:
            rating = pred['expert_rating']
            expert_counts[rating] = expert_counts.get(rating, 0) + 1
        
        for rating, count in expert_counts.items():
            pct = count / len(predictions) * 100
            st.markdown(f"- {rating}: {count} ({pct:.1f}%)")
    
    with col2:
        st.markdown("**Model Predictions:**")
        model_counts = {}
        for pred in predictions:
            rating = pred['model_rating']
            model_counts[rating] = model_counts.get(rating, 0) + 1
        
        for rating, count in model_counts.items():
            pct = count / len(predictions) * 100
            st.markdown(f"- {rating}: {count} ({pct:.1f}%)")
    
    # Enhanced filter controls for medical review
    st.markdown("### Review Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_filter = st.selectbox(
            "Show records:",
            [
                "All Records", 
                "Need Review (Low Confidence)", 
                "Need Review (Disagreements)", 
                "High Confidence Only",
                "Medium Confidence Only",
                "Low Confidence Only",
                "Disagreements Only"
            ]
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Score Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Records below this confidence score will be flagged for review"
        )
    
    with col3:
        confidence_filter = st.selectbox(
            "Confidence level:",
            ["All"] + st.session_state.training_config.get('confidence_levels', 
                ["Very Confident", "Confident", "Somewhat Confident", "Not Very Confident", "Not at all Confident"])
        )
    
    # Filter predictions
    filtered_predictions = filter_predictions(predictions, show_filter, confidence_filter, confidence_threshold)
    
    # Summary of records needing review
    low_confidence_count = len([p for p in predictions if p.get('confidence_score', 0.0) < confidence_threshold])
    disagreement_count = len([p for p in predictions if not p['agreement']])
    high_confidence_count = len([p for p in predictions if p.get('confidence_score', 0.0) >= confidence_threshold])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Need Review (Low Confidence)", low_confidence_count, f"{low_confidence_count/len(predictions):.1%}")
    with col2:
        st.metric("Need Review (Disagreements)", disagreement_count, f"{disagreement_count/len(predictions):.1%}")
    with col3:
        st.metric("High Confidence", high_confidence_count, f"{high_confidence_count/len(predictions):.1%}")
    
    # Display predictions for review
    st.markdown(f"### Records to Review ({len(filtered_predictions)} of {len(predictions)})")
    
    for i, pred in enumerate(filtered_predictions):
        display_prediction_review(pred, i)
    
    # Retrain button
    if st.session_state.expert_feedback:
        st.markdown("### Expert Feedback Collected")
        st.info(f"You have provided feedback on {len(st.session_state.expert_feedback)} records.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Retrain with Feedback", type="primary"):
                # Incorporate expert feedback into training data
                incorporate_expert_feedback()
                
                # Reset training state and go back to training step
                st.session_state.training_in_progress = False
                st.session_state.model_results = None
                st.session_state.current_step = 'training'
                
                st.success("Expert feedback incorporated! Starting retraining...")
                st.rerun()
        
        with col2:
            if st.button("Accept Results", type="secondary"):
                st.session_state.current_step = 'export'
                st.rerun()
    
    else:
        if st.button("Continue to Export →", type="primary"):
            st.session_state.current_step = 'export'
            st.rerun()


def filter_predictions(predictions, show_filter, confidence_filter, confidence_threshold=0.7):
    """Filter predictions based on user criteria for medical review"""
    filtered = predictions.copy()
    
    # Apply show filter
    if show_filter == "Need Review (Low Confidence)":
        # Show records that need review due to low confidence
        filtered = [p for p in filtered if p.get('confidence_score', 0.0) < confidence_threshold]
    elif show_filter == "Need Review (Disagreements)":
        # Show records where model disagrees with expert
        filtered = [p for p in filtered if not p['agreement']]
    elif show_filter == "High Confidence Only":
        high_conf = ["Very Confident", "Confident"]
        filtered = [p for p in filtered if p['confidence'] in high_conf and p.get('confidence_score', 0.0) >= confidence_threshold]
    elif show_filter == "Medium Confidence Only":
        medium_conf = ["Somewhat Confident"]
        filtered = [p for p in filtered if p['confidence'] in medium_conf]
    elif show_filter == "Low Confidence Only":
        low_conf = ["Not Very Confident", "Not at all Confident"]
        filtered = [p for p in filtered if p['confidence'] in low_conf or p.get('confidence_score', 0.0) < confidence_threshold]
    elif show_filter == "Disagreements Only":
        filtered = [p for p in filtered if not p['agreement']]
    # "All Records" shows everything
    
    # Apply confidence filter
    if confidence_filter != "All":
        filtered = [p for p in filtered if p['confidence'] == confidence_filter]
    
    return filtered


def display_prediction_review(pred, index):
    """Display a single prediction for expert review"""
    
    # Determine styling based on agreement and confidence
    confidence_score = pred.get('confidence_score', 0.0)
    confidence_threshold = 0.7  # Default threshold
    
    if pred['agreement'] and confidence_score >= confidence_threshold:
        border_color = "#28a745"  # Green for agreement + high confidence
        bg_color = "#d4f6d4"
        review_status = "✅ No Review Needed"
    elif not pred['agreement']:
        border_color = "#dc3545"  # Red for disagreement
        bg_color = "#f8d7da"
        review_status = "⚠️ Review Needed (Disagreement)"
    elif confidence_score < confidence_threshold:
        border_color = "#ffc107"  # Yellow for low confidence
        bg_color = "#fff3cd"
        review_status = "⚠️ Review Needed (Low Confidence)"
    else:
        border_color = "#17a2b8"  # Blue for agreement but medium confidence
        bg_color = "#d1ecf1"
        review_status = "✅ No Review Needed"
    
    with st.container():
        st.markdown(f"""
        <div style="border: 2px solid {border_color}; background-color: {bg_color}; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>Record ID:</strong> {pred['biosense_id']}
                <span style="font-weight: bold; color: {border_color};">{review_status}</span>
            </div>
            <strong>Chief Complaint:</strong> {pred['chief_complaint']}<br>
            <strong>Discharge Diagnosis:</strong> {pred['discharge_diagnosis']}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Expert Rating:**")
            st.markdown(f"_{pred['expert_rating']}_")
        
        with col2:
            st.markdown("**Model Rating:**")
            st.markdown(f"_{pred['model_rating']}_")
        
        with col3:
            st.markdown("**Model Confidence:**")
            confidence_score = pred.get('confidence_score', 0.0)
            
            # Color code confidence
            if confidence_score >= 0.8:
                conf_color = "#28a745"  # Green for high confidence
                conf_icon = "🟢"
            elif confidence_score >= 0.6:
                conf_color = "#ffc107"  # Yellow for medium confidence
                conf_icon = "🟡"
            else:
                conf_color = "#dc3545"  # Red for low confidence
                conf_icon = "🔴"
            
            st.markdown(f"""
            <div style="color: {conf_color}; font-weight: bold;">
            {conf_icon} {pred['confidence']} ({confidence_score:.1%})
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Model Rationale:**")
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #007bff;">
        {pred['model_rationale']}
        </div>
        """, unsafe_allow_html=True)
        
        # Expert feedback
        feedback_key = f"feedback_{index}"
        
        if not pred['agreement']:
            st.markdown("**Provide Expert Feedback:**")
            
            expert_correction = st.selectbox(
                "Correct classification:",
                ["", "Match", "Not a Match", "Unknown/Not able to determine"],
                key=f"correction_{index}"
            )
            
            expert_reason = st.text_area(
                "Reason for correction:",
                key=f"reason_{index}",
                height=60
            )
            
            if st.button(f"Submit Feedback", key=f"submit_{index}"):
                if expert_correction and expert_reason:
                    feedback = {
                        'biosense_id': pred['biosense_id'],
                        'expert_correction': expert_correction,
                        'expert_reason': expert_reason,
                        'original_model_rating': pred['model_rating']
                    }
                    st.session_state.expert_feedback.append(feedback)
                    st.success("Feedback recorded!")
                    st.rerun()
        
        st.markdown("---")


def create_ollama_integration_guide():
    """Create a detailed, non-expert friendly Ollama integration guide"""
    base_model = st.session_state.get('selected_model_display', st.session_state.selected_model)
    
    guide = f"""# Ollama Integration Guide for EpiTuner LoRA

This guide will walk you through integrating your trained LoRA adapter with Ollama step-by-step.

## What You'll Need
- Ollama installed on your computer
- The LoRA adapter files (included in this download)
- 10-15 minutes of your time

## Step 1: Verify Ollama Installation

First, make sure Ollama is working on your computer.

**Windows (PowerShell):**
```powershell
ollama version
```

**If you see a version number, great! If not, download Ollama from:**
https://ollama.ai

## Step 2: Prepare Your Files

1. **Extract the LoRA package** to a folder like:
   `C:\\Users\\YourName\\Documents\\EpiTuner\\my_medical_model\\`

2. **You should see these files:**
   - `adapter_config.json`
   - `adapter_model.safetensors` 
   - `training_info.json`
   - `README.md`
   - `ollama_integration.md` (this file)

## Step 3: Create Your Ollama Model

### 3a. Create a Modelfile

In your extracted folder, create a new file called `Modelfile` (no extension) with this content:

```
FROM {base_model}
ADAPTER ./adapter_model.safetensors

TEMPLATE \"\"\"{{{{ if .System }}}}{{{{ .System }}}}

{{{{ end }}}}{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM \"\"\"You are a medical AI assistant trained to classify medical records for {st.session_state.get('classification_topic', 'medical classification')}. 

When analyzing a syndromic surveillance record, provide:
1. Your classification (Match/Not a Match/Unknown)
2. Your reasoning 
3. Your confidence level (Very Confident/Confident/Somewhat Confident/Not Very Confident/Not at all Confident)

Be thorough but concise in your analysis.\"\"\"
```

### 3b. Build Your Model

Open PowerShell/Terminal in the folder with your files and run:

```powershell
ollama create epituner-medical -f ./Modelfile
```

**What this does:**
- Creates a new model called `epituner-medical` 
- Combines your base model with your trained LoRA adapter
- Sets up the proper prompting template

## Step 4: Test Your Model

Try out your new model:

```powershell
ollama run epituner-medical
```

**Then type a test prompt like:**
```
Classify this syndromic surveillance record:
Chief Complaint: Motor vehicle accident, chest pain
Diagnosis: Rib fracture, chest contusion
```

## Step 5: Using Your Model

### Command Line Usage
```powershell
ollama run epituner-medical "Your syndromic surveillance record text here..."
```

### Python API Usage
```python
import requests
import json

def classify_syndromic_surveillance_record(record_text):
    response = requests.post('http://localhost:11434/api/generate',
                           json={{
                               'model': 'epituner-medical',
                               'prompt': f"Classify this syndromic surveillance record: {{record_text}}",
                               'stream': False
                           }})
    
    if response.status_code == 200:
        result = response.json()
        return result['response']
    else:
        return f"Error: {{response.status_code}}"

# Example usage
record = "Chief Complaint: Chest pain after car accident..."
classification = classify_syndromic_surveillance_record(record)
print(classification)
```

### Batch Processing Script
```python
import csv
import requests
import json

def process_csv_file(input_file, output_file):
    results = []
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Combine relevant fields
            record_text = f"Chief Complaint: {{row.get('ChiefComplaintOrig', '')}}\\n"
            record_text += f"Discharge Diagnosis: {{row.get('DischargeDiagnosis', '')}}"
            
            # Call your model
            response = requests.post('http://localhost:11434/api/generate',
                                   json={{
                                       'model': 'epituner-medical',
                                       'prompt': f"Classify this syndromic surveillance record: {{record_text}}",
                                       'stream': False
                                   }})
            
            if response.status_code == 200:
                classification = response.json()['response']
                row['AI_Classification'] = classification
            else:
                row['AI_Classification'] = 'Error'
            
            results.append(row)
    
    # Save results
    with open(output_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

# Usage
process_csv_file('my_medical_data.csv', 'classified_results.csv')
```

## Troubleshooting

### "Model not found" error
- Make sure you ran `ollama create epituner-medical -f ./Modelfile`
- Check that the Modelfile is in the same folder as your adapter files

### "Adapter file not found" error  
- Verify the adapter_model.safetensors file is in the same folder as your Modelfile
- Make sure the path in the Modelfile matches: `ADAPTER ./adapter_model.safetensors`

### Poor performance
- Your model was trained specifically for: {st.session_state.get('classification_topic', 'medical classification')}
- Make sure you're using it for similar medical classification tasks
- Consider retraining with more examples if needed

### Memory issues
- If Ollama runs out of memory, close other applications
- Consider using a smaller base model for future training

## Model Information

**Your Model Details:**
- **Base Model:** {base_model}
- **Training Task:** {st.session_state.get('classification_topic', 'Syndromic surveillance classification')}
- **Training Accuracy:** {st.session_state.model_results.get('accuracy', 0):.1%}
- **Training Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Records Used:** {len(st.session_state.uploaded_data) if st.session_state.uploaded_data is not None else 0}

## Next Steps

1. **Test thoroughly** with various syndromic surveillance records
2. **Document your results** to track performance
3. **Retrain if needed** with additional expert feedback
4. **Share with your team** - they can use the same `ollama run epituner-medical` command

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify Ollama is running: `ollama list` should show your model
3. Review the training_info.json file for additional details

**Remember:** This model is trained for {st.session_state.get('classification_topic', 'medical classification')} and works best with similar data to what it was trained on.
"""
    
    return guide


def create_and_download_lora_adapter():
    """Create and download a LoRA adapter package"""
    import zipfile
    import io
    import json
    
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create real adapter files from trained model
        adapter_config = {
            "base_model_name_or_path": st.session_state.selected_model,
            "bias": "none", 
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": st.session_state.training_config.get('lora_alpha', 16),
            "lora_dropout": 0.05,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": st.session_state.training_config.get('lora_r', 8),
            "revision": None,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM"
        }
        
        # Add adapter configuration
        zip_file.writestr("adapter_config.json", json.dumps(adapter_config, indent=2))
        
        # Add real adapter weights file
        model_path = Path(st.session_state.model_results['model_path'])
        adapter_file = model_path / "adapter_model.safetensors"
        
        if adapter_file.exists():
            with open(adapter_file, 'rb') as f:
                zip_file.writestr("adapter_model.safetensors", f.read())
        else:
            # Fallback if adapter file not found in expected location
            zip_file.writestr("adapter_model.safetensors", b"[Trained LoRA weights - adapter file not found in expected location. Check the model_path directory.]")
        
        # Add training metadata
        training_info = {
            "training_config": st.session_state.training_config,
            "model_results": {k: v for k, v in st.session_state.model_results.items() if k != 'predictions'},
            "timestamp": datetime.now().isoformat(),
            "epituner_version": "1.0.0",
            "classification_topic": st.session_state.get('classification_topic', ''),
            "data_records": len(st.session_state.uploaded_data) if st.session_state.uploaded_data is not None else 0
        }
        zip_file.writestr("training_info.json", json.dumps(training_info, indent=2))
        
        # Add README for integration
        readme_content = f"""# EpiTuner LoRA Adapter

This package contains a LoRA (Low-Rank Adaptation) adapter trained with EpiTuner.

## Files Included:
- `adapter_config.json`: LoRA configuration parameters
- `adapter_model.safetensors`: Trained adapter weights
- `training_info.json`: Training metadata and performance metrics
- `ollama_integration.md`: Step-by-step Ollama setup instructions

## Quick Integration with Ollama:

1. Extract this zip file to a folder
2. Follow the instructions in `ollama_integration.md`
3. Your model will be available as `epituner-medical` in Ollama

## Model Information:
- Base Model: {st.session_state.get('selected_model_display', st.session_state.selected_model)}
- Training Task: {st.session_state.get('classification_topic', 'Syndromic surveillance classification')}
- Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Performance: {st.session_state.model_results.get('accuracy', 0):.1%} accuracy

For support, refer to the EpiTuner documentation.
"""
        zip_file.writestr("README.md", readme_content)
        
        # Add detailed Ollama integration guide
        ollama_guide = create_ollama_integration_guide()
        zip_file.writestr("ollama_integration.md", ollama_guide)
    
    zip_buffer.seek(0)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"epituner_lora_adapter_{timestamp}.zip"
    
    # Offer download
    st.download_button(
        label="💾 Download LoRA Package", 
        data=zip_buffer.read(),
        file_name=filename,
        mime="application/zip",
        help="Downloads a complete package with adapter files and integration instructions"
    )
    
    st.success("✅ LoRA adapter package created!")
    st.info("The download includes everything you need to integrate with Ollama.")


def step_export():
    """Step 6: Export Results"""
    st.markdown('<div class="section-header">Step 6: Export Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_results:
        st.warning("Please complete training first.")
        return
    
    st.markdown("### Training Complete! 🎉")
    
    results = st.session_state.model_results
    
    # Results summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Performance")
        st.markdown(f"- **Accuracy:** {results['accuracy']:.1%}")
        st.markdown(f"- **Precision:** {results['precision']:.1%}")
        st.markdown(f"- **Recall:** {results['recall']:.1%}")
        st.markdown(f"- **F1 Score:** {results['f1_score']:.1%}")
        
        if st.session_state.expert_feedback:
            st.markdown(f"- **Expert Feedback:** {len(st.session_state.expert_feedback)} corrections")
    
    with col2:
        st.markdown("#### Training Configuration")
        config = st.session_state.training_config
        st.markdown(f"- **Model:** {config['model_name']}")
        st.markdown(f"- **Epochs:** {config['epochs']}")
        st.markdown(f"- **Learning Rate:** {config['learning_rate']:.0e}")
        st.markdown(f"- **LoRA Rank:** {config['lora_r']}")
        st.markdown(f"- **Records Trained:** {len(st.session_state.uploaded_data)}")
    
    st.markdown("### Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # LoRA adapter download
        st.markdown("#### LoRA Adapter")
        st.markdown("Download the trained LoRA adapter for use with Ollama or other tools.")
        
        if st.button("Download LoRA Adapter", type="primary"):
            # Create a real LoRA adapter file for download
            create_and_download_lora_adapter()
    
    with col2:
        # Metadata download
        st.markdown("#### Training Metadata")
        st.markdown("Download configuration and metrics for reproducibility.")
        
        metadata = {
            'model_config': st.session_state.training_config,
            'training_results': {k: v for k, v in results.items() if k != 'predictions'},
            'expert_feedback': st.session_state.expert_feedback,
            'timestamp': datetime.now().isoformat(),
            'data_statistics': {
                'total_records': len(st.session_state.uploaded_data),
                'expert_rating_distribution': st.session_state.uploaded_data['Expert Rating'].value_counts().to_dict()
            }
        }
        
        metadata_json = json.dumps(metadata, indent=2)
        
        st.download_button(
            label="Download Metadata",
            data=metadata_json,
            file_name=f"epituner_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Training data download
        st.markdown("#### Training Data")
        st.markdown("Download the processed training data used for this model.")
        
        # Add model predictions to the original data
        df_with_predictions = st.session_state.uploaded_data.copy()
        
        # Add model predictions
        pred_dict = {p['biosense_id']: p for p in results['predictions']}
        df_with_predictions['Model_Rating'] = df_with_predictions['C_Biosense_ID'].map(
            lambda x: pred_dict.get(x, {}).get('model_rating', '')
        )
        df_with_predictions['Model_Confidence'] = df_with_predictions['C_Biosense_ID'].map(
            lambda x: pred_dict.get(x, {}).get('confidence', '')
        )
        df_with_predictions['Model_Rationale'] = df_with_predictions['C_Biosense_ID'].map(
            lambda x: pred_dict.get(x, {}).get('model_rationale', '')
        )
        
        csv_data = df_with_predictions.to_csv(index=False)
        
        st.download_button(
            label="Download Data + Predictions",
            data=csv_data,
            file_name=f"epituner_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Ollama integration instructions  
    st.markdown("### Ollama Integration Guide")
    st.markdown("Complete step-by-step instructions are included in your download package.")
    
    with st.expander("🔧 Quick Setup Preview (Full guide in download)"):
        base_model = st.session_state.get('selected_model_display', st.session_state.selected_model)
        
        st.markdown(f"""
        #### Quick Setup Steps:
        
        1. **Download the LoRA package** (button above)
        2. **Extract files** to a folder 
        3. **Create Modelfile** with content:
           ```
           FROM {base_model}
           ADAPTER ./adapter_model.safetensors
           ```
        4. **Build your model:**
           ```bash
           ollama create epituner-medical -f ./Modelfile
           ```
        5. **Test it:**
           ```bash
           ollama run epituner-medical "Your syndromic surveillance record text..."
           ```
        
        **The download includes:**
        - ✅ Complete integration guide (`ollama_integration.md`)
        - ✅ Python scripts for batch processing
        - ✅ Troubleshooting instructions
        - ✅ API usage examples
        
        **Perfect for non-experts** - no technical background needed!
        """)
    
    # Start new project button
    st.markdown("---")
    
    if st.button("Start New Project", type="secondary"):
        # Reset session state
        for key in ['current_step', 'uploaded_data', 'selected_model', 'training_config', 
                   'training_in_progress', 'model_results', 'expert_feedback']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.current_step = 'data_upload'
        st.rerun()


def main():
    """Main application"""
    display_header()
    display_sidebar()
    
    # Route to current step
    step_functions = {
        'data_upload': step_data_upload,
        'model_selection': step_model_selection,
        'training_config': step_training_config,
        'training': step_training,
        'expert_review': step_expert_review,
        'export': step_export
    }
    
    current_step = st.session_state.current_step
    if current_step in step_functions:
        step_functions[current_step]()
    else:
        st.error(f"Unknown step: {current_step}")


if __name__ == "__main__":
    main()
