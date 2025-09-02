#!/usr/bin/env python3
"""
EpiTuner - Simplified Medical LoRA Training
Always Works™ implementation with minimal complexity
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Import our simplified components
from epituner.core.data_processor import MedicalDataProcessor
from epituner.core.simple_trainer import SimpleTrainer, TrainingResult
from epituner.core.simple_inference import SimpleInference, PredictionResult
from epituner.utils.hardware import get_gpu_info, get_ollama_models, validate_system


# Page config
st.set_page_config(
    page_title="EpiTuner - Medical LoRA Training",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'topic' not in st.session_state:
    st.session_state.topic = ""
if 'training_result' not in st.session_state:
    st.session_state.training_result = None

def main():
    """Main application - simplified 5-step process"""
    st.title("🏥 EpiTuner - Medical LoRA Training")
    st.markdown("Simple, reliable medical data classification training")
    
    # Show current step
    steps = ['upload', 'model', 'train', 'review', 'download']
    current_idx = steps.index(st.session_state.step)
    
    st.markdown("### Progress")
    cols = st.columns(5)
    for i, step in enumerate(['📁 Upload', '🤖 Model', '🎯 Train', '📊 Review', '💾 Download']):
        with cols[i]:
            if i == current_idx:
                st.markdown(f"**{step}** ⬅️")
            elif i < current_idx:
                st.markdown(f"✅ {step}")
            else:
                st.markdown(f"⭕ {step}")
    
    st.markdown("---")
    
    # Route to appropriate step
    if st.session_state.step == 'upload':
        step_upload()
    elif st.session_state.step == 'model':
        step_model() 
    elif st.session_state.step == 'train':
        step_train()
    elif st.session_state.step == 'review':
        step_review()
    elif st.session_state.step == 'download':
        step_download()

def step_upload():
    """Step 1: Upload data"""
    st.header("📁 Upload Medical Data")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            # Validate and load data
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            is_valid, errors = MedicalDataProcessor.validate_csv(temp_path)
            
            if is_valid:
                df = MedicalDataProcessor.load_data(temp_path)
                st.session_state.data = df
                
                st.success("✅ Data loaded successfully!")
                st.dataframe(df.head())
                
                stats = MedicalDataProcessor.get_stats(df)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", stats['total_records'])
                with col2:
                    st.metric("Match Cases", stats['rating_distribution'].get('Match', 0))
                with col3:
                    st.metric("Complete", stats['complete_records'])
                
                if st.button("Next: Select Model →", type="primary"):
                    st.session_state.step = 'model'
                    st.rerun()
            else:
                st.error("❌ Data validation failed:")
                for error in errors:
                    st.error(error)
            
            Path(temp_path).unlink(missing_ok=True)
            
        except Exception as e:
            st.error(f"Error: {e}")

def step_model():
    """Step 2: Select model"""
    st.header("🤖 Select Model")
    
    if st.session_state.data is None:
        st.error("Upload data first")
        return
    
    # Model selection
    models = get_ollama_models()
    if models:
        selected = st.selectbox("Choose Ollama model:", [""] + models)
        if selected:
            # Map to HuggingFace name
            mapping = {
                'llama3.2:1b': 'meta-llama/Llama-3.2-1B-Instruct',
                'phi3': 'microsoft/Phi-3-mini-4k-instruct',
                'mistral': 'mistralai/Mistral-7B-Instruct-v0.1'
            }
            st.session_state.model_name = mapping.get(selected, selected)
            st.success(f"Selected: {st.session_state.model_name}")
    else:
        st.error("No Ollama models found. Install with: ollama pull llama3.2:1b")
    
    # Classification task
    topic = st.text_area("What should the model classify?", 
                        value="Motor vehicle collisions in medical records")
    st.session_state.topic = topic
    
    if st.session_state.model_name and topic:
        if st.button("Next: Train →", type="primary"):
            st.session_state.step = 'train'
            st.rerun()

def step_train():
    """Step 3: Train model"""
    st.header("🎯 Train Model")
    
    if not all([st.session_state.data is not None, st.session_state.model_name, st.session_state.topic]):
        st.error("Complete previous steps first")
        return
    
    # Training settings
    epochs = st.slider("Epochs", 1, 5, 2)
    learning_rate = st.selectbox("Learning Rate", [1e-4, 2e-4, 3e-4], index=1)
    
    st.info(f"Training {len(st.session_state.data)} records on {st.session_state.model_name}")
    
    if st.button("🚀 Start Training", type="primary"):
        # Prepare data
        training_texts = MedicalDataProcessor.prepare_for_training(
            st.session_state.data, st.session_state.topic)
        
        output_dir = f"outputs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(message: str, progress: float):
            progress_bar.progress(progress)
            status_text.text(message)
        
        # Train
        trainer = SimpleTrainer(st.session_state.model_name, progress_callback)
        result = trainer.train(training_texts, output_dir, epochs, learning_rate)
        
        st.session_state.training_result = result
        
        if result.success:
            st.success("🎉 Training completed!")
            st.session_state.step = 'review'
            st.rerun()
        else:
            st.error(f"Training failed: {result.error_message}")

def step_review():
    """Step 4: Review results"""
    st.header("📊 Review Results")
    
    result = st.session_state.training_result
    if not result or not result.success:
        st.error("No successful training found")
        return
    
    st.success("✅ Training completed successfully!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Loss", f"{result.training_loss:.4f}")
        st.metric("Steps", result.steps_completed)
    
    with col2:
        st.info(f"Model saved to: {result.model_path}")
    
    if st.button("Next: Download →", type="primary"):
        st.session_state.step = 'download'
        st.rerun()

def step_download():
    """Step 5: Download model"""
    st.header("💾 Download Model")
    
    result = st.session_state.training_result
    if not result or not result.success:
        st.error("No model to download")
        return
    
    st.success("🎉 Your model is ready!")
    
    # Simple download options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Files")
        st.info(f"Location: {result.model_path}")
        st.markdown("Copy this folder to use your trained model")
    
    with col2:
        st.markdown("### Start Over")
        if st.button("🆕 New Project"):
            for key in ['step', 'data', 'model_name', 'topic', 'training_result']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 'upload'
            st.rerun()

if __name__ == "__main__":
    main()

