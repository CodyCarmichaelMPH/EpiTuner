"""
EpiTuner GUI - Streamlit Interface for Ollama Fine-Tuning and Evaluation Suite

This module provides a comprehensive web-based GUI for the entire EpiTuner suite,
enabling non-technical users to perform all operations through an intuitive interface.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import time
from datetime import datetime

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from data_loader import DataLoader
from schema_mapper import SchemaMapper
from formatter import Formatter
from fine_tuner import FineTuner
from inference_runner import InferenceRunner
from contextualizer import Contextualizer
from debugging_logger import get_logger, LogLevel


class EpiTunerGUI:
    """
    Streamlit-based GUI for the EpiTuner suite.
    
    Provides interfaces for:
    - Data loading and validation
    - Schema mapping and rating standardization
    - Prompt formatting
    - Model fine-tuning
    - Inference execution
    - Results visualization and export
    """
    
    def __init__(self):
        """Initialize the GUI."""
        self.setup_page_config()
        self.setup_session_state()
        self.logger = get_logger(debug_mode=True)
        
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="EpiTuner - Ollama Fine-Tuning Suite",
            page_icon="",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        
        if 'rating_mapping' not in st.session_state:
            st.session_state.rating_mapping = {}
        
        if 'formatted_prompts' not in st.session_state:
            st.session_state.formatted_prompts = None
        
        if 'inference_results' not in st.session_state:
            st.session_state.inference_results = None
        
        if 'current_topic' not in st.session_state:
            st.session_state.current_topic = ""
    
    def render_header(self):
        """Render the main header."""
        st.title("EpiTuner - Ollama Fine-Tuning and Evaluation Suite")
        st.markdown("---")
        
        # Progress indicator
        steps = ["Data Upload", "Schema Mapping", "Prompt Formatting",
                 "Model Operations", "Results & Export"]
        
        cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols, steps)):
            if i + 1 < st.session_state.current_step:
                col.success(step)
            elif i + 1 == st.session_state.current_step:
                col.info(step)
            else:
                col.write(step)
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        with st.sidebar:
            st.header("Navigation & Settings")
            
            # Step navigation
            st.subheader("Current Step")
            step_names = {
                1: "Data Upload",
                2: "Schema Mapping", 
                3: "Prompt Formatting",
                4: "Model Operations",
                5: "Results & Export"
            }
            st.info(f"Step {st.session_state.current_step}: {step_names[st.session_state.current_step]}")
            
            # Quick navigation
            st.subheader("Quick Navigation")
            for step_num, step_name in step_names.items():
                if st.button(f"Go to {step_name}", key=f"nav_{step_num}"):
                    st.session_state.current_step = step_num
                    st.rerun()
            
            st.markdown("---")
            
            # Settings
            st.subheader("Settings")
            debug_mode = st.checkbox("Debug Mode", value=True, key="debug_mode")
            interactive_debug = st.checkbox("Interactive Debug", value=False, key="interactive_debug")
            
            # Model settings
            st.subheader("Model Settings")
            
            # Get available models from Ollama
            available_models = self.get_available_models()
            if available_models:
                model_name = st.selectbox(
                    "Select Model",
                    options=available_models,
                    index=0 if "llama3.2:1b" in available_models else 0,
                    key="model_name"
                )
            else:
                model_name = st.text_input("Model Name", value="llama3.2:1b", key="model_name")
                st.warning("Could not fetch models from Ollama. Please enter model name manually.")
            
            server_url = st.text_input("Ollama Server URL", value="http://localhost:11434", key="server_url")
            
            # Batch settings
            st.subheader("Batch Settings")
            batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=100, key="batch_size")
            
            st.markdown("---")
            
            # System info
            st.subheader("System Info")
            st.write(f"Session ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Clear session button
            if st.button("Clear Session", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                self.setup_session_state()
                st.rerun()
    
    def get_available_models(self) -> List[str]:
        """Get available models from Ollama server."""
        try:
            import requests
            import json
            
            server_url = st.session_state.get('server_url', 'http://localhost:11434')
            response = requests.get(f"{server_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
            else:
                return []
        except Exception as e:
            st.session_state.get('debug_mode', True) and st.write(f"Debug: Could not fetch models: {e}")
            return []
    
    def step_1_data_upload(self):
        """Step 1: Data upload and validation."""
        st.header("Data Upload & Validation")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                st.metric("File Type", uploaded_file.type)
            with col3:
                st.metric("File Name", uploaded_file.name)
            
            # Load and validate data
            try:
                with st.spinner("Loading and validating data..."):
                    data_loader = DataLoader(debug_mode=st.session_state.get('debug_mode', True))
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Load dataset
                        df = data_loader.load_dataset(tmp_path)
                        
                        # Validate schema
                        is_valid, missing_fields, suggestions = data_loader.validate_schema(df)
                        
                        if is_valid:
                            st.success("Data validation successful!")
                            
                            # Display data preview
                            st.subheader("Data Preview")
                            st.dataframe(df.head(10), use_container_width=True)
                            
                            # Display data info
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Rows", len(df))
                                st.metric("Total Columns", len(df.columns))
                            with col2:
                                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                            
                            # Store processed data
                            st.session_state.processed_data = df
                            
                            # Check for expert ratings
                            has_expert_ratings = 'Expert Rating' in df.columns
                            if has_expert_ratings:
                                unique_ratings = data_loader.extract_unique_ratings(df)
                                st.info(f"Found {len(unique_ratings)} unique expert ratings: {unique_ratings}")
                            
                            # Proceed button
                            if st.button("Proceed to Schema Mapping", type="primary"):
                                st.session_state.current_step = 2
                                st.rerun()
                        
                        else:
                            st.error("Data validation failed!")
                            st.subheader("Missing Fields:")
                            for field in missing_fields:
                                st.write(f"• {field}")
                            
                            if suggestions:
                                st.subheader("Suggestions:")
                                for field, suggestion in suggestions.items():
                                    st.write(f"• {field}: {suggestion}")
                    
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                self.logger.error(f"Data upload failed: {e}", "gui", "step_1_data_upload", error=e)
    
    def step_2_schema_mapping(self):
        """Step 2: Schema mapping and rating standardization."""
        st.header("Schema Mapping & Rating Standardization")
        
        if st.session_state.processed_data is None:
            st.warning("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.processed_data
        
        # Check for expert ratings
        has_expert_ratings = 'Expert Rating' in df.columns
        
        if has_expert_ratings:
            st.subheader("Expert Rating Standardization")
            
            # Extract unique ratings
            data_loader = DataLoader(debug_mode=st.session_state.get('debug_mode', True))
            unique_ratings = data_loader.extract_unique_ratings(df)
            
            st.write(f"Found {len(unique_ratings)} unique rating values:")
            
            # Rating mapping interface
            rating_mapping = {}
            
            # Predefined mapping options
            mapping_options = {
                "Binary (0/1)": {0: "No Match", 1: "Match"},
                "Three-level (0/1/2)": {0: "No Match", 1: "Partial Match", 2: "Full Match"},
                "Five-level (0-4)": {0: "No Match", 1: "Weak Match", 2: "Partial Match", 3: "Strong Match", 4: "Full Match"}
            }
            
            # Mapping method selection
            st.write("### Choose Your Rating Mapping Approach")
            
            mapping_methods = {
                "Manual Mapping": {
                    "description": "I'll manually map each rating to what it means",
                    "best_for": "When you know exactly what each rating means"
                },
                "Predefined Mapping": {
                    "description": "Choose from standard rating schemes",
                    "best_for": "When you want to use established classification modes"
                },
                "Auto-detect": {
                    "description": "Let the system suggest mappings based on your data",
                    "best_for": "When you want quick setup with review options"
                }
            }
            
            # Show method descriptions
            for method, info in mapping_methods.items():
                with st.expander(f"**{method}** - {info['description']}"):
                    st.write(f"**Best for:** {info['best_for']}")
            
            mapping_method = st.radio(
                "**Select your preferred approach:**",
                list(mapping_methods.keys()),
                format_func=lambda x: f"**{x}** - {mapping_methods[x]['description']}"
            )
            
            if mapping_method == "Manual Mapping":
                st.write("### Rating Standardization")
                st.write("""
                **What does this mean?** We need to convert your expert ratings into a standard format that the AI model can understand.
                
                **Standard Rating Scale:**
                - **No Match (0)**: The case does NOT align with the target topic
                - **Partial Match (1)**: The case somewhat aligns but is unclear or borderline
                - **Clear Match (2)**: The case clearly aligns with the target topic
                """)
                
                st.write("---")
                st.write("**Now, let's map your ratings:**")
                
                # Create a more user-friendly mapping interface
                for rating in unique_ratings:
                    st.write(f"### Rating: **{rating}**")
                    
                    # Show what this rating currently means (if we can infer from the data)
                    rating_description = self._get_rating_description(rating, df)
                    if rating_description:
                        st.info(f"**Current meaning:** {rating_description}")
                    
                    # Provide clear mapping options
                    mapping_options = {
                        0: "No Match - Does NOT align with topic",
                        1: "Partial Match - Somewhat aligns but unclear", 
                        2: "Clear Match - Clearly aligns with topic"
                    }
                    
                    selected_mapping = st.radio(
                        f"What does rating '{rating}' mean in your data?",
                        options=list(mapping_options.keys()),
                        format_func=lambda x: mapping_options[x],
                            key=f"map_{rating}"
                        )
                    
                    rating_mapping[rating] = selected_mapping
                    st.write("---")
            
            elif mapping_method == "Predefined Mapping":
                st.write("### Choose a Standard Rating Scheme")
                
                predefined_schemes = {
                    "Binary (Match/No Match)": {
                        "description": "Simple yes/no classification",
                        "mapping": {0: "No Match", 1: "Clear Match"}
                    },
                    "Three-Level (No/Partial/Clear Match)": {
                        "description": "Standard classification with partial matches",
                        "mapping": {0: "No Match", 1: "Partial Match", 2: "Clear Match"}
                    },
                    "Confidence-Based (Low/Medium/High)": {
                        "description": "Based on confidence in the match",
                        "mapping": {0: "Low Confidence", 1: "Medium Confidence", 2: "High Confidence"}
                    }
                }
                
                selected_scheme = st.selectbox(
                    "Choose a rating scheme:",
                    list(predefined_schemes.keys())
                )
                
                scheme_info = predefined_schemes[selected_scheme]
                st.info(f"**{scheme_info['description']}**")
                
                # Show the scheme mapping
                st.write("**Rating Scheme:**")
                for value, label in scheme_info['mapping'].items():
                    st.write(f"- {value}: {label}")
                
                st.write("---")
                st.write("**Map your ratings to this scheme:**")
                
                # Create mapping interface for predefined scheme
                for rating in unique_ratings:
                    rating_description = self._get_rating_description(rating, df)
                    if rating_description:
                        st.info(f"**Rating {rating}:** {rating_description}")
                    
                    selected_mapping = st.selectbox(
                        f"Map rating '{rating}' to:",
                        options=list(scheme_info['mapping'].keys()),
                        format_func=lambda x: scheme_info['mapping'][x],
                        key=f"predef_map_{rating}"
                    )
                    
                    rating_mapping[rating] = selected_mapping
            
            else:  # Auto-detect
                st.write("### Auto-Detection Mode")
                st.info("""
                **What this does:** The system will automatically try to understand your rating scale 
                and suggest a mapping. You can review and adjust the suggestions below.
                """)
                
                # Smart auto-detection logic
                suggested_mapping = {}
                
                for rating in unique_ratings:
                    if isinstance(rating, (int, float)):
                        # For numeric ratings, assume they're already on a reasonable scale
                        if rating == 0:
                            suggested_mapping[rating] = 0  # No match
                        elif rating == 1:
                            suggested_mapping[rating] = 2  # Clear match
                        elif rating == 2:
                            suggested_mapping[rating] = 1  # Partial match
                        else:
                            # For other numbers, map to closest standard value
                            suggested_mapping[rating] = min(2, max(0, int(rating)))
                    elif isinstance(rating, str):
                        # Try to extract numeric value
                        try:
                            numeric_value = int(rating)
                            suggested_mapping[rating] = min(2, max(0, numeric_value))
                        except ValueError:
                            # For non-numeric strings, try to infer meaning
                            rating_lower = rating.lower()
                            if any(word in rating_lower for word in ['no', 'not', 'negative', '0']):
                                suggested_mapping[rating] = 0
                            elif any(word in rating_lower for word in ['yes', 'positive', 'clear', 'match', '1']):
                                suggested_mapping[rating] = 2
                            else:
                                suggested_mapping[rating] = 1  # Default to partial
                
                st.write("**Suggested Mapping:**")
                mapping_options = {
                    0: "No Match - Does NOT align with topic",
                    1: "Partial Match - Somewhat aligns but unclear", 
                    2: "Clear Match - Clearly aligns with topic"
                }
                
                for rating in unique_ratings:
                    st.write(f"### Rating: **{rating}**")
                    
                    rating_description = self._get_rating_description(rating, df)
                    if rating_description:
                        st.info(f"**Current meaning:** {rating_description}")
                    
                    suggested = suggested_mapping.get(rating, 1)
                    
                    selected_mapping = st.radio(
                        f"Suggested mapping for '{rating}':",
                        options=list(mapping_options.keys()),
                        index=suggested,
                        format_func=lambda x: mapping_options[x],
                        key=f"auto_map_{rating}"
                    )
                    
                    rating_mapping[rating] = selected_mapping
                    st.write("---")
            
            # Apply mapping
            if st.button("Apply Rating Mapping", type="primary"):
                try:
                    with st.spinner("Applying rating mapping..."):
                        schema_mapper = SchemaMapper(debug_mode=st.session_state.get('debug_mode', True))
                        
                        # Apply mapping
                        mapped_df = schema_mapper.apply_mapping(df, rating_mapping)
                        
                        # Store results
                        st.session_state.processed_data = mapped_df
                        st.session_state.rating_mapping = rating_mapping
                        
                        st.success("Rating mapping applied successfully!")
                        
                        # Show mapping results
                        st.subheader("Mapping Results")
                        
                        # Create a more user-friendly mapping display
                        mapping_display = []
                        rating_labels = {
                            0: "No Match",
                            1: "Partial Match", 
                            2: "Clear Match"
                        }
                        
                        for orig, std in rating_mapping.items():
                            rating_description = self._get_rating_description(orig, df)
                            mapping_display.append({
                                "Your Rating": orig,
                                "Standardized As": rating_labels.get(std, f"Unknown ({std})"),
                                "Meaning": rating_description or "No description available"
                            })
                        
                        mapping_df = pd.DataFrame(mapping_display)
                        st.dataframe(mapping_df, use_container_width=True)
                        
                        # Show summary
                        st.success(f"Successfully mapped {len(rating_mapping)} rating values!")
                        
                        # Show distribution
                        st.write("**Rating Distribution:**")
                        rating_counts = mapped_df['Standardized_Rating'].value_counts().sort_index()
                        for rating, count in rating_counts.items():
                            label = rating_labels.get(rating, f"Unknown ({rating})")
                            st.write(f"- {label}: {count} cases")
                        
                        # Show data preview
                        st.subheader("Updated Data Preview")
                        st.dataframe(mapped_df.head(10), use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error applying mapping: {str(e)}")
                    self.logger.error(f"Schema mapping failed: {e}", "gui", "step_2_schema_mapping", error=e)
        
        else:
            st.info("No expert ratings found. Proceeding to inference-only mode.")
        
        # Topic selection
        st.subheader("Target Topic")
        st.write("""
        **What is the target topic?** This is the specific condition or category 
        you want to evaluate each case against. For example, if you're looking at cases 
        and want to know which ones are related to "viral infections", that would be your target topic.
        """)
        
        topic = st.text_input(
            "Enter the specific topic to evaluate cases against:",
            value=st.session_state.get('current_topic', ''),
            placeholder="e.g., Viral Infections, Cardiac Conditions, Respiratory Issues"
        )
        
        # Show example for sample data
        if 'sample_dataset.csv' in str(st.session_state.get('uploaded_file', '')):
            st.info("""
            **For the sample data:** The cases include various complaints like fever, chest pain, 
            headache, etc. The expert ratings show which cases are related to "Viral Infections" 
            (rating 1 = viral-related, rating 0 = not viral-related).
            """)
        
        if topic:
            st.session_state.current_topic = topic
        
        # Proceed button
        if st.button("Proceed to Prompt Formatting", type="primary"):
            st.session_state.current_step = 3
            st.rerun()
    
    def _get_rating_description(self, rating: Any, df: pd.DataFrame) -> Optional[str]:
        """
        Get a description of what a rating means based on the data.
        
        Args:
            rating: The rating value to describe
            df: The DataFrame containing the data
            
        Returns:
            Description string or None if no description found
        """
        try:
            # Look for rationale column to understand the rating
            rationale_col = None
            for col in df.columns:
                if 'rationale' in col.lower() or 'reason' in col.lower():
                    rationale_col = col
                    break
            
            if rationale_col:
                # Get sample rationales for this rating
                rating_data = df[df['Expert Rating'] == rating]
                if not rating_data.empty:
                    sample_rationales = rating_data[rationale_col].dropna().head(2).tolist()
                    if sample_rationales:
                        return f"Sample: {'; '.join(sample_rationales)}"
            
            # If no rationale, try to infer from the data
            rating_data = df[df['Expert Rating'] == rating]
            if not rating_data.empty:
                # Look for patterns in complaints or diagnoses
                complaint_cols = [col for col in df.columns if 'complaint' in col.lower()]
                diagnosis_cols = [col for col in df.columns if 'diagnosis' in col.lower()]
                
                if complaint_cols and diagnosis_cols:
                    sample_complaints = rating_data[complaint_cols[0]].dropna().head(2).tolist()
                    sample_diagnoses = rating_data[diagnosis_cols[0]].dropna().head(2).tolist()
                    
                    if sample_complaints and sample_diagnoses:
                        return f"Examples: {sample_complaints[0]} → {sample_diagnoses[0]}"
            
            return None
            
        except Exception:
            return None
    
    def step_3_prompt_formatting(self):
        """Step 3: Prompt formatting and preparation."""
        st.header("Prompt Formatting & Preparation")
        
        if st.session_state.processed_data is None:
            st.warning("No data loaded. Please go back to Step 1.")
            return
        
        if not st.session_state.current_topic:
            st.warning("No topic specified. Please go back to Step 2.")
            return
        
        df = st.session_state.processed_data
        topic = st.session_state.current_topic
        
        # Explain what we're doing
        st.write(f"""
        **What we're doing:** Converting your data into prompts that ask the AI model to evaluate 
        each case against the topic **"{topic}"**. The model will then predict whether each case 
        aligns with this topic and provide reasoning.
        
        **Context Summary Approach (Recommended):** Instead of showing individual training cases, 
        this method extracts key patterns from your expert-rated data to create a context summary 
        like "Look for respiratory symptoms: cough, fever, sore throat, etc." This context is then 
        used to evaluate new cases.
        """)
        
        # Show what we have
        st.info(f"""
        **Data Summary:**
        - **Topic to evaluate:** {topic}
        - **Total cases:** {len(df)} 
        - **Cases with expert ratings:** {len(df[df['Standardized_Rating'].notna()]) if 'Standardized_Rating' in df.columns else 0}
        - **Cases for inference:** {len(df[df['Standardized_Rating'].isna()]) if 'Standardized_Rating' in df.columns else len(df)}
        """)
        
        # Formatting options
        st.subheader("Formatting Options")
        
        col1, col2 = st.columns(2)
        with col1:
            include_rationale = st.checkbox("Include Expert Rationale", value=True)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=100)
        
        with col2:
            custom_template = st.checkbox("Use Custom Template", value=False)
            prompt_type = st.selectbox(
                "Prompt Type",
                ["Context Summary (Recommended)", "Auto-detect", "Training Only", "Inference Only", "Mixed"]
            )
        
        # Custom template input
        if custom_template:
            st.subheader("Custom Template")
            template_text = st.text_area(
                "Enter custom prompt template:",
                value="""INPUT:
You are evaluating whether this case aligns with topic '{topic}'.

Patient Data:
{context_block}

OUTPUT:
Rating: {rating}
Rationale: {rationale}""",
                height=200
            )
        else:
            template_text = None
        
        # Format prompts
        if st.button("Format Prompts", type="primary"):
            try:
                with st.spinner("Formatting prompts..."):
                    formatter = Formatter(debug_mode=st.session_state.get('debug_mode', True))
                    
                    # Determine prompt type
                    has_ratings = 'Standardized_Rating' in df.columns
                    
                    if prompt_type == "Context Summary (Recommended)":
                        # Use the new context summary approach
                        prompts = formatter.create_context_summary_prompts(df, topic, template_text)
                        st.info("Using Context Summary approach: Extracting key patterns from training data to create evaluation context.")
                    elif prompt_type == "Auto-detect":
                        if has_ratings:
                            prompts = formatter.create_training_prompts(
                                df, topic, include_rationale, template_text
                            )
                        else:
                            prompts = formatter.create_inference_prompts(df, topic, template_text)
                    elif prompt_type == "Training Only":
                        if not has_ratings:
                            st.error("No expert ratings found for training prompts!")
                            return
                        prompts = formatter.create_training_prompts(
                            df, topic, include_rationale, template_text
                        )
                    elif prompt_type == "Inference Only":
                        prompts = formatter.create_inference_prompts(df, topic, template_text)
                    else:  # Mixed
                        mixed_prompts = formatter.create_mixed_prompts(
                            df, topic, include_rationale, 
                            {'training': template_text, 'inference': template_text} if template_text else None
                        )
                        prompts = mixed_prompts['training'] + mixed_prompts['inference']
                    
                    # Store formatted prompts
                    st.session_state.formatted_prompts = prompts
                    
                    st.success(f"Formatted {len(prompts)} prompts successfully!")
                    
                    # Show prompt statistics
                    stats = formatter.get_prompt_statistics(prompts)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Prompts", stats['total_prompts'])
                    with col2:
                        st.metric("Avg Length", stats['avg_prompt_length'])
                    with col3:
                        st.metric("Topics", len(stats['topics']))
                    
                    # Show prompt preview
                    st.subheader("Prompt Preview")
                    if prompts:
                        preview_prompt = prompts[0]
                        st.text_area(
                            "First Prompt:",
                            preview_prompt['prompt'],
                            height=300,
                            disabled=True
                        )
                    
                    # Show rating distribution if available
                    if stats['rating_distribution']:
                        st.subheader("Rating Distribution")
                        rating_df = pd.DataFrame([
                            {"Rating": rating, "Count": count}
                            for rating, count in stats['rating_distribution'].items()
                        ])
                        st.bar_chart(rating_df.set_index("Rating"))
            
            except Exception as e:
                st.error(f"Error formatting prompts: {str(e)}")
                self.logger.error(f"Prompt formatting failed: {e}", "gui", "step_3_prompt_formatting", error=e)
        
        # Proceed button
        if st.session_state.formatted_prompts:
            if st.button("Proceed to Model Operations", type="primary"):
                st.session_state.current_step = 4
                st.rerun()
    
    def step_4_model_operations(self):
        """Step 4: Model operations (fine-tuning, inference, contextualization)."""
        st.header("Model Operations")
        
        if st.session_state.formatted_prompts is None:
            st.warning("No formatted prompts available. Please go back to Step 3.")
            return
        
        prompts = st.session_state.formatted_prompts
        
        # Operation selection
        st.subheader("Choose Operation")
        operation = st.radio(
            "Select the operation to perform:",
            ["Run Inference", "Fine-tune Model", "Create Contextualizer", "All Operations"]
        )
        
        # Model settings
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            # Get available models from Ollama
            available_models = self.get_available_models()
            if available_models:
                model_name = st.selectbox(
                    "Select Model",
                    options=available_models,
                    index=0 if "llama3.2:1b" in available_models else 0,
                    key="model_name_step4"
                )
            else:
                model_name = st.text_input("Model Name", value=st.session_state.get('model_name', 'llama3.2:1b'), key="model_name_step4")
                st.warning("Could not fetch models from Ollama. Please enter model name manually.")
            
            server_url = st.text_input("Server URL", value=st.session_state.get('server_url', 'http://localhost:11434'))
        
        with col2:
            max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=512)
            temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        
        # Run operations
        if st.button("Start Operation", type="primary"):
            try:
                with st.spinner("Running model operations..."):
                    results = {}
                    
                    if operation in ["Run Inference", "All Operations"]:
                        st.subheader("Running Inference...")
                        
                        # Show low power mode message
                        st.info("""
                        **Low Power Mode:** Processing may take 1-3 minutes per case on lower-powered hardware. 
                        This is normal - the system is optimized for low power mode performance.
                        """)
                        
                        # Use longer timeouts for low power mode
                        inference_runner = InferenceRunner(
                            debug_mode=st.session_state.get('debug_mode', True),
                            timeout=180,  # 3 minutes per request
                            batch_size=1,  # Process one at a time for low power mode
                            max_retries=2
                        )
                        
                        # Extract prompt texts from the formatted prompts
                        prompt_texts = [p['prompt'] for p in prompts]
                        row_ids = [p.get('id', f'row_{i}') for i, p in enumerate(prompts)]
                        
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run inference with progress updates
                        status_text.text("Starting inference...")
                        inference_results = inference_runner.run_batch_inference(
                            prompt_texts, model_name, row_ids
                        )
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Inference completed!")
                        
                        results['inference'] = inference_results
                        st.session_state.inference_results = inference_results
                        
                        # Count successful vs failed results
                        successful_results = [r for r in inference_results if r.get('prediction') is not None]
                        failed_results = [r for r in inference_results if r.get('prediction') is None]
                        
                        if successful_results:
                            st.success(f"Inference completed! {len(successful_results)} successful, {len(failed_results)} failed")
                            
                            # Show sample results
                            if successful_results:
                                st.write("**Sample Results:**")
                                sample_result = successful_results[0]
                                st.write(f"- Prediction: {sample_result.get('prediction', 'N/A')}")
                                st.write(f"- Rationale: {sample_result.get('rationale', 'N/A')[:100]}...")
                        else:
                            st.warning("Inference completed but no successful results. This may be due to low power mode performance limitations.")
                    
                    if operation in ["Fine-tune Model", "All Operations"]:
                        st.subheader("Fine-tuning Model...")
                        
                        # Check if we have training data
                        training_prompts = [p for p in prompts if 'rating' in p]
                        
                        if training_prompts:
                            # Use low power mode settings for fine-tuning
                            fine_tuner = FineTuner(
                                debug_mode=st.session_state.get('debug_mode', True),
                                fallback_mode=True  # Use contextualization instead of fine-tuning in low power mode
                            )
                            
                            # Convert prompts back to DataFrame format for fine-tuning
                            # This is a simplified approach - in practice, you'd want to preserve the original data
                            training_data = []
                            for prompt in training_prompts:
                                if 'rating' in prompt:
                                    training_data.append({
                                        'prompt': prompt['prompt'],
                                        'rating': prompt['rating']
                                    })
                            
                            # Fine-tune model using process_dataset
                            success, fine_tune_results = fine_tuner.process_dataset(
                                df=st.session_state.processed_data,
                                model_name=model_name,
                                rating_mapping=st.session_state.rating_mapping,
                                target_topics=st.session_state.current_topic
                            )
                            
                            results['fine_tuning'] = fine_tune_results
                            st.success("Fine-tuning completed!")
                        else:
                            st.warning("No training data available for fine-tuning.")
                    
                    if operation in ["Create Contextualizer", "All Operations"]:
                        st.subheader("Creating Contextualizer...")
                        
                        contextualizer = Contextualizer(debug_mode=st.session_state.get('debug_mode', True))
                        
                        # Process dataset with contextualizer
                        context_results_df, context_metadata = contextualizer.process_dataset(
                            df=st.session_state.processed_data,
                            topics=st.session_state.current_topic,
                            rating_mapping=st.session_state.rating_mapping,
                            model_name=model_name
                        )
                        
                        context_results = {
                            'results_df': context_results_df,
                            'metadata': context_metadata
                        }
                        
                        results['contextualizer'] = context_results
                        st.success("Contextualizer created!")
                    
                    # Store results
                    st.session_state.model_results = results
                    
                    st.success("All operations completed successfully!")
            
            except Exception as e:
                st.error(f"Error during model operations: {str(e)}")
                self.logger.error(f"Model operations failed: {e}", "gui", "step_4_model_operations", error=e)
        
        # Proceed button
        if st.session_state.get('inference_results') or st.session_state.get('model_results'):
            if st.button("Proceed to Results & Export", type="primary"):
                st.session_state.current_step = 5
                st.rerun()
    
    def step_5_results_export(self):
        """Step 5: Results visualization and export."""
        st.header("Results & Export")
        
        # Results overview
        st.subheader("Results Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.processed_data is not None:
                st.metric("Total Rows", len(st.session_state.processed_data))
        with col2:
            if st.session_state.formatted_prompts is not None:
                st.metric("Total Prompts", len(st.session_state.formatted_prompts))
        with col3:
            if st.session_state.inference_results is not None:
                st.metric("Inference Results", len(st.session_state.inference_results))
        
        # Inference results
        if st.session_state.inference_results:
            st.subheader("Inference Results")
            
            # Convert to DataFrame for display
            results_df = pd.DataFrame(st.session_state.inference_results)
            
            # Display results
            st.dataframe(results_df, use_container_width=True)
            
            # Results analysis
            if 'prediction' in results_df.columns and 'rating' in results_df.columns:
                st.subheader("Prediction Analysis")
                
                # Accuracy calculation
                correct_predictions = (results_df['prediction'] == results_df['rating']).sum()
                total_predictions = len(results_df)
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("Correct Predictions", correct_predictions)
                with col3:
                    st.metric("Total Predictions", total_predictions)
                
                # Confusion matrix
                if len(results_df) > 0:
                    st.subheader("Confusion Matrix")
                    confusion_data = pd.crosstab(
                        results_df['rating'], 
                        results_df['prediction'], 
                        margins=True
                    )
                    st.dataframe(confusion_data, use_container_width=True)
        
        # Export options
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export processed data
            if st.session_state.processed_data is not None:
                csv_data = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data (CSV)",
                    data=csv_data,
                    file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Export formatted prompts
            if st.session_state.formatted_prompts is not None:
                prompts_json = json.dumps(st.session_state.formatted_prompts, indent=2)
                st.download_button(
                    label="Download Formatted Prompts (JSON)",
                    data=prompts_json,
                    file_name=f"formatted_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            # Export inference results
            if st.session_state.inference_results is not None:
                results_csv = pd.DataFrame(st.session_state.inference_results).to_csv(index=False)
                st.download_button(
                    label="Download Inference Results (CSV)",
                    data=results_csv,
                    file_name=f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Export session report
            if st.button("Generate Session Report"):
                report = self.generate_session_report()
                st.download_button(
                    label="Download Session Report (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name=f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Session summary
        st.subheader("Session Summary")
        
        summary_data = {
            "Session ID": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "Topic": st.session_state.current_topic,
            "Data Rows": len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0,
            "Formatted Prompts": len(st.session_state.formatted_prompts) if st.session_state.formatted_prompts is not None else 0,
            "Inference Results": len(st.session_state.inference_results) if st.session_state.inference_results is not None else 0,
            "Rating Mapping": st.session_state.rating_mapping
        }
        
        summary_df = pd.DataFrame([summary_data])
        st.dataframe(summary_df, use_container_width=True)
    
    def generate_session_report(self) -> Dict[str, Any]:
        """Generate a comprehensive session report."""
        report = {
            "session_info": {
                "session_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
                "timestamp": datetime.now().isoformat(),
                "topic": st.session_state.current_topic,
                "model_name": st.session_state.get('model_name', 'llama3.2:3b'),
                "server_url": st.session_state.get('server_url', 'http://localhost:11434')
            },
            "data_summary": {
                "total_rows": len(st.session_state.processed_data) if st.session_state.processed_data is not None else 0,
                "total_columns": len(st.session_state.processed_data.columns) if st.session_state.processed_data is not None else 0,
                "has_expert_ratings": 'Expert Rating' in st.session_state.processed_data.columns if st.session_state.processed_data is not None else False
            },
            "processing_summary": {
                "rating_mapping": st.session_state.rating_mapping,
                "formatted_prompts": len(st.session_state.formatted_prompts) if st.session_state.formatted_prompts is not None else 0
            },
            "results_summary": {
                "inference_results": len(st.session_state.inference_results) if st.session_state.inference_results is not None else 0,
                "model_results": bool(st.session_state.get('model_results'))
            }
        }
        
        return report
    
    def run(self):
        """Main method to run the GUI."""
        try:
            # Render components
            self.render_header()
            self.render_sidebar()
            
            # Render current step
            if st.session_state.current_step == 1:
                self.step_1_data_upload()
            elif st.session_state.current_step == 2:
                self.step_2_schema_mapping()
            elif st.session_state.current_step == 3:
                self.step_3_prompt_formatting()
            elif st.session_state.current_step == 4:
                self.step_4_model_operations()
            elif st.session_state.current_step == 5:
                self.step_5_results_export()
            
        except Exception as e:
            st.error(f"GUI Error: {str(e)}")
            self.logger.error(f"GUI error: {e}", "gui", "run", error=e)


def main():
    """Main function to run the EpiTuner GUI."""
    gui = EpiTunerGUI()
    gui.run()


if __name__ == "__main__":
    main() 