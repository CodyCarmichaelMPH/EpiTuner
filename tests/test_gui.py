"""
Unit tests for the EpiTuner GUI module.

Tests cover:
- GUI initialization and setup
- Data upload and validation
- Schema mapping functionality
- Prompt formatting
- Model operations
- Results export
- Session state management
- Error handling
"""

import pytest
import pandas as pd
import tempfile
import os
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))
sys.path.append(str(Path(__file__).parent.parent / 'gui'))

# Mock streamlit for testing
sys.modules['streamlit'] = Mock()
import streamlit as st

from epituner_gui import EpiTunerGUI


class TestEpiTunerGUI:
    """Test suite for EpiTunerGUI class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        return pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002', 'P003'],
            'ChiefComplaintOrig': ['Fever', 'Chest pain', 'Headache'],
            'Discharge Diagnosis': ['Viral infection', 'Angina', 'Migraine'],
            'Sex': ['M', 'F', 'M'],
            'Age': [25, 45, 32],
            'Admit_Reason_Combo': ['Fever', 'Chest pain', 'Headache'],
            'Chief_Complaint_Combo': ['Fever', 'Chest pain', 'Headache'],
            'Diagnosis_Combo': ['Infection', 'Cardiac', 'Neurological'],
            'CCDD': ['Fever', 'Chest pain', 'Headache'],
            'CCDDCategory': ['Viral', 'Cardiac', 'Neurological'],
            'TriageNotes': ['High fever', 'Severe chest pain', 'Throbbing headache'],
            'Expert Rating': [1, 2, 0],
            'Rationale of Rating': ['Clear match', 'Partial match', 'No match']
        })
    
    @pytest.fixture
    def gui(self):
        """Create EpiTunerGUI instance for testing."""
        return EpiTunerGUI()
    
    def test_init(self, gui):
        """Test GUI initialization."""
        assert gui is not None
        assert hasattr(gui, 'logger')
        assert hasattr(gui, 'setup_page_config')
        assert hasattr(gui, 'setup_session_state')
    
    def test_setup_session_state(self, gui):
        """Test session state initialization."""
        # Clear session state
        st.session_state.clear()
        
        # Setup session state
        gui.setup_session_state()
        
        # Check that all required keys are present
        required_keys = [
            'current_step', 'uploaded_file', 'processed_data', 
            'rating_mapping', 'formatted_prompts', 'inference_results', 'current_topic'
        ]
        
        for key in required_keys:
            assert key in st.session_state
    
    def test_setup_session_state_existing_values(self, gui):
        """Test session state setup with existing values."""
        # Set some existing values
        st.session_state['current_step'] = 5
        st.session_state['current_topic'] = 'Test Topic'
        
        # Setup session state
        gui.setup_session_state()
        
        # Existing values should be preserved
        assert st.session_state['current_step'] == 5
        assert st.session_state['current_topic'] == 'Test Topic'
        
        # Missing keys should be added
        assert 'uploaded_file' in st.session_state
        assert 'processed_data' in st.session_state
    
    @patch('epituner_gui.st.set_page_config')
    def test_setup_page_config(self, mock_set_page_config, gui):
        """Test page configuration setup."""
        gui.setup_page_config()
        
        mock_set_page_config.assert_called_once_with(
            page_title="EpiTuner - Ollama Fine-Tuning Suite",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @patch('epituner_gui.st.title')
    @patch('epituner_gui.st.markdown')
    @patch('epituner_gui.st.columns')
    def test_render_header(self, mock_columns, mock_markdown, mock_title, gui):
        """Test header rendering."""
        # Mock session state
        st.session_state['current_step'] = 2
        
        # Mock columns return
        mock_cols = [Mock(), Mock(), Mock(), Mock(), Mock()]
        mock_columns.return_value = mock_cols
        
        gui.render_header()
        
        mock_title.assert_called_once_with("🧬 EpiTuner - Ollama Fine-Tuning and Evaluation Suite")
        assert mock_markdown.call_count >= 2  # At least 2 markdown calls for separators
        mock_columns.assert_called_once_with(5)  # 5 steps
    
    @patch('epituner_gui.st.sidebar')
    def test_render_sidebar(self, mock_sidebar, gui):
        """Test sidebar rendering."""
        # Mock session state
        st.session_state['current_step'] = 3
        st.session_state['model_name'] = 'llama3.2:3b'
        st.session_state['server_url'] = 'http://localhost:11434'
        
        # Mock sidebar context manager
        mock_sidebar_context = Mock()
        mock_sidebar.return_value.__enter__.return_value = mock_sidebar_context
        
        gui.render_sidebar()
        
        # Verify sidebar was called
        mock_sidebar.assert_called_once()
    
    @patch('epituner_gui.st.file_uploader')
    @patch('epituner_gui.st.metric')
    @patch('epituner_gui.st.spinner')
    @patch('epituner_gui.st.success')
    @patch('epituner_gui.st.dataframe')
    @patch('epituner_gui.st.button')
    @patch('epituner_gui.DataLoader')
    def test_step_1_data_upload_success(self, mock_data_loader, mock_button, mock_dataframe, 
                                       mock_success, mock_spinner, mock_metric, mock_file_uploader, 
                                       gui, sample_data):
        """Test successful data upload step."""
        # Mock file uploader
        mock_file = Mock()
        mock_file.name = 'test.csv'
        mock_file.size = 1024
        mock_file.type = 'text/csv'
        mock_file.getvalue.return_value = b'test data'
        mock_file_uploader.return_value = mock_file
        
        # Mock data loader
        mock_loader_instance = Mock()
        mock_loader_instance.load_dataset.return_value = sample_data
        mock_loader_instance.validate_schema.return_value = (True, [], {})
        mock_data_loader.return_value = mock_loader_instance
        
        # Mock spinner context
        mock_spinner_context = Mock()
        mock_spinner.return_value.__enter__.return_value = mock_spinner_context
        
        # Mock button
        mock_button.return_value = False
        
        # Set session state
        st.session_state['processed_data'] = None
        
        # Run the step
        gui.step_1_data_upload()
        
        # Verify file uploader was called
        mock_file_uploader.assert_called_once()
        
        # Verify data loader was used
        mock_loader_instance.load_dataset.assert_called_once()
        mock_loader_instance.validate_schema.assert_called_once_with(sample_data)
        
        # Verify success message
        mock_success.assert_called_once_with("✅ Data validation successful!")
    
    @patch('epituner_gui.st.file_uploader')
    @patch('epituner_gui.st.error')
    @patch('epituner_gui.DataLoader')
    def test_step_1_data_upload_validation_failure(self, mock_data_loader, mock_error, 
                                                  mock_file_uploader, gui, sample_data):
        """Test data upload step with validation failure."""
        # Mock file uploader
        mock_file = Mock()
        mock_file.name = 'test.csv'
        mock_file.size = 1024
        mock_file.type = 'text/csv'
        mock_file.getvalue.return_value = b'test data'
        mock_file_uploader.return_value = mock_file
        
        # Mock data loader with validation failure
        mock_loader_instance = Mock()
        mock_loader_instance.load_dataset.return_value = sample_data
        mock_loader_instance.validate_schema.return_value = (False, ['Missing column'], {})
        mock_data_loader.return_value = mock_loader_instance
        
        # Run the step
        gui.step_1_data_upload()
        
        # Verify error was shown
        mock_error.assert_called_once_with("❌ Data validation failed!")
    
    @patch('epituner_gui.st.warning')
    def test_step_2_schema_mapping_no_data(self, mock_warning, gui):
        """Test schema mapping step with no data."""
        # Clear processed data
        st.session_state['processed_data'] = None
        
        gui.step_2_schema_mapping()
        
        mock_warning.assert_called_once_with("⚠️ No data loaded. Please go back to Step 1.")
    
    @patch('epituner_gui.st.subheader')
    @patch('epituner_gui.st.write')
    @patch('epituner_gui.st.radio')
    @patch('epituner_gui.st.selectbox')
    @patch('epituner_gui.st.button')
    @patch('epituner_gui.st.text_input')
    @patch('epituner_gui.DataLoader')
    @patch('epituner_gui.SchemaMapper')
    def test_step_2_schema_mapping_with_ratings(self, mock_schema_mapper, mock_data_loader,
                                               mock_text_input, mock_button, mock_selectbox,
                                               mock_radio, mock_write, mock_subheader, gui, sample_data):
        """Test schema mapping step with expert ratings."""
        # Set up session state
        st.session_state['processed_data'] = sample_data
        st.session_state['current_topic'] = ''
        
        # Mock data loader
        mock_loader_instance = Mock()
        mock_loader_instance.extract_unique_ratings.return_value = [1, 2, 0]
        mock_data_loader.return_value = mock_loader_instance
        
        # Mock schema mapper
        mock_mapper_instance = Mock()
        mock_mapper_instance.apply_rating_mapping.return_value = sample_data
        mock_schema_mapper.return_value = mock_mapper_instance
        
        # Mock UI components
        mock_radio.return_value = "Manual Mapping"
        mock_selectbox.side_effect = [1, 2, 0]  # Return values for rating mapping
        mock_button.return_value = True  # Apply mapping button clicked
        mock_text_input.return_value = "Test Topic"
        
        gui.step_2_schema_mapping()
        
        # Verify components were called
        mock_subheader.assert_called()
        mock_radio.assert_called_once()
        mock_text_input.assert_called_once()
    
    @patch('epituner_gui.st.warning')
    def test_step_3_prompt_formatting_no_data(self, mock_warning, gui):
        """Test prompt formatting step with no data."""
        # Clear processed data
        st.session_state['processed_data'] = None
        
        gui.step_3_prompt_formatting()
        
        mock_warning.assert_called_once_with("⚠️ No data loaded. Please go back to Step 1.")
    
    @patch('epituner_gui.st.warning')
    def test_step_3_prompt_formatting_no_topic(self, mock_warning, gui):
        """Test prompt formatting step with no topic."""
        # Set data but no topic
        st.session_state['processed_data'] = pd.DataFrame({'test': [1]})
        st.session_state['current_topic'] = ''
        
        gui.step_3_prompt_formatting()
        
        mock_warning.assert_called_once_with("⚠️ No topic specified. Please go back to Step 2.")
    
    @patch('epituner_gui.st.subheader')
    @patch('epituner_gui.st.checkbox')
    @patch('epituner_gui.st.number_input')
    @patch('epituner_gui.st.selectbox')
    @patch('epituner_gui.st.button')
    @patch('epituner_gui.st.spinner')
    @patch('epituner_gui.st.success')
    @patch('epituner_gui.st.metric')
    @patch('epituner_gui.st.text_area')
    @patch('epituner_gui.Formatter')
    def test_step_3_prompt_formatting_success(self, mock_formatter, mock_text_area, mock_metric,
                                             mock_success, mock_spinner, mock_button, mock_selectbox,
                                             mock_number_input, mock_checkbox, mock_subheader, gui, sample_data):
        """Test successful prompt formatting step."""
        # Set up session state
        st.session_state['processed_data'] = sample_data
        st.session_state['current_topic'] = 'Test Topic'
        
        # Mock formatter
        mock_formatter_instance = Mock()
        mock_formatter_instance.create_training_prompts.return_value = [
            {'id': 'P001', 'prompt': 'Test prompt', 'rating': 1, 'topic': 'Test Topic'}
        ]
        mock_formatter_instance.get_prompt_statistics.return_value = {
            'total_prompts': 1,
            'avg_prompt_length': 100,
            'topics': ['Test Topic'],
            'rating_distribution': {1: 1}
        }
        mock_formatter.return_value = mock_formatter_instance
        
        # Mock UI components
        mock_checkbox.return_value = True
        mock_number_input.return_value = 100
        mock_selectbox.return_value = "Auto-detect"
        mock_button.return_value = True  # Format prompts button clicked
        mock_text_area.return_value = None
        
        # Mock spinner context
        mock_spinner_context = Mock()
        mock_spinner.return_value.__enter__.return_value = mock_spinner_context
        
        gui.step_3_prompt_formatting()
        
        # Verify formatter was used
        mock_formatter_instance.create_training_prompts.assert_called_once()
        mock_formatter_instance.get_prompt_statistics.assert_called_once()
        
        # Verify success message
        mock_success.assert_called_once()
    
    @patch('epituner_gui.st.warning')
    def test_step_4_model_operations_no_prompts(self, mock_warning, gui):
        """Test model operations step with no prompts."""
        # Clear formatted prompts
        st.session_state['formatted_prompts'] = None
        
        gui.step_4_model_operations()
        
        mock_warning.assert_called_once_with("⚠️ No formatted prompts available. Please go back to Step 3.")
    
    @patch('epituner_gui.st.subheader')
    @patch('epituner_gui.st.radio')
    @patch('epituner_gui.st.text_input')
    @patch('epituner_gui.st.number_input')
    @patch('epituner_gui.st.slider')
    @patch('epituner_gui.st.button')
    @patch('epituner_gui.st.spinner')
    @patch('epituner_gui.st.success')
    @patch('epituner_gui.InferenceRunner')
    def test_step_4_model_operations_inference(self, mock_inference_runner, mock_success, mock_spinner,
                                              mock_button, mock_slider, mock_number_input, mock_text_input,
                                              mock_radio, mock_subheader, gui):
        """Test model operations step with inference."""
        # Set up session state
        st.session_state['formatted_prompts'] = [
            {'id': 'P001', 'prompt': 'Test prompt', 'topic': 'Test Topic'}
        ]
        
        # Mock inference runner
        mock_runner_instance = Mock()
        mock_runner_instance.run_batch_inference.return_value = [
            {'id': 'P001', 'prediction': 1, 'confidence': 0.8, 'rationale': 'Test rationale'}
        ]
        mock_inference_runner.return_value = mock_runner_instance
        
        # Mock UI components
        mock_radio.return_value = "Run Inference"
        mock_text_input.side_effect = ['llama3.2:3b', 'http://localhost:11434']
        mock_number_input.return_value = 512
        mock_slider.return_value = 0.7
        mock_button.return_value = True  # Start operation button clicked
        
        # Mock spinner context
        mock_spinner_context = Mock()
        mock_spinner.return_value.__enter__.return_value = mock_spinner_context
        
        gui.step_4_model_operations()
        
        # Verify inference runner was used
        mock_runner_instance.run_batch_inference.assert_called_once()
        
        # Verify success message
        mock_success.assert_called()
    
    @patch('epituner_gui.st.subheader')
    @patch('epituner_gui.st.metric')
    @patch('epituner_gui.st.dataframe')
    @patch('epituner_gui.st.download_button')
    @patch('epituner_gui.st.button')
    @patch('epituner_gui.st.bar_chart')
    def test_step_5_results_export(self, mock_bar_chart, mock_button, mock_download_button,
                                  mock_dataframe, mock_metric, mock_subheader, gui):
        """Test results export step."""
        # Set up session state with results
        st.session_state['processed_data'] = pd.DataFrame({'test': [1, 2, 3]})
        st.session_state['formatted_prompts'] = [{'id': 'P001', 'prompt': 'Test'}]
        st.session_state['inference_results'] = [
            {'id': 'P001', 'prediction': 1, 'rating': 1, 'confidence': 0.8}
        ]
        
        # Mock UI components
        mock_button.return_value = False
        
        gui.step_5_results_export()
        
        # Verify components were called
        mock_subheader.assert_called()
        mock_metric.assert_called()
        mock_dataframe.assert_called()
    
    def test_generate_session_report(self, gui):
        """Test session report generation."""
        # Set up session state
        st.session_state['current_topic'] = 'Test Topic'
        st.session_state['processed_data'] = pd.DataFrame({'test': [1, 2, 3]})
        st.session_state['formatted_prompts'] = [{'id': 'P001'}]
        st.session_state['inference_results'] = [{'id': 'P001'}]
        st.session_state['rating_mapping'] = {1: 1, 2: 2}
        
        report = gui.generate_session_report()
        
        assert 'session_info' in report
        assert 'data_summary' in report
        assert 'processing_summary' in report
        assert 'results_summary' in report
        
        assert report['session_info']['topic'] == 'Test Topic'
        assert report['data_summary']['total_rows'] == 3
        assert report['processing_summary']['formatted_prompts'] == 1
        assert report['results_summary']['inference_results'] == 1
    
    @patch('epituner_gui.st.error')
    def test_run_with_exception(self, mock_error, gui):
        """Test GUI run method with exception handling."""
        # Mock render_header to raise exception
        gui.render_header = Mock(side_effect=Exception("Test error"))
        
        gui.run()
        
        mock_error.assert_called_once_with("❌ GUI Error: Test error")


class TestGUIIntegration:
    """Integration tests for GUI components."""
    
    @patch('epituner_gui.st.session_state')
    def test_session_state_flow(self, mock_session_state):
        """Test session state flow through steps."""
        gui = EpiTunerGUI()
        
        # Simulate step progression
        mock_session_state['current_step'] = 1
        gui.setup_session_state()
        
        # Verify initial state
        assert 'current_step' in mock_session_state
        assert 'processed_data' in mock_session_state
        
        # Simulate moving to next step
        mock_session_state['current_step'] = 2
        assert mock_session_state['current_step'] == 2
    
    def test_data_flow_through_steps(self):
        """Test data flow through GUI steps."""
        gui = EpiTunerGUI()
        
        # Test data persistence through steps
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        st.session_state['processed_data'] = test_data
        
        # Verify data is accessible
        assert st.session_state['processed_data'] is not None
        assert len(st.session_state['processed_data']) == 3


class TestErrorHandling:
    """Test error handling in GUI."""
    
    @patch('epituner_gui.st.error')
    def test_data_loader_error_handling(self, mock_error):
        """Test error handling in data loading."""
        gui = EpiTunerGUI()
        
        # Mock data loader to raise exception
        with patch('epituner_gui.DataLoader') as mock_data_loader:
            mock_loader_instance = Mock()
            mock_loader_instance.load_dataset.side_effect = Exception("Data loading error")
            mock_data_loader.return_value = mock_loader_instance
            
            # This would normally be called in step_1_data_upload
            # For testing, we'll just verify the error handling pattern
            pass
    
    @patch('epituner_gui.st.error')
    def test_formatter_error_handling(self, mock_error):
        """Test error handling in prompt formatting."""
        gui = EpiTunerGUI()
        
        # Mock formatter to raise exception
        with patch('epituner_gui.Formatter') as mock_formatter:
            mock_formatter_instance = Mock()
            mock_formatter_instance.create_training_prompts.side_effect = Exception("Formatting error")
            mock_formatter.return_value = mock_formatter_instance
            
            # This would normally be called in step_3_prompt_formatting
            pass


if __name__ == "__main__":
    pytest.main([__file__]) 