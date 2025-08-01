"""
Unit tests for the FineTuner module.

Tests cover:
- Fine-tuning capability checking
- Training data preparation
- Configuration creation
- Fine-tuning execution
- Model set configuration
- Validation and error handling
- Edge cases
"""

import pytest
import pandas as pd
import tempfile
import os
import json
import yaml
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from fine_tuner import (
    FineTuner, FineTuningNotSupportedError, DatasetTooSmallWarning,
    TrainingFailedError, ModelSetConfigError
)


class TestFineTuner:
    """Test suite for FineTuner class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        return pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'ChiefComplaintOrig': ['Fever', 'Chest pain', 'Headache', 'Nausea', 'Fatigue'],
            'Discharge Diagnosis': ['Viral infection', 'Angina', 'Migraine', 'Gastritis', 'Anemia'],
            'Sex': ['M', 'F', 'M', 'F', 'M'],
            'Age': [25, 45, 32, 28, 55],
            'Admit_Reason_Combo': ['Fever', 'Chest pain', 'Headache', 'Nausea', 'Fatigue'],
            'Chief_Complaint_Combo': ['Fever', 'Chest pain', 'Headache', 'Nausea', 'Fatigue'],
            'Diagnosis_Combo': ['Infection', 'Cardiac', 'Neurological', 'GI', 'Hematological'],
            'CCDD': ['Fever', 'Chest pain', 'Headache', 'Nausea', 'Fatigue'],
            'CCDDCategory': ['Viral', 'Cardiac', 'Neurological', 'GI', 'Hematological'],
            'TriageNotes': ['High fever', 'Severe chest pain', 'Throbbing headache', 'Vomiting', 'Pale'],
            'Expert Rating': ['Match', 'Partial Match', 'Match', 'Does Not Match', 'Partial Match'],
            'Rationale of Rating': ['Clear match', 'Partial match', 'Strong match', 'No match', 'Weak match'],
            'Standardized_Rating': [1, 2, 1, 0, 2],
            'formatted_prompt': [
                'Context: Patient Info: Age 25, Sex M\nChief Complaint: Fever\nTask: Evaluate alignment...',
                'Context: Patient Info: Age 45, Sex F\nChief Complaint: Chest pain\nTask: Evaluate alignment...',
                'Context: Patient Info: Age 32, Sex M\nChief Complaint: Headache\nTask: Evaluate alignment...',
                'Context: Patient Info: Age 28, Sex F\nChief Complaint: Nausea\nTask: Evaluate alignment...',
                'Context: Patient Info: Age 55, Sex M\nChief Complaint: Fatigue\nTask: Evaluate alignment...'
            ]
        })
    
    @pytest.fixture
    def fine_tuner(self):
        """Create FineTuner instance for testing."""
        return FineTuner(debug_mode=True, fallback_mode=True)
    
    @pytest.fixture
    def rating_mapping(self):
        """Create rating mapping for testing."""
        return {
            'Match': 1,
            'Does Not Match': 0,
            'Unknown': -1,
            'Partial Match': 2
        }
    
    def test_init(self, fine_tuner):
        """Test FineTuner initialization."""
        assert fine_tuner.debug_mode is True
        assert fine_tuner.fallback_mode is True
        assert hasattr(fine_tuner, 'logger')
    
    @patch('subprocess.run')
    def test_check_fine_tuning_capability_success(self, mock_run, fine_tuner):
        """Test successful fine-tuning capability check."""
        # Mock successful ollama version check
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ollama version 0.1.0"
        
        result = fine_tuner.check_fine_tuning_capability("llama2")
        assert result is True
    
    @patch('subprocess.run')
    def test_check_fine_tuning_capability_ollama_not_found(self, mock_run, fine_tuner):
        """Test fine-tuning capability check when Ollama not found."""
        mock_run.side_effect = FileNotFoundError("ollama not found")
        
        result = fine_tuner.check_fine_tuning_capability("llama2")
        assert result is False
    
    @patch('subprocess.run')
    def test_check_fine_tuning_capability_model_not_found(self, mock_run, fine_tuner):
        """Test fine-tuning capability check when model not found."""
        # Mock successful ollama version check
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ollama version 0.1.0"
        
        # Mock model not found
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "model not found"
        
        result = fine_tuner.check_fine_tuning_capability("nonexistent-model")
        assert result is False
    
    def test_prepare_training_data(self, fine_tuner, sample_data, rating_mapping):
        """Test training data preparation."""
        training_data = fine_tuner.prepare_training_data(sample_data, rating_mapping)
        
        assert len(training_data) == 5
        assert all(isinstance(example, dict) for example in training_data)
        assert all("input" in example and "output" in example for example in training_data)
        
        # Check first example
        first_example = training_data[0]
        assert "Context: Patient Info: Age 25, Sex M" in first_example["input"]
        assert "Rating: 1, Rationale: Clear match" in first_example["output"]
    
    def test_prepare_training_data_without_formatted_prompt(self, fine_tuner, sample_data, rating_mapping):
        """Test training data preparation without formatted_prompt column."""
        # Remove formatted_prompt column
        sample_data_no_prompt = sample_data.drop(columns=['formatted_prompt'])
        
        training_data = fine_tuner.prepare_training_data(sample_data_no_prompt, rating_mapping)
        
        assert len(training_data) == 5
        # Should create context from available fields
        assert all("Patient: Age" in example["input"] for example in training_data)
    
    def test_create_training_config(self, fine_tuner, sample_data, rating_mapping):
        """Test training configuration creation."""
        training_data = fine_tuner.prepare_training_data(sample_data, rating_mapping)
        
        config = fine_tuner.create_training_config(
            "llama2", training_data, epochs=5, learning_rate=0.001, batch_size=8
        )
        
        assert config["model"] == "llama2"
        assert config["epochs"] == 5
        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 8
        assert config["training_data"] == training_data
    
    def test_create_training_config_defaults(self, fine_tuner, sample_data, rating_mapping):
        """Test training configuration creation with defaults."""
        training_data = fine_tuner.prepare_training_data(sample_data, rating_mapping)
        
        config = fine_tuner.create_training_config("llama2", training_data)
        
        assert config["epochs"] == fine_tuner.DEFAULT_EPOCHS
        assert config["learning_rate"] == fine_tuner.DEFAULT_LEARNING_RATE
        assert config["batch_size"] == fine_tuner.DEFAULT_BATCH_SIZE
    
    def test_save_training_data_jsonl(self, fine_tuner, sample_data, rating_mapping):
        """Test saving training data in JSONL format."""
        training_data = fine_tuner.prepare_training_data(sample_data, rating_mapping)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            fine_tuner.save_training_data_jsonl(training_data, output_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 5
                
                # Check first line is valid JSON
                first_example = json.loads(lines[0])
                assert "input" in first_example
                assert "output" in first_example
        finally:
            os.unlink(output_path)
    
    def test_save_training_config_yaml(self, fine_tuner, sample_data, rating_mapping):
        """Test saving training configuration in YAML format."""
        training_data = fine_tuner.prepare_training_data(sample_data, rating_mapping)
        config = fine_tuner.create_training_config("llama2", training_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            fine_tuner.save_training_config_yaml(config, output_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                assert loaded_config["model"] == "llama2"
                assert loaded_config["epochs"] == fine_tuner.DEFAULT_EPOCHS
        finally:
            os.unlink(output_path)
    
    @patch('subprocess.Popen')
    def test_execute_fine_tuning_success(self, mock_popen, fine_tuner, sample_data, rating_mapping):
        """Test successful fine-tuning execution."""
        training_data = fine_tuner.prepare_training_data(sample_data, rating_mapping)
        config = fine_tuner.create_training_config("llama2", training_data)
        
        # Mock successful fine-tuning process
        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = "Training progress: 50%"
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        
        success = fine_tuner.execute_fine_tuning("llama2", training_data, config, "llama2-finetuned")
        assert success is True
    
    @patch('subprocess.Popen')
    def test_execute_fine_tuning_failure(self, mock_popen, fine_tuner, sample_data, rating_mapping):
        """Test failed fine-tuning execution."""
        training_data = fine_tuner.prepare_training_data(sample_data, rating_mapping)
        config = fine_tuner.create_training_config("llama2", training_data)
        
        # Mock failed fine-tuning process
        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = ""
        mock_process.poll.return_value = 1
        mock_process.stderr.read.return_value = "Training failed"
        mock_popen.return_value = mock_process
        
        success = fine_tuner.execute_fine_tuning("llama2", training_data, config, "llama2-finetuned")
        assert success is False
    
    def test_create_model_set_config(self, fine_tuner, sample_data, rating_mapping):
        """Test model set configuration creation."""
        target_topics = "respiratory infections and cardiac conditions"
        
        config = fine_tuner.create_model_set_config("llama2", sample_data, rating_mapping, target_topics)
        
        assert config["base_model"] == "llama2"
        assert config["target_topics"] == target_topics
        assert "system_prompt" in config
        assert "few_shot_examples" in config
        assert len(config["few_shot_examples"]) > 0
        assert config["dataset_size"] == 5
    
    def test_create_system_prompt(self, fine_tuner, rating_mapping):
        """Test system prompt creation."""
        target_topics = "respiratory infections"
        
        system_prompt = fine_tuner._create_system_prompt(rating_mapping, target_topics)
        
        assert "expert evaluator" in system_prompt.lower()
        assert target_topics in system_prompt
        assert "rating:" in system_prompt.lower()
        assert "rationale:" in system_prompt.lower()
    
    def test_select_few_shot_examples(self, fine_tuner, sample_data, rating_mapping):
        """Test few-shot example selection."""
        examples = fine_tuner._select_few_shot_examples(sample_data, rating_mapping, num_examples=3)
        
        assert len(examples) <= 3
        assert all(isinstance(example, dict) for example in examples)
        assert all("input" in example and "output" in example for example in examples)
    
    def test_save_model_set_config(self, fine_tuner, sample_data, rating_mapping):
        """Test saving model set configuration."""
        config = fine_tuner.create_model_set_config("llama2", sample_data, rating_mapping, "test topics")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            fine_tuner.save_model_set_config(config, output_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                assert loaded_config["base_model"] == "llama2"
                assert "few_shot_examples" in loaded_config
        finally:
            os.unlink(output_path)
    
    @patch('subprocess.run')
    def test_validate_fine_tuned_model_success(self, mock_run, fine_tuner):
        """Test successful fine-tuned model validation."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Model information"
        
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        results = fine_tuner.validate_fine_tuned_model("llama2-finetuned", test_df)
        
        assert results["model_name"] == "llama2-finetuned"
        assert results["validation_status"] == "success"
        assert results["model_available"] is True
    
    @patch('subprocess.run')
    def test_validate_fine_tuned_model_failure(self, mock_run, fine_tuner):
        """Test failed fine-tuned model validation."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Model not found"
        
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        results = fine_tuner.validate_fine_tuned_model("llama2-finetuned", test_df)
        
        assert results["validation_status"] == "failed"
        assert results["model_available"] is False
    
    def test_process_dataset_fine_tuning_approach(self, fine_tuner, sample_data, rating_mapping):
        """Test dataset processing with fine-tuning approach."""
        with patch.object(fine_tuner, 'check_fine_tuning_capability', return_value=True):
            with patch.object(fine_tuner, 'execute_fine_tuning', return_value=True):
                with patch.object(fine_tuner, 'validate_fine_tuned_model') as mock_validate:
                    mock_validate.return_value = {
                        "model_name": "llama2-finetuned",
                        "validation_status": "success",
                        "model_available": True
                    }
                    
                    success, metadata = fine_tuner.process_dataset(
                        sample_data, "llama2", rating_mapping, "test topics"
                    )
                    
                    assert success is True
                    assert metadata["fine_tuning_info"]["approach"] == "fine_tuning"
                    assert metadata["fine_tuning_info"]["success"] is True
    
    def test_process_dataset_model_set_approach(self, fine_tuner, sample_data, rating_mapping):
        """Test dataset processing with model set configuration approach."""
        with patch.object(fine_tuner, 'check_fine_tuning_capability', return_value=False):
            success, metadata = fine_tuner.process_dataset(
                sample_data, "llama2", rating_mapping, "test topics"
            )
            
            assert success is True
            assert metadata["model_set_info"]["approach"] == "model_set_configuration"
            assert metadata["model_set_info"]["success"] is True
    
    def test_process_dataset_too_small_with_fallback(self, fine_tuner, rating_mapping):
        """Test dataset processing with small dataset and fallback enabled."""
        # Create small dataset
        small_data = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'formatted_prompt': ['Test prompt'],
            'Standardized_Rating': [1],
            'Rationale of Rating': ['Test rationale']
        })
        
        with patch.object(fine_tuner, 'check_fine_tuning_capability', return_value=True):
            success, metadata = fine_tuner.process_dataset(
                small_data, "llama2", rating_mapping, "test topics"
            )
            
            # Should fall back to model set configuration
            assert success is True
            assert metadata["model_set_info"]["approach"] == "model_set_configuration"
    
    def test_process_dataset_too_small_without_fallback(self, fine_tuner, rating_mapping):
        """Test dataset processing with small dataset and fallback disabled."""
        fine_tuner.fallback_mode = False
        
        # Create small dataset
        small_data = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'formatted_prompt': ['Test prompt'],
            'Standardized_Rating': [1],
            'Rationale of Rating': ['Test rationale']
        })
        
        with patch.object(fine_tuner, 'check_fine_tuning_capability', return_value=True):
            with pytest.raises(DatasetTooSmallWarning):
                fine_tuner.process_dataset(
                    small_data, "llama2", rating_mapping, "test topics"
                )
    
    def test_save_metadata(self, fine_tuner):
        """Test metadata saving."""
        metadata = {
            "test": "data",
            "nested": {"key": "value"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            fine_tuner.save_metadata(metadata, output_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_metadata = json.load(f)
                assert loaded_metadata["test"] == "data"
                assert loaded_metadata["nested"]["key"] == "value"
        finally:
            os.unlink(output_path)
    
    def test_edge_case_empty_dataset(self, fine_tuner, rating_mapping):
        """Test edge case with empty dataset."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(KeyError):
            fine_tuner.prepare_training_data(empty_data, rating_mapping)
    
    def test_edge_case_missing_rating_column(self, fine_tuner, sample_data, rating_mapping):
        """Test edge case with missing rating column."""
        data_no_rating = sample_data.drop(columns=['Standardized_Rating', 'Expert Rating'])
        
        training_data = fine_tuner.prepare_training_data(data_no_rating, rating_mapping)
        
        # Should handle missing rating gracefully
        assert len(training_data) == 5
        assert all("Rating: -1" in example["output"] for example in training_data)
    
    def test_edge_case_all_missing_values(self, fine_tuner, rating_mapping):
        """Test edge case with all missing values."""
        data_all_missing = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'formatted_prompt': [None],
            'Standardized_Rating': [None],
            'Rationale of Rating': [None]
        })
        
        training_data = fine_tuner.prepare_training_data(data_all_missing, rating_mapping)
        
        # Should handle missing values gracefully
        assert len(training_data) == 1
        assert "Rating: -1" in training_data[0]["output"]
        assert "No rationale provided" in training_data[0]["output"] 