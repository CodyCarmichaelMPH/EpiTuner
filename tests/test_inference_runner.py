"""
Tests for the Inference Runner module.

This module contains comprehensive tests for the InferenceRunner class,
covering all major functionality including model checking, inference,
response parsing, and error handling.
"""

import pytest
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add scripts directory to path
import sys
sys.path.append('scripts')

from inference_runner import InferenceRunner, ModelNotFoundError, ResponseParsingError, InferenceError


class TestInferenceRunner:
    """Test cases for InferenceRunner class."""
    
    @pytest.fixture
    def runner(self):
        """Create a test instance of InferenceRunner."""
        return InferenceRunner(debug_mode=True, batch_size=3, timeout=5, max_retries=2)
    
    @pytest.fixture
    def sample_prompts(self):
        """Create sample formatted prompts for testing."""
        return [
            {
                'C_BioSense_ID': 'test_001',
                'formatted_prompt': 'Test prompt 1 for cardiac conditions'
            },
            {
                'C_BioSense_ID': 'test_002', 
                'formatted_prompt': 'Test prompt 2 for cardiac conditions'
            },
            {
                'C_BioSense_ID': 'test_003',
                'formatted_prompt': 'Test prompt 3 for cardiac conditions'
            }
        ]
    
    @pytest.fixture
    def sample_df(self, sample_prompts):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(sample_prompts)
    
    def test_init(self, runner):
        """Test InferenceRunner initialization."""
        assert runner.debug_mode is True
        assert runner.batch_size == 3
        assert runner.timeout == 5
        assert runner.max_retries == 2
        assert runner.model_metadata == {}
        assert runner.logger is not None
    
    @patch('subprocess.run')
    def test_check_model_availability_success(self, mock_run, runner):
        """Test successful model availability check."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "llama2:latest\nmistral:latest\n"
        
        result = runner.check_model_availability('llama2')
        
        assert result is True
        mock_run.assert_called_once_with(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
    
    @patch('subprocess.run')
    def test_check_model_availability_not_found(self, mock_run, runner):
        """Test model availability check when model not found."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "mistral:latest\n"
        
        result = runner.check_model_availability('llama2')
        
        assert result is False
    
    @patch('subprocess.run')
    def test_check_model_availability_error(self, mock_run, runner):
        """Test model availability check with error."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Error: ollama not found"
        
        result = runner.check_model_availability('llama2')
        
        assert result is False
    
    @patch('subprocess.run')
    def test_check_model_availability_timeout(self, mock_run, runner):
        """Test model availability check with timeout."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired(['ollama', 'list'], 5)
        
        result = runner.check_model_availability('llama2')
        
        assert result is False
    
    @patch('subprocess.run')
    def test_get_model_metadata_success(self, mock_run, runner):
        """Test successful model metadata retrieval."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """
        Model: llama2:latest
        Size: 3.8GB
        Parameters: 7B
        """
        
        metadata = runner.get_model_metadata('llama2')
        
        assert metadata['name'] == 'llama2'
        assert metadata['available'] is True
        assert metadata['size'] == '3.8GB'
        assert metadata['parameters'] == '7B'
        assert runner.model_metadata == metadata
    
    @patch('subprocess.run')
    def test_get_model_metadata_not_found(self, mock_run, runner):
        """Test model metadata retrieval when model not found."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Error: model not found"
        
        with pytest.raises(ModelNotFoundError, match="Model 'llama2' not found"):
            runner.get_model_metadata('llama2')
    
    def test_extract_model_size(self, runner):
        """Test model size extraction from show output."""
        show_output = "Model: llama2\nSize: 3.8GB\nParameters: 7B"
        size = runner._extract_model_size(show_output)
        assert size == '3.8GB'
    
    def test_extract_model_size_not_found(self, runner):
        """Test model size extraction when size not found."""
        show_output = "Model: llama2\nParameters: 7B"
        size = runner._extract_model_size(show_output)
        assert size is None
    
    def test_extract_parameters(self, runner):
        """Test parameter extraction from show output."""
        show_output = "Model: llama2\nSize: 3.8GB\nParameters: 7B"
        params = runner._extract_parameters(show_output)
        assert params == '7B'
    
    def test_extract_parameters_not_found(self, runner):
        """Test parameter extraction when parameters not found."""
        show_output = "Model: llama2\nSize: 3.8GB"
        params = runner._extract_parameters(show_output)
        assert params is None
    
    @patch('subprocess.run')
    def test_run_inference_success(self, mock_run, runner):
        """Test successful inference run."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"prediction": 1, "rationale": "Clear match", "confidence": 0.9}'
        
        result = runner.run_inference('test prompt', 'llama2')
        
        assert result['prediction'] == 1
        assert result['rationale'] == 'Clear match'
        assert result['confidence'] == 0.9
        
        mock_run.assert_called_once_with(
            ['ollama', 'run', 'llama2', 'test prompt'],
            capture_output=True,
            text=True,
            timeout=5
        )
    
    @patch('subprocess.run')
    def test_run_inference_retry_success(self, mock_run, runner):
        """Test inference with retry on first failure."""
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr='Error'),
            MagicMock(returncode=0, stdout='{"prediction": 0, "rationale": "No match"}')
        ]
        
        result = runner.run_inference('test prompt', 'llama2')
        
        assert result['prediction'] == 0
        assert result['rationale'] == 'No match'
        assert len(mock_run.call_args_list) == 2
    
    @patch('subprocess.run')
    def test_run_inference_all_retries_fail(self, mock_run, runner):
        """Test inference when all retries fail."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = 'Persistent error'
        
        with pytest.raises(InferenceError, match="All 2 attempts failed"):
            runner.run_inference('test prompt', 'llama2')
        
        assert len(mock_run.call_args_list) == 2
    
    @patch('subprocess.run')
    def test_run_inference_timeout(self, mock_run, runner):
        """Test inference with timeout."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired(['ollama', 'run'], 5)
        
        with pytest.raises(InferenceError, match="All 2 attempts timed out"):
            runner.run_inference('test prompt', 'llama2')
    
    def test_parse_response_json(self, runner):
        """Test parsing JSON response."""
        response = '{"prediction": 1, "rationale": "Clear match", "confidence": 0.9}'
        result = runner._parse_response(response)
        
        assert result['prediction'] == 1
        assert result['rationale'] == 'Clear match'
        assert result['confidence'] == 0.9
    
    def test_parse_response_structured_text(self, runner):
        """Test parsing structured text response."""
        response = 'Prediction: 0\nRationale: No match found\nConfidence: 0.8'
        result = runner._parse_response(response)
        
        assert result['prediction'] == 0
        assert 'No match found' in result['rationale']
        assert result['confidence'] == 0.8
    
    def test_parse_response_unstructured(self, runner):
        """Test parsing unstructured response."""
        response = 'This is an unstructured response with no clear prediction'
        result = runner._parse_response(response)
        
        assert result['prediction'] is None
        assert result['rationale'] == 'This is an unstructured response with no clear prediction'
        assert result['confidence'] == 0.0
    
    def test_parse_response_no_prediction(self, runner):
        """Test parsing response with no prediction."""
        response = 'This response has no numeric prediction'
        
        result = runner._parse_response(response)
        assert result['prediction'] is None
        assert result['rationale'] == response
        assert result['confidence'] == 0.0
    
    def test_extract_prediction(self, runner):
        """Test prediction extraction."""
        response = 'The prediction is 1 and the confidence is 0.9'
        prediction = runner._extract_prediction(response)
        assert prediction == 1
    
    def test_extract_prediction_negative(self, runner):
        """Test negative prediction extraction."""
        response = 'The prediction is -1 for this case'
        prediction = runner._extract_prediction(response)
        assert prediction == -1
    
    def test_extract_prediction_none(self, runner):
        """Test prediction extraction when no numbers found."""
        response = 'No numeric prediction in this response'
        prediction = runner._extract_prediction(response)
        assert prediction is None
    
    def test_extract_rationale(self, runner):
        """Test rationale extraction."""
        response = 'Prediction: 1\nRationale: This is the explanation\nConfidence: 0.9'
        rationale = runner._extract_rationale(response)
        assert 'This is the explanation' in rationale
    
    def test_extract_rationale_fallback(self, runner):
        """Test rationale extraction fallback to full response."""
        response = 'Prediction: 1\nConfidence: 0.9'
        rationale = runner._extract_rationale(response)
        assert rationale == 'Prediction: 1\nConfidence: 0.9'
    
    def test_extract_confidence(self, runner):
        """Test confidence extraction."""
        response = 'Prediction: 1\nConfidence: 0.85\nRationale: Test'
        confidence = runner._extract_confidence(response)
        assert confidence == 0.85
    
    def test_extract_confidence_clamp(self, runner):
        """Test confidence extraction with clamping."""
        response = 'Prediction: 1\nConfidence: 1.5\nRationale: Test'
        confidence = runner._extract_confidence(response)
        assert confidence == 1.0
    
    def test_extract_confidence_none(self, runner):
        """Test confidence extraction when not found."""
        response = 'Prediction: 1\nRationale: Test'
        confidence = runner._extract_confidence(response)
        assert confidence is None
    
    @patch.object(InferenceRunner, 'run_inference')
    def test_run_batch_inference(self, mock_run_inference, runner, sample_prompts):
        """Test batch inference processing."""
        mock_run_inference.side_effect = [
            {'prediction': 1, 'rationale': 'Match', 'confidence': 0.9},
            {'prediction': 0, 'rationale': 'No match', 'confidence': 0.8},
            {'prediction': 1, 'rationale': 'Match', 'confidence': 0.95}
        ]
        
        prompts = [p['formatted_prompt'] for p in sample_prompts]
        row_ids = [p['C_BioSense_ID'] for p in sample_prompts]
        
        results = runner.run_batch_inference(prompts, 'llama2', row_ids)
        
        assert len(results) == 3
        assert results[0]['C_BioSense_ID'] == 'test_001'
        assert results[0]['prediction'] == 1
        assert results[1]['C_BioSense_ID'] == 'test_002'
        assert results[1]['prediction'] == 0
        assert results[2]['C_BioSense_ID'] == 'test_003'
        assert results[2]['prediction'] == 1
    
    @patch.object(InferenceRunner, 'run_inference')
    def test_run_batch_inference_with_error(self, mock_run_inference, runner, sample_prompts):
        """Test batch inference with error handling."""
        mock_run_inference.side_effect = [
            {'prediction': 1, 'rationale': 'Match', 'confidence': 0.9},
            InferenceError('Model error'),
            {'prediction': 1, 'rationale': 'Match', 'confidence': 0.95}
        ]
        
        prompts = [p['formatted_prompt'] for p in sample_prompts]
        row_ids = [p['C_BioSense_ID'] for p in sample_prompts]
        
        results = runner.run_batch_inference(prompts, 'llama2', row_ids)
        
        assert len(results) == 3
        assert results[0]['prediction'] == 1
        assert results[1]['prediction'] is None
        assert 'Error: Model error' in results[1]['rationale']
        assert results[2]['prediction'] == 1
    
    @patch.object(InferenceRunner, 'check_model_availability')
    @patch.object(InferenceRunner, 'get_model_metadata')
    @patch.object(InferenceRunner, 'run_batch_inference')
    def test_process_dataset(self, mock_batch_inference, mock_get_metadata, 
                           mock_check_availability, runner, sample_df):
        """Test dataset processing."""
        mock_check_availability.return_value = True
        mock_get_metadata.return_value = {'name': 'llama2', 'available': True}
        mock_batch_inference.return_value = [
            {'C_BioSense_ID': 'test_001', 'prediction': 1, 'rationale': 'Match', 'confidence': 0.9},
            {'C_BioSense_ID': 'test_002', 'prediction': 0, 'rationale': 'No match', 'confidence': 0.8},
            {'C_BioSense_ID': 'test_003', 'prediction': 1, 'rationale': 'Match', 'confidence': 0.95}
        ]
        
        result_df = runner.process_dataset(sample_df, 'llama2')
        
        assert len(result_df) == 3
        assert 'prediction' in result_df.columns
        assert 'rationale' in result_df.columns
        assert 'confidence' in result_df.columns
        
        mock_check_availability.assert_called_once_with('llama2')
        mock_get_metadata.assert_called_once_with('llama2')
        mock_batch_inference.assert_called_once()
    
    @patch.object(InferenceRunner, 'check_model_availability')
    def test_process_dataset_model_not_found(self, mock_check_availability, runner, sample_df):
        """Test dataset processing with model not found."""
        mock_check_availability.return_value = False
        
        with pytest.raises(ModelNotFoundError, match="Model 'llama2' is not available"):
            runner.process_dataset(sample_df, 'llama2')
    
    def test_process_dataset_missing_prompt_column(self, runner):
        """Test dataset processing with missing prompt column."""
        df = pd.DataFrame({'C_BioSense_ID': ['test_001'], 'other_column': ['test']})
        
        with pytest.raises(ValueError, match="Prompt column 'formatted_prompt' not found"):
            runner.process_dataset(df, 'llama2')
    
    def test_save_results_csv(self, runner, sample_df):
        """Test saving results to CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'results.csv')
            runner.save_results(sample_df, output_path, 'csv')
            
            assert os.path.exists(output_path)
            loaded_df = pd.read_csv(output_path)
            assert len(loaded_df) == len(sample_df)
    
    def test_save_results_json(self, runner, sample_df):
        """Test saving results to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'results.json')
            runner.save_results(sample_df, output_path, 'json')
            
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                data = json.load(f)
            assert len(data) == len(sample_df)
    
    def test_save_results_invalid_format(self, runner, sample_df):
        """Test saving results with invalid format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'results.txt')
            
            with pytest.raises(ValueError, match="Unsupported format"):
                runner.save_results(sample_df, output_path, 'txt')
    
    def test_create_inference_metadata(self, runner, sample_df):
        """Test inference metadata creation."""
        runner.model_metadata = {'name': 'llama2', 'size': '3.8GB'}
        
        metadata = runner.create_inference_metadata('llama2', sample_df, 10.5)
        
        assert metadata['model_name'] == 'llama2'
        assert metadata['model_metadata'] == {'name': 'llama2', 'size': '3.8GB'}
        assert metadata['total_rows'] == 3
        assert metadata['processing_time_seconds'] == 10.5
        assert metadata['batch_size'] == 3
        assert 'timestamp' in metadata
        assert 'columns_in_output' in metadata
        assert 'prediction_stats' in metadata
    
    def test_calculate_prediction_stats(self, runner):
        """Test prediction statistics calculation."""
        df = pd.DataFrame({
            'prediction': [1, 0, 1, -1, 1],
            'confidence': [0.9, 0.8, 0.95, 0.5, 0.85]
        })
        
        stats = runner._calculate_prediction_stats(df)
        
        assert stats['total_predictions'] == 5
        assert set(stats['unique_predictions']) == {1, 0, -1}
        assert stats['prediction_counts'] == {1: 3, 0: 1, -1: 1}
        assert abs(stats['mean_confidence'] - 0.8) < 0.01
    
    def test_calculate_prediction_stats_no_predictions(self, runner):
        """Test prediction statistics with no predictions."""
        df = pd.DataFrame({'other_column': ['test']})
        
        stats = runner._calculate_prediction_stats(df)
        
        assert stats == {}
    
    def test_save_metadata(self, runner):
        """Test metadata saving."""
        metadata = {'test': 'data', 'number': 42}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'metadata.json')
            runner.save_metadata(metadata, output_path)
            
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                loaded_metadata = json.load(f)
            assert loaded_metadata == metadata


class TestInferenceRunnerIntegration:
    """Integration tests for InferenceRunner."""
    
    @pytest.fixture
    def runner(self):
        """Create a test instance of InferenceRunner."""
        return InferenceRunner(debug_mode=True, batch_size=2)
    
    def test_full_pipeline_with_mock(self, runner):
        """Test full inference pipeline with mocked Ollama."""
        # Create sample data
        df = pd.DataFrame({
            'C_BioSense_ID': ['test_001', 'test_002'],
            'formatted_prompt': ['prompt 1', 'prompt 2']
        })
        
        with patch.object(runner, 'check_model_availability', return_value=True), \
             patch.object(runner, 'get_model_metadata', return_value={'name': 'llama2'}), \
             patch.object(runner, 'run_batch_inference') as mock_batch:
            
            mock_batch.return_value = [
                {'C_BioSense_ID': 'test_001', 'prediction': 1, 'rationale': 'Match', 'confidence': 0.9},
                {'C_BioSense_ID': 'test_002', 'prediction': 0, 'rationale': 'No match', 'confidence': 0.8}
            ]
            
            result_df = runner.process_dataset(df, 'llama2')
            
            assert len(result_df) == 2
            assert result_df.iloc[0]['prediction'] == 1
            assert result_df.iloc[1]['prediction'] == 0
    
    def test_error_handling_pipeline(self, runner):
        """Test error handling in full pipeline."""
        df = pd.DataFrame({
            'C_BioSense_ID': ['test_001'],
            'formatted_prompt': ['prompt 1']
        })
        
        with patch.object(runner, 'check_model_availability', return_value=False):
            with pytest.raises(ModelNotFoundError):
                runner.process_dataset(df, 'nonexistent_model')


if __name__ == "__main__":
    pytest.main([__file__]) 