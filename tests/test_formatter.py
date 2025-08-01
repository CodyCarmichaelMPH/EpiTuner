"""
Unit tests for the Formatter module.

Tests cover:
- Training prompt creation
- Inference prompt creation
- Mixed prompt creation
- Input validation
- Prompt template customization
- Batch processing
- File operations (JSONL)
- Prompt statistics
- Edge cases
"""

import pytest
import pandas as pd
import tempfile
import os
import json
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from formatter import Formatter, FormatterError, ValidationError


class TestFormatter:
    """Test suite for Formatter class."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training dataset."""
        return pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002', 'P003'],
            'Context_Block': [
                'Chief Complaint: Fever, cough\nDischarge Diagnosis: Viral pneumonia\nTriage Notes: Persistent cough',
                'Chief Complaint: Chest pain\nDischarge Diagnosis: Angina\nTriage Notes: Severe chest pain',
                'Chief Complaint: Headache\nDischarge Diagnosis: Migraine\nTriage Notes: Throbbing headache'
            ],
            'Standardized_Rating': [1, 2, 0],
            'Rationale of Rating': [
                'Clear respiratory infection',
                'Cardiac symptoms present',
                'Neurological condition'
            ]
        })
    
    @pytest.fixture
    def sample_inference_data(self):
        """Create sample inference dataset."""
        return pd.DataFrame({
            'C_BioSense_ID': ['P004', 'P005', 'P006'],
            'Context_Block': [
                'Chief Complaint: Shortness of breath\nDischarge Diagnosis: Asthma\nTriage Notes: Wheezing',
                'Chief Complaint: Abdominal pain\nDischarge Diagnosis: Appendicitis\nTriage Notes: Severe pain',
                'Chief Complaint: Dizziness\nDischarge Diagnosis: Vertigo\nTriage Notes: Balance issues'
            ]
        })
    
    @pytest.fixture
    def formatter(self):
        """Create Formatter instance for testing."""
        return Formatter(debug_mode=True)
    
    def test_init(self, formatter):
        """Test Formatter initialization."""
        assert formatter.debug_mode is True
        assert hasattr(formatter, 'logger')
        assert formatter.training_template is not None
        assert formatter.inference_template is not None
    
    def test_get_default_templates(self, formatter):
        """Test default template generation."""
        training_template = formatter._get_default_training_template()
        inference_template = formatter._get_default_inference_template()
        
        assert '{topic}' in training_template
        assert '{context_block}' in training_template
        assert '{rating}' in training_template
        assert '{rationale}' in training_template
        
        assert '{topic}' in inference_template
        assert '{context_block}' in inference_template
        assert '{rating}' not in inference_template  # Should not be in inference template
    
    def test_validate_input_data_success(self, formatter, sample_training_data):
        """Test successful input validation."""
        is_valid, errors = formatter.validate_input_data(sample_training_data)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_input_data_missing_required_columns(self, formatter):
        """Test validation with missing required columns."""
        invalid_data = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'Some_Other_Column': ['data']
        })
        
        is_valid, errors = formatter.validate_input_data(invalid_data)
        assert is_valid is False
        assert len(errors) > 0
        assert any('Context_Block' in error for error in errors)
    
    def test_validate_input_data_missing_ratings(self, formatter):
        """Test validation with missing ratings in training data."""
        data_with_missing_ratings = pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002'],
            'Context_Block': ['Context 1', 'Context 2'],
            'Standardized_Rating': [1, None]  # Missing rating
        })
        
        is_valid, errors = formatter.validate_input_data(data_with_missing_ratings)
        assert is_valid is False
        assert any('missing values' in error.lower() for error in errors)
    
    def test_validate_input_data_empty_contexts(self, formatter):
        """Test validation with empty context blocks."""
        data_with_empty_contexts = pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002'],
            'Context_Block': ['Valid context', None]  # Empty context
        })
        
        is_valid, errors = formatter.validate_input_data(data_with_empty_contexts)
        assert is_valid is False
        assert any('empty context' in error.lower() for error in errors)
    
    def test_create_training_prompts_success(self, formatter, sample_training_data):
        """Test successful training prompt creation."""
        prompts = formatter.create_training_prompts(
            sample_training_data, 
            "Respiratory Issues", 
            include_rationale=True
        )
        
        assert len(prompts) == 3
        assert all('prompt' in prompt for prompt in prompts)
        assert all('rating' in prompt for prompt in prompts)
        assert all('rationale' in prompt for prompt in prompts)
        assert all('topic' in prompt for prompt in prompts)
        
        # Check first prompt content
        first_prompt = prompts[0]
        assert 'Respiratory Issues' in first_prompt['prompt']
        assert 'Viral pneumonia' in first_prompt['prompt']
        assert first_prompt['rating'] == 1
        assert 'Clear respiratory infection' in first_prompt['rationale']
    
    def test_create_training_prompts_without_rationale(self, formatter, sample_training_data):
        """Test training prompt creation without rationale."""
        prompts = formatter.create_training_prompts(
            sample_training_data, 
            "Respiratory Issues", 
            include_rationale=False
        )
        
        assert len(prompts) == 3
        assert all(prompt['rationale'] is None for prompt in prompts)
    
    def test_create_training_prompts_custom_template(self, formatter, sample_training_data):
        """Test training prompt creation with custom template."""
        custom_template = """CUSTOM TEMPLATE:
Topic: {topic}
Data: {context_block}
Expert Rating: {rating}
Expert Reason: {rationale}"""
        
        prompts = formatter.create_training_prompts(
            sample_training_data, 
            "Respiratory Issues", 
            include_rationale=True,
            custom_template=custom_template
        )
        
        assert len(prompts) == 3
        assert all('CUSTOM TEMPLATE:' in prompt['prompt'] for prompt in prompts)
        assert all('Topic: Respiratory Issues' in prompt['prompt'] for prompt in prompts)
    
    def test_create_training_prompts_missing_ratings(self, formatter):
        """Test training prompt creation with missing ratings."""
        data_with_missing_ratings = pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002'],
            'Context_Block': ['Context 1', 'Context 2'],
            'Standardized_Rating': [1, None]  # Missing rating
        })
        
        prompts = formatter.create_training_prompts(
            data_with_missing_ratings, 
            "Test Topic", 
            include_rationale=True
        )
        
        # Should skip the row with missing rating
        assert len(prompts) == 1
        assert prompts[0]['rating'] == 1
    
    def test_create_inference_prompts_success(self, formatter, sample_inference_data):
        """Test successful inference prompt creation."""
        prompts = formatter.create_inference_prompts(
            sample_inference_data, 
            "Respiratory Issues"
        )
        
        assert len(prompts) == 3
        assert all('prompt' in prompt for prompt in prompts)
        assert all('topic' in prompt for prompt in prompts)
        assert all('rating' not in prompt for prompt in prompts)  # No ratings in inference
        
        # Check first prompt content
        first_prompt = prompts[0]
        assert 'Respiratory Issues' in first_prompt['prompt']
        assert 'Asthma' in first_prompt['prompt']
        assert '<predicted integer>' in first_prompt['prompt']
    
    def test_create_inference_prompts_custom_template(self, formatter, sample_inference_data):
        """Test inference prompt creation with custom template."""
        custom_template = """INFERENCE TEMPLATE:
Topic: {topic}
Patient Data: {context_block}
Please predict: <rating> and <rationale>"""
        
        prompts = formatter.create_inference_prompts(
            sample_inference_data, 
            "Respiratory Issues",
            custom_template=custom_template
        )
        
        assert len(prompts) == 3
        assert all('INFERENCE TEMPLATE:' in prompt['prompt'] for prompt in prompts)
        assert all('Please predict:' in prompt['prompt'] for prompt in prompts)
    
    def test_create_mixed_prompts(self, formatter):
        """Test mixed prompt creation with both training and inference data."""
        # Create mixed dataset
        mixed_data = pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002', 'P003', 'P004'],
            'Context_Block': ['Context 1', 'Context 2', 'Context 3', 'Context 4'],
            'Standardized_Rating': [1, 2, None, None],  # Some with ratings, some without
            'Rationale of Rating': ['Reason 1', 'Reason 2', None, None]
        })
        
        mixed_prompts = formatter.create_mixed_prompts(
            mixed_data, 
            "Test Topic", 
            include_rationale=True
        )
        
        assert 'training' in mixed_prompts
        assert 'inference' in mixed_prompts
        assert len(mixed_prompts['training']) == 2  # 2 rows with ratings
        assert len(mixed_prompts['inference']) == 2  # 2 rows without ratings
    
    def test_save_prompts_to_jsonl(self, formatter, sample_training_data):
        """Test saving prompts to JSONL file."""
        prompts = formatter.create_training_prompts(
            sample_training_data, 
            "Test Topic"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_file = f.name
        
        try:
            formatter.save_prompts_to_jsonl(prompts, temp_file)
            
            # Verify file was created and contains correct data
            assert os.path.exists(temp_file)
            
            with open(temp_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3
                
                # Parse each line
                for line in lines:
                    prompt_data = json.loads(line)
                    assert 'prompt' in prompt_data
                    assert 'topic' in prompt_data
                    assert prompt_data['topic'] == 'Test Topic'
        
        finally:
            os.unlink(temp_file)
    
    def test_load_prompts_from_jsonl(self, formatter):
        """Test loading prompts from JSONL file."""
        test_prompts = [
            {'id': 'P001', 'prompt': 'Test prompt 1', 'topic': 'Test Topic'},
            {'id': 'P002', 'prompt': 'Test prompt 2', 'topic': 'Test Topic'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_file = f.name
            for prompt in test_prompts:
                f.write(json.dumps(prompt) + '\n')
        
        try:
            loaded_prompts = formatter.load_prompts_from_jsonl(temp_file)
            
            assert len(loaded_prompts) == 2
            assert loaded_prompts[0]['id'] == 'P001'
            assert loaded_prompts[1]['id'] == 'P002'
            assert all(prompt['topic'] == 'Test Topic' for prompt in loaded_prompts)
        
        finally:
            os.unlink(temp_file)
    
    def test_load_prompts_from_jsonl_invalid_json(self, formatter):
        """Test loading prompts from JSONL file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_file = f.name
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')  # Invalid JSON
            f.write('{"another": "valid"}\n')
        
        try:
            loaded_prompts = formatter.load_prompts_from_jsonl(temp_file)
            
            # Should load valid JSON lines and skip invalid ones
            assert len(loaded_prompts) == 2
        
        finally:
            os.unlink(temp_file)
    
    def test_get_prompt_statistics(self, formatter, sample_training_data):
        """Test prompt statistics generation."""
        prompts = formatter.create_training_prompts(
            sample_training_data, 
            "Test Topic"
        )
        
        stats = formatter.get_prompt_statistics(prompts)
        
        assert stats['total_prompts'] == 3
        assert stats['avg_prompt_length'] > 0
        assert 'Test Topic' in stats['topics']
        assert len(stats['rating_distribution']) > 0
        assert stats['min_prompt_length'] > 0
        assert stats['max_prompt_length'] > 0
    
    def test_get_prompt_statistics_empty(self, formatter):
        """Test prompt statistics with empty prompt list."""
        stats = formatter.get_prompt_statistics([])
        
        assert stats['total_prompts'] == 0
        assert stats['avg_prompt_length'] == 0
        assert len(stats['topics']) == 0
        assert len(stats['rating_distribution']) == 0
    
    def test_validate_prompt_format_success(self, formatter):
        """Test successful prompt format validation."""
        valid_prompt = {
            'id': 'P001',
            'prompt': 'Valid prompt text',
            'topic': 'Test Topic'
        }
        
        is_valid, errors = formatter.validate_prompt_format(valid_prompt)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_prompt_format_missing_fields(self, formatter):
        """Test prompt format validation with missing fields."""
        invalid_prompt = {
            'id': 'P001'
            # Missing 'prompt' and 'topic'
        }
        
        is_valid, errors = formatter.validate_prompt_format(invalid_prompt)
        assert is_valid is False
        assert len(errors) > 0
        assert any('prompt' in error for error in errors)
        assert any('topic' in error for error in errors)
    
    def test_validate_prompt_format_empty_prompt(self, formatter):
        """Test prompt format validation with empty prompt."""
        invalid_prompt = {
            'id': 'P001',
            'prompt': '',  # Empty prompt
            'topic': 'Test Topic'
        }
        
        is_valid, errors = formatter.validate_prompt_format(invalid_prompt)
        assert is_valid is False
        assert any('empty' in error.lower() for error in errors)
    
    def test_validate_prompt_format_invalid_rating(self, formatter):
        """Test prompt format validation with invalid rating."""
        invalid_prompt = {
            'id': 'P001',
            'prompt': 'Valid prompt',
            'topic': 'Test Topic',
            'rating': 'invalid'  # Invalid rating type
        }
        
        is_valid, errors = formatter.validate_prompt_format(invalid_prompt)
        assert is_valid is False
        assert any('integer' in error.lower() for error in errors)
    
    def test_batch_process(self, formatter):
        """Test batch processing functionality."""
        # Create larger dataset
        large_data = pd.DataFrame({
            'C_BioSense_ID': [f'P{i:03d}' for i in range(1, 251)],  # 250 rows
            'Context_Block': [f'Context for patient {i}' for i in range(1, 251)],
            'Standardized_Rating': [i % 3 for i in range(1, 251)],  # 0, 1, 2 cycling
            'Rationale of Rating': [f'Rationale {i}' for i in range(1, 251)]
        })
        
        prompts = formatter.batch_process(
            large_data, 
            "Test Topic", 
            batch_size=50,
            include_rationale=True
        )
        
        assert len(prompts) == 250
        assert all('prompt' in prompt for prompt in prompts)
        assert all(prompt['topic'] == 'Test Topic' for prompt in prompts)
    
    def test_set_custom_templates(self, formatter):
        """Test setting custom templates."""
        original_training = formatter.training_template
        original_inference = formatter.inference_template
        
        new_training = "NEW TRAINING TEMPLATE: {topic} {context_block} {rating}"
        new_inference = "NEW INFERENCE TEMPLATE: {topic} {context_block}"
        
        formatter.set_custom_templates(
            training_template=new_training,
            inference_template=new_inference
        )
        
        assert formatter.training_template == new_training
        assert formatter.inference_template == new_inference
        assert formatter.training_template != original_training
        assert formatter.inference_template != original_inference
    
    def test_set_custom_templates_partial(self, formatter):
        """Test setting only one custom template."""
        original_training = formatter.training_template
        original_inference = formatter.inference_template
        
        new_training = "NEW TRAINING TEMPLATE: {topic} {context_block} {rating}"
        
        formatter.set_custom_templates(training_template=new_training)
        
        assert formatter.training_template == new_training
        assert formatter.inference_template == original_inference  # Unchanged
    
    def test_error_handling_missing_context(self, formatter):
        """Test error handling with missing context data."""
        data_with_missing_context = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'Context_Block': [None]  # Missing context
        })
        
        prompts = formatter.create_inference_prompts(
            data_with_missing_context, 
            "Test Topic"
        )
        
        # Should handle missing context gracefully
        assert len(prompts) == 1
        assert 'Data unavailable' in prompts[0]['prompt']
    
    def test_error_handling_file_operations(self, formatter):
        """Test error handling in file operations."""
        # Test saving to non-existent directory
        prompts = [{'id': 'P001', 'prompt': 'Test', 'topic': 'Test'}]
        
        with pytest.raises(FormatterError):
            formatter.save_prompts_to_jsonl(prompts, '/non/existent/path/file.jsonl')
        
        # Test loading non-existent file
        with pytest.raises(FormatterError):
            formatter.load_prompts_from_jsonl('/non/existent/file.jsonl')


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self, formatter):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValidationError):
            formatter.create_training_prompts(empty_df, "Test Topic")
    
    def test_single_row_dataframe(self, formatter):
        """Test handling of single row DataFrame."""
        single_row_df = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'Context_Block': ['Single context'],
            'Standardized_Rating': [1],
            'Rationale of Rating': ['Single rationale']
        })
        
        prompts = formatter.create_training_prompts(single_row_df, "Test Topic")
        assert len(prompts) == 1
    
    def test_very_long_context(self, formatter):
        """Test handling of very long context blocks."""
        long_context = "x" * 10000  # Very long context
        long_context_df = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'Context_Block': [long_context],
            'Standardized_Rating': [1],
            'Rationale of Rating': ['Test']
        })
        
        prompts = formatter.create_training_prompts(long_context_df, "Test Topic")
        assert len(prompts) == 1
        assert len(prompts[0]['prompt']) > 10000
    
    def test_special_characters_in_data(self, formatter):
        """Test handling of special characters in data."""
        special_data = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'Context_Block': ['Context with special chars: éñüß@#$%^&*()'],
            'Standardized_Rating': [1],
            'Rationale of Rating': ['Rationale with special chars: éñüß@#$%^&*()']
        })
        
        prompts = formatter.create_training_prompts(special_data, "Test Topic")
        assert len(prompts) == 1
        assert 'éñüß@#$%^&*()' in prompts[0]['prompt']
    
    def test_numeric_string_ratings(self, formatter):
        """Test handling of numeric ratings stored as strings."""
        string_ratings_df = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'Context_Block': ['Test context'],
            'Standardized_Rating': ['1'],  # String instead of int
            'Rationale of Rating': ['Test rationale']
        })
        
        prompts = formatter.create_training_prompts(string_ratings_df, "Test Topic")
        assert len(prompts) == 1
        assert prompts[0]['rating'] == 1  # Should be converted to int


if __name__ == "__main__":
    pytest.main([__file__]) 