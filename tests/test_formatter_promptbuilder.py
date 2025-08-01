"""
Tests for the Formatter PromptBuilder module.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.append('scripts')

from formatter_promptbuilder import PromptBuilder, FormattingError, ValidationError


class TestPromptBuilder:
    """Test cases for the PromptBuilder class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
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
            'TriageNotes': ['High fever with chills', 'Severe chest pain', 'Throbbing headache'],
            'Expert Rating': [1, 2, 1],
            'Standardized_Rating': [1, 0, 1]
        }
    
    @pytest.fixture
    def formatter(self):
        """Create a PromptBuilder instance for testing."""
        return PromptBuilder(debug_mode=False)
    
    @pytest.fixture
    def rating_mapping(self):
        """Create sample rating mapping."""
        return {
            'Match': 1,
            'Does Not Match': 0,
            'Unknown': -1,
            'Partial Match': 2
        }
    
    def test_initialization(self, formatter):
        """Test PromptBuilder initialization."""
        assert formatter is not None
        assert formatter.prompt_template == formatter.DEFAULT_PROMPT_TEMPLATE
        assert len(formatter.REQUIRED_FIELDS) > 0
    
    def test_set_prompt_template(self, formatter):
        """Test setting custom prompt template."""
        custom_template = "Custom template with {age} and {sex}"
        formatter.set_prompt_template(custom_template)
        assert formatter.prompt_template == custom_template
    
    def test_validate_row_data_complete(self, formatter, sample_data):
        """Test validation with complete row data."""
        df = pd.DataFrame(sample_data)
        row = df.iloc[0]
        
        is_valid, missing_fields = formatter.validate_row_data(row)
        assert is_valid is True
        assert len(missing_fields) == 0
    
    def test_validate_row_data_missing(self, formatter, sample_data):
        """Test validation with missing fields."""
        df = pd.DataFrame(sample_data)
        row = df.iloc[0].copy()
        row['Age'] = None  # Remove a required field
        
        is_valid, missing_fields = formatter.validate_row_data(row)
        assert is_valid is False
        assert 'Age' in missing_fields
    
    def test_preprocess_text(self, formatter):
        """Test text preprocessing."""
        # Test normal text
        assert formatter.preprocess_text("  Hello World  ") == "Hello World"
        
        # Test None/NaN values
        assert formatter.preprocess_text(None) == "N/A"
        assert formatter.preprocess_text(pd.NA) == "N/A"
        
        # Test empty string
        assert formatter.preprocess_text("") == "N/A"
        assert formatter.preprocess_text("   ") == "N/A"
    
    def test_create_schema_description(self, formatter, rating_mapping):
        """Test schema description creation."""
        description = formatter.create_schema_description(rating_mapping)
        
        assert "Match (1)" in description
        assert "Does Not Match (0)" in description
        assert "Unknown (-1)" in description
        assert "Partial Match (2)" in description
    
    def test_build_prompt(self, formatter, sample_data, rating_mapping):
        """Test prompt building."""
        df = pd.DataFrame(sample_data)
        row = df.iloc[0]
        target_topics = "respiratory infections"
        
        prompt = formatter.build_prompt(row, target_topics, rating_mapping)
        
        # Check that prompt contains expected elements
        assert "Context:" in prompt
        assert "Task:" in prompt
        assert "Age 25" in prompt
        assert "Sex M" in prompt
        assert "Fever" in prompt
        assert "respiratory infections" in prompt
        assert "Match (1)" in prompt
        assert "Numeric rating" in prompt
    
    def test_build_prompt_with_missing_data(self, formatter, sample_data, rating_mapping):
        """Test prompt building with missing data."""
        df = pd.DataFrame(sample_data)
        row = df.iloc[0].copy()
        row['Age'] = None
        row['TriageNotes'] = ""
        
        target_topics = "respiratory infections"
        prompt = formatter.build_prompt(row, target_topics, rating_mapping)
        
        # Should still work and use "N/A" for missing values
        assert "Age N/A" in prompt
        assert "N/A" in prompt
    
    def test_format_dataset(self, formatter, sample_data, rating_mapping):
        """Test formatting entire dataset."""
        df = pd.DataFrame(sample_data)
        target_topics = "respiratory infections"
        
        formatted_df = formatter.format_dataset(df, target_topics, rating_mapping)
        
        assert len(formatted_df) == len(df)
        assert 'Prompt' in formatted_df.columns
        assert 'Row_ID' in formatted_df.columns
        
        # Check that all prompts were created
        assert all(formatted_df['Prompt'].notna())
        assert all(formatted_df['Row_ID'].notna())
    
    def test_save_formatted_dataset(self, formatter, sample_data, rating_mapping):
        """Test saving formatted dataset."""
        df = pd.DataFrame(sample_data)
        formatted_df = formatter.format_dataset(df, "test topics", rating_mapping)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_formatted.csv")
            formatter.save_formatted_dataset(formatted_df, output_path)
            
            assert os.path.exists(output_path)
            
            # Verify file can be read back
            loaded_df = pd.read_csv(output_path)
            assert len(loaded_df) == len(formatted_df)
            assert 'Prompt' in loaded_df.columns
    
    def test_save_prompts_jsonl(self, formatter, sample_data, rating_mapping):
        """Test saving prompts in JSONL format."""
        df = pd.DataFrame(sample_data)
        formatted_df = formatter.format_dataset(df, "test topics", rating_mapping)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_prompts.jsonl")
            formatter.save_prompts_jsonl(formatted_df, output_path)
            
            assert os.path.exists(output_path)
            
            # Verify JSONL format
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == len(formatted_df)
                
                # Check first line is valid JSON
                import json
                first_line = json.loads(lines[0])
                assert 'C_BioSense_ID' in first_line
                assert 'prompt' in first_line
    
    def test_create_prompt_metadata(self, formatter, sample_data, rating_mapping):
        """Test metadata creation."""
        df = pd.DataFrame(sample_data)
        target_topics = "respiratory infections"
        
        metadata = formatter.create_prompt_metadata(df, target_topics, rating_mapping)
        
        assert 'formatting_info' in metadata
        assert 'data_quality' in metadata
        assert metadata['formatting_info']['total_rows'] == len(df)
        assert metadata['formatting_info']['target_topics'] == target_topics
        assert metadata['formatting_info']['rating_mapping'] == rating_mapping
    
    def test_save_metadata(self, formatter, sample_data, rating_mapping):
        """Test saving metadata."""
        df = pd.DataFrame(sample_data)
        metadata = formatter.create_prompt_metadata(df, "test topics", rating_mapping)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_metadata.json")
            formatter.save_metadata(metadata, output_path)
            
            assert os.path.exists(output_path)
            
            # Verify JSON can be loaded back
            import json
            with open(output_path, 'r') as f:
                loaded_metadata = json.load(f)
                assert loaded_metadata == metadata
    
    def test_process_dataset_complete(self, formatter, sample_data, rating_mapping):
        """Test complete processing pipeline."""
        df = pd.DataFrame(sample_data)
        target_topics = "respiratory infections"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            formatted_df, metadata = formatter.process_dataset(
                df, target_topics, rating_mapping, output_dir=temp_dir
            )
            
            # Check outputs
            assert len(formatted_df) == len(df)
            assert 'Prompt' in formatted_df.columns
            
            # Check files were created
            assert os.path.exists(os.path.join(temp_dir, "formatted_dataset.csv"))
            assert os.path.exists(os.path.join(temp_dir, "prompts.jsonl"))
            assert os.path.exists(os.path.join(temp_dir, "formatting_metadata.json"))
            
            # Check metadata
            assert metadata['formatting_info']['total_rows'] == len(df)
    
    def test_error_handling_missing_required_field(self, formatter, sample_data, rating_mapping):
        """Test error handling with missing required field."""
        df = pd.DataFrame(sample_data)
        # Remove a required column
        df = df.drop('C_BioSense_ID', axis=1)
        
        target_topics = "respiratory infections"
        
        # Should still work but log warnings
        formatted_df = formatter.format_dataset(df, target_topics, rating_mapping)
        assert len(formatted_df) == len(df)  # Should still process all rows
    
    def test_custom_template_integration(self, formatter, sample_data, rating_mapping):
        """Test integration with custom template."""
        custom_template = """
Patient: {age} year old {sex}
Complaint: {chief_complaint}
Diagnosis: {discharge_diagnosis}
Topics: {target_topics}
Schema: {schema_description}
"""
        
        formatter.set_prompt_template(custom_template)
        df = pd.DataFrame(sample_data)
        row = df.iloc[0]
        target_topics = "respiratory infections"
        
        prompt = formatter.build_prompt(row, target_topics, rating_mapping)
        
        assert "Patient:" in prompt
        assert "25 year old M" in prompt
        assert "Complaint:" in prompt
        assert "Fever" in prompt


if __name__ == "__main__":
    pytest.main([__file__]) 