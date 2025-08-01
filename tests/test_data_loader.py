"""
Unit tests for the DataLoader module.

Tests cover:
- Dataset loading and validation
- Schema validation with missing fields
- Data cleaning and type casting
- Context block creation
- Rating standardization
- Error handling
- Edge cases
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys
import json

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from data_loader import DataLoader, SchemaError, DataTypeError, FileError


class TestDataLoader:
    """Test suite for DataLoader class."""
    
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
            'Expert Rating': [1, 2, 1, 0, 2],
            'Rationale of Rating': ['Clear match', 'Partial match', 'Strong match', 'No match', 'Weak match']
        })
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance for testing."""
        return DataLoader(debug_mode=True)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_init(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader.debug_mode is True
        assert hasattr(data_loader, 'logger')
    
    def test_load_dataset_success(self, data_loader, temp_csv_file):
        """Test successful dataset loading."""
        df = data_loader.load_dataset(temp_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert len(df.columns) == 13
    
    def test_load_dataset_file_not_found(self, data_loader):
        """Test loading non-existent file."""
        with pytest.raises(FileError, match="File not found"):
            data_loader.load_dataset("nonexistent_file.csv")
    
    def test_load_dataset_empty_file(self, data_loader):
        """Test loading empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            
            with pytest.raises(FileError, match="File is empty"):
                data_loader.load_dataset(f.name)
        
        os.unlink(f.name)
    
    def test_validate_schema_success(self, data_loader, sample_data):
        """Test successful schema validation."""
        is_valid, missing_fields, suggestions = data_loader.validate_schema(sample_data)
        assert is_valid is True
        assert len(missing_fields) == 0
        assert len(suggestions) == 0
    
    def test_validate_schema_missing_fields(self, data_loader, sample_data):
        """Test schema validation with missing fields."""
        # Remove required field
        incomplete_data = sample_data.drop(columns=['Age'])
        is_valid, missing_fields, suggestions = data_loader.validate_schema(incomplete_data)
        
        assert is_valid is False
        assert 'Age' in missing_fields
        assert len(suggestions) > 0
    
    def test_validate_schema_column_mapping_suggestions(self, data_loader, sample_data):
        """Test column mapping suggestions for missing fields."""
        # Rename a column to something similar
        sample_data = sample_data.rename(columns={'Age': 'Patient_Age'})
        incomplete_data = sample_data.drop(columns=['Expert Rating'])
        
        is_valid, missing_fields, suggestions = data_loader.validate_schema(incomplete_data)
        
        assert is_valid is False
        assert 'Expert Rating' in missing_fields
        assert 'Expert Rating' in suggestions
    
    def test_clean_dataset_success(self, data_loader, sample_data):
        """Test successful dataset cleaning."""
        # Add some missing values
        sample_data.loc[0, 'TriageNotes'] = None
        sample_data.loc[1, 'Age'] = None
        
        cleaned_df = data_loader.clean_dataset(sample_data)
        
        assert len(cleaned_df) == 5  # No rows dropped
        assert cleaned_df.loc[0, 'TriageNotes'] == ''  # Missing value filled
        assert cleaned_df.loc[1, 'Age'] == 0  # Missing age filled with 0
    
    def test_clean_dataset_drop_missing_id(self, data_loader, sample_data):
        """Test dropping rows with missing C_BioSense_ID."""
        # Add missing ID
        sample_data.loc[2, 'C_BioSense_ID'] = None
        
        cleaned_df = data_loader.clean_dataset(sample_data)
        
        assert len(cleaned_df) == 4  # One row dropped
        assert 'P003' not in cleaned_df['C_BioSense_ID'].values
    
    def test_cast_data_types_success(self, data_loader, sample_data):
        """Test successful data type casting."""
        # Convert Age to string to test casting
        sample_data['Age'] = sample_data['Age'].astype(str)
        
        cleaned_df = data_loader.clean_dataset(sample_data)
        
        # Check that Age is integer (could be int32 or int64)
        assert pd.api.types.is_integer_dtype(cleaned_df['Age'])
        assert pd.api.types.is_integer_dtype(cleaned_df['Expert Rating'])
    
    def test_cast_data_types_mixed_ratings(self, data_loader, sample_data):
        """Test handling mixed rating types."""
        # Add string rating
        sample_data.loc[0, 'Expert Rating'] = 'High'
        
        cleaned_df = data_loader.clean_dataset(sample_data)
        
        # Should handle mixed types gracefully - string becomes 'nan' string
        # The current implementation converts non-numeric to NaN, then converts to string
        assert cleaned_df.loc[0, 'Expert Rating'] == 'nan'  # String 'High' becomes 'nan'
        assert cleaned_df.loc[1, 'Expert Rating'] == 2  # Integer preserved
    
    def test_extract_unique_ratings(self, data_loader, sample_data):
        """Test extraction of unique ratings."""
        unique_ratings = data_loader.extract_unique_ratings(sample_data)
        
        expected_ratings = [1, 2, 0]
        assert set(unique_ratings) == set(expected_ratings)
    
    def test_extract_unique_ratings_missing_column(self, data_loader, sample_data):
        """Test extraction with missing Expert Rating column."""
        incomplete_data = sample_data.drop(columns=['Expert Rating'])
        
        with pytest.raises(SchemaError, match="Expert Rating column not found"):
            data_loader.extract_unique_ratings(incomplete_data)
    
    def test_create_context_block(self, data_loader, sample_data):
        """Test context block creation."""
        df_with_context = data_loader.create_context_block(sample_data)
        
        assert 'Context_Block' in df_with_context.columns
        
        # Check first row context
        first_context = df_with_context.loc[0, 'Context_Block']
        assert 'Age: 25' in first_context
        assert 'Sex: M' in first_context
        assert 'Chief Complaint: Fever' in first_context
        assert 'Discharge Diagnosis: Viral infection' in first_context
    
    def test_create_context_block_missing_fields(self, data_loader, sample_data):
        """Test context block creation with missing fields."""
        # Remove some text fields
        incomplete_data = sample_data.drop(columns=['TriageNotes', 'CCDDCategory'])
        
        df_with_context = data_loader.create_context_block(incomplete_data)
        
        assert 'Context_Block' in df_with_context.columns
        # Should still create context with available fields
        first_context = df_with_context.loc[0, 'Context_Block']
        assert 'Chief Complaint: Fever' in first_context
        assert 'Triage Notes:' not in first_context  # Missing field
    
    def test_add_standardized_rating_column(self, data_loader, sample_data):
        """Test adding standardized rating column."""
        rating_mapping = {1: 1, 2: 0, 0: -1}  # 1=Match, 2=Does Not Match, 0=Unknown
        
        df_with_standardized = data_loader.add_standardized_rating_column(sample_data, rating_mapping)
        
        assert 'Standardized_Rating' in df_with_standardized.columns
        assert df_with_standardized.loc[0, 'Standardized_Rating'] == 1  # 1 -> 1
        assert df_with_standardized.loc[1, 'Standardized_Rating'] == 0  # 2 -> 0
        assert df_with_standardized.loc[3, 'Standardized_Rating'] == -1  # 0 -> -1
    
    def test_add_standardized_rating_column_unmapped_values(self, data_loader, sample_data):
        """Test handling unmapped rating values."""
        rating_mapping = {1: 1}  # Only map one value
        
        df_with_standardized = data_loader.add_standardized_rating_column(sample_data, rating_mapping)
        
        assert df_with_standardized.loc[0, 'Standardized_Rating'] == 1  # Mapped
        assert df_with_standardized.loc[1, 'Standardized_Rating'] == -1  # Unmapped -> -1
    
    def test_process_dataset_complete_pipeline(self, data_loader, temp_csv_file):
        """Test complete dataset processing pipeline."""
        rating_mapping = {1: 1, 2: 0, 0: -1}
        
        df, unique_ratings, metadata = data_loader.process_dataset(temp_csv_file, rating_mapping)
        
        # Check DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'Context_Block' in df.columns
        assert 'Standardized_Rating' in df.columns
        
        # Check unique ratings
        assert set(unique_ratings) == {1, 2, 0}
        
        # Check metadata
        assert 'original_shape' in metadata
        assert 'cleaned_shape' in metadata
        assert 'unique_ratings' in metadata
        assert 'rating_mapping' in metadata
    
    def test_process_dataset_without_rating_mapping(self, data_loader, temp_csv_file):
        """Test processing without rating mapping."""
        df, unique_ratings, metadata = data_loader.process_dataset(temp_csv_file)
        
        assert 'Standardized_Rating' in df.columns
        assert all(df['Standardized_Rating'] == -1)  # All set to -1 (Unknown)
    
    def test_process_dataset_schema_error(self, data_loader):
        """Test processing with schema error."""
        # Create incomplete dataset
        incomplete_data = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'ChiefComplaintOrig': ['Fever']
            # Missing required fields
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            
            with pytest.raises(SchemaError, match="Schema validation failed"):
                data_loader.process_dataset(f.name)
        
        os.unlink(f.name)
    
    def test_save_processed_dataset(self, data_loader, sample_data):
        """Test saving processed dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            data_loader.save_processed_dataset(sample_data, output_path)
            
            # Verify file was created and contains data
            assert os.path.exists(output_path)
            saved_df = pd.read_csv(output_path)
            assert len(saved_df) == 5
            assert len(saved_df.columns) == 13
        finally:
            os.unlink(output_path)
    
    def test_save_metadata(self, data_loader):
        """Test saving metadata."""
        metadata = {
            'original_shape': (5, 13),
            'cleaned_shape': (5, 15),
            'unique_ratings': [1, 2, 0],
            'rating_mapping': {1: 1, 2: 0, 0: -1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            data_loader.save_metadata(metadata, output_path)
            
            # Verify file was created and contains metadata
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                saved_metadata = json.loads(f.read())
            assert saved_metadata['original_shape'] == [5, 13]
            assert saved_metadata['unique_ratings'] == [1, 2, 0]
        finally:
            os.unlink(output_path)
    
    def test_edge_case_empty_dataset(self, data_loader):
        """Test handling empty dataset."""
        empty_data = pd.DataFrame(columns=DataLoader.REQUIRED_FIELDS)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_data.to_csv(f.name, index=False)
            
            df, unique_ratings, metadata = data_loader.process_dataset(f.name)
            
            assert len(df) == 0
            assert len(unique_ratings) == 0
        
        os.unlink(f.name)
    
    def test_edge_case_single_row(self, data_loader):
        """Test handling single row dataset."""
        single_row_data = pd.DataFrame({
            'C_BioSense_ID': ['P001'],
            'ChiefComplaintOrig': ['Fever'],
            'Discharge Diagnosis': ['Viral infection'],
            'Sex': ['M'],
            'Age': [25],
            'Admit_Reason_Combo': ['Fever'],
            'Chief_Complaint_Combo': ['Fever'],
            'Diagnosis_Combo': ['Infection'],
            'CCDD': ['Fever'],
            'CCDDCategory': ['Viral'],
            'TriageNotes': ['High fever'],
            'Expert Rating': [1],
            'Rationale of Rating': ['Clear match']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            single_row_data.to_csv(f.name, index=False)
            
            df, unique_ratings, metadata = data_loader.process_dataset(f.name)
            
            assert len(df) == 1
            assert len(unique_ratings) == 1
            assert unique_ratings[0] == 1
        
        os.unlink(f.name)
    
    def test_edge_case_all_missing_values(self, data_loader):
        """Test handling dataset with all missing values."""
        missing_data = pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002'],
            'ChiefComplaintOrig': [None, ''],
            'Discharge Diagnosis': [None, ''],
            'Sex': [None, ''],
            'Age': [None, None],
            'Admit_Reason_Combo': [None, ''],
            'Chief_Complaint_Combo': [None, ''],
            'Diagnosis_Combo': [None, ''],
            'CCDD': [None, ''],
            'CCDDCategory': [None, ''],
            'TriageNotes': [None, ''],
            'Expert Rating': [None, ''],
            'Rationale of Rating': [None, '']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            missing_data.to_csv(f.name, index=False)
            
            df, unique_ratings, metadata = data_loader.process_dataset(f.name)
            
            assert len(df) == 2  # No rows dropped (only C_BioSense_ID is required)
            assert all(df['Age'] == 0)  # Missing ages filled with 0
            assert all(df['ChiefComplaintOrig'] == '')  # Missing text filled with empty string
        
        os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__]) 