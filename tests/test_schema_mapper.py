"""
Test suite for Schema Mapper Module
"""

import unittest
import pandas as pd
import tempfile
import json
import os
from pathlib import Path

# Add scripts directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from schema_mapper import SchemaMapper, MappingError, DataTypeError


class TestSchemaMapper(unittest.TestCase):
    """Test cases for SchemaMapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapper = SchemaMapper(debug_mode=True)
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'Expert Rating': [1, 2, 0, 'Match', 'No Match'],
            'Rationale of Rating': ['Clear match', 'Partial match', 'No match', 'Strong match', 'No relation']
        })
        
        # Sample mapping
        self.sample_mapping = {
            1: 1,  # Match
            2: 2,  # Partial Match
            0: 0,  # Does Not Match
            'Match': 1,  # Match
            'No Match': 0  # Does Not Match
        }
    
    def test_extract_unique_ratings(self):
        """Test extraction of unique ratings."""
        unique_ratings = self.mapper.extract_unique_ratings(self.sample_data)
        expected = [1, 2, 0, 'Match', 'No Match']
        
        self.assertEqual(len(unique_ratings), len(expected))
        for rating in expected:
            self.assertIn(rating, unique_ratings)
    
    def test_extract_unique_ratings_missing_column(self):
        """Test extraction with missing Expert Rating column."""
        df_no_rating = self.sample_data.drop(columns=['Expert Rating'])
        
        with self.assertRaises(MappingError):
            self.mapper.extract_unique_ratings(df_no_rating)
    
    def test_validate_mapping_valid(self):
        """Test mapping validation with valid mapping."""
        is_valid, unmapped = self.mapper.validate_mapping(self.sample_data, self.sample_mapping)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(unmapped), 0)
    
    def test_validate_mapping_invalid(self):
        """Test mapping validation with invalid mapping."""
        incomplete_mapping = {1: 1, 2: 2}  # Missing some mappings
        
        is_valid, unmapped = self.mapper.validate_mapping(self.sample_data, incomplete_mapping)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(unmapped), 0)
    
    def test_apply_mapping(self):
        """Test applying mapping to create Standardized_Rating column."""
        df_mapped = self.mapper.apply_mapping(self.sample_data, self.sample_mapping)
        
        # Check that Standardized_Rating column was added
        self.assertIn('Standardized_Rating', df_mapped.columns)
        
        # Check mapping results
        expected_ratings = [1, 2, 0, 1, 0]  # Based on sample_mapping
        actual_ratings = df_mapped['Standardized_Rating'].tolist()
        
        self.assertEqual(actual_ratings, expected_ratings)
    
    def test_apply_mapping_invalid(self):
        """Test applying mapping with invalid mapping."""
        incomplete_mapping = {1: 1, 2: 2}  # Missing some mappings
        
        with self.assertRaises(MappingError):
            self.mapper.apply_mapping(self.sample_data, incomplete_mapping)
    
    def test_create_mapping_metadata(self):
        """Test creation of mapping metadata."""
        unique_ratings = [1, 2, 0, 'Match', 'No Match']
        metadata = self.mapper.create_mapping_metadata(self.sample_mapping, unique_ratings)
        
        # Check required keys
        required_keys = ['original_values', 'mapped_values', 'schema', 'standard_ratings', 'mapping_summary']
        for key in required_keys:
            self.assertIn(key, metadata)
        
        # Check values
        self.assertEqual(metadata['original_values'], unique_ratings)
        self.assertEqual(metadata['schema'], self.sample_mapping)
        self.assertEqual(metadata['mapping_summary']['total_mappings'], len(self.sample_mapping))
    
    def test_save_and_load_mapping_metadata(self):
        """Test saving and loading mapping metadata."""
        unique_ratings = [1, 2, 0, 'Match', 'No Match']
        metadata = self.mapper.create_mapping_metadata(self.sample_mapping, unique_ratings)
        
        # Save metadata
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.mapper.save_mapping_metadata(metadata, temp_path)
            
            # Load metadata
            loaded_metadata = self.mapper.load_mapping_metadata(temp_path)
            
            # Compare
            self.assertEqual(loaded_metadata['original_values'], metadata['original_values'])
            # Schema keys are converted to strings in JSON, so compare the values
            self.assertEqual(loaded_metadata['schema'], {str(k): v for k, v in metadata['schema'].items()})
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_suggest_mapping(self):
        """Test automatic mapping suggestion."""
        unique_ratings = [1, 2, 0, 'Match', 'No Match', 'Partial', 'Unknown']
        suggested_mapping = self.mapper.suggest_mapping(unique_ratings)
        
        # Check that all ratings have mappings
        self.assertEqual(len(suggested_mapping), len(unique_ratings))
        
        # Check specific mappings
        self.assertEqual(suggested_mapping[1], 1)  # Should map to Match
        self.assertEqual(suggested_mapping['Match'], 1)  # Should map to Match
        self.assertEqual(suggested_mapping['No Match'], 0)  # Should map to Does Not Match
        self.assertEqual(suggested_mapping['Partial'], 2)  # Should map to Partial Match
    
    def test_get_mapping_statistics(self):
        """Test getting mapping statistics."""
        stats = self.mapper.get_mapping_statistics(self.sample_data)
        
        # Check required keys
        required_keys = ['total_rows', 'unique_ratings', 'rating_distribution', 'rating_percentages']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check values
        self.assertEqual(stats['total_rows'], len(self.sample_data))
        self.assertEqual(stats['unique_ratings'], 5)  # 5 unique ratings in sample data
    
    def test_process_mapping_pipeline(self):
        """Test complete mapping processing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df_mapped, metadata = self.mapper.process_mapping(
                self.sample_data, 
                self.sample_mapping, 
                output_dir=temp_dir
            )
            
            # Check output DataFrame
            self.assertIn('Standardized_Rating', df_mapped.columns)
            self.assertEqual(len(df_mapped), len(self.sample_data))
            
            # Check metadata
            self.assertIn('mapping_summary', metadata)
            
            # Check that metadata file was created
            metadata_file = Path(temp_dir) / "rating_mapping_metadata.json"
            self.assertTrue(metadata_file.exists())
    
    def test_standard_ratings_constant(self):
        """Test that standard ratings are properly defined."""
        expected_ratings = {
            'Match': 1,
            'Does Not Match': 0,
            'Unknown': -1,
            'Partial Match': 2
        }
        
        self.assertEqual(self.mapper.STANDARD_RATINGS, expected_ratings)


class TestSchemaMapperIntegration(unittest.TestCase):
    """Integration tests for SchemaMapper with real data."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.mapper = SchemaMapper(debug_mode=True)
        
        # Create more realistic test data
        self.test_data = pd.DataFrame({
            'C_BioSense_ID': [f'P{i:03d}' for i in range(1, 11)],
            'Expert Rating': [1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
            'Rationale of Rating': [f'Rationale {i}' for i in range(1, 11)]
        })
    
    def test_large_dataset_mapping(self):
        """Test mapping with larger dataset."""
        # Create larger dataset
        large_data = pd.concat([self.test_data] * 10, ignore_index=True)
        
        # Create mapping
        mapping = {1: 1, 2: 2, 0: 0}
        
        # Process mapping
        df_mapped, metadata = self.mapper.process_mapping(large_data, mapping)
        
        # Verify results
        self.assertEqual(len(df_mapped), len(large_data))
        self.assertIn('Standardized_Rating', df_mapped.columns)
        
        # Check that all ratings were mapped correctly
        for original, mapped in mapping.items():
            mask = large_data['Expert Rating'] == original
            self.assertTrue(all(df_mapped.loc[mask, 'Standardized_Rating'] == mapped))
    
    def test_mixed_data_types(self):
        """Test mapping with mixed data types in ratings."""
        mixed_data = pd.DataFrame({
            'C_BioSense_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'Expert Rating': [1, '2', 0.0, 'Match', True],
            'Rationale of Rating': ['R1', 'R2', 'R3', 'R4', 'R5']
        })
        
        # Create mapping that handles mixed types
        mapping = {1: 1, '2': 2, 0.0: 0, 'Match': 1, True: 1}
        
        # Process mapping
        df_mapped, metadata = self.mapper.process_mapping(mixed_data, mapping)
        
        # Verify results
        self.assertEqual(len(df_mapped), len(mixed_data))
        self.assertIn('Standardized_Rating', df_mapped.columns)


if __name__ == '__main__':
    unittest.main() 