"""
Test suite for the Contextualizer module.

Tests the contextualizer functionality including few-shot sampling,
meta-prompt construction, and inference execution.
"""

import unittest
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.append('scripts')

from contextualizer import Contextualizer, ContextualizerError, PromptConstructionError, ModelResponseError


class TestContextualizer(unittest.TestCase):
    """Test cases for the Contextualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.contextualizer = Contextualizer(debug_mode=True, max_rows_context=5)
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'C_BioSense_ID': ['patient_001', 'patient_002', 'patient_003', 'patient_004', 'patient_005'],
            'ChiefComplaintOrig': ['Chest pain', 'Headache', 'Fever', 'Shortness of breath', 'Dizziness'],
            'Discharge Diagnosis': ['Angina', 'Migraine', 'Viral infection', 'Pneumonia', 'Vertigo'],
            'TriageNotes': ['Severe chest pain', 'Throbbing headache', 'High fever', 'Difficulty breathing', 'Spinning sensation'],
            'Expert Rating': [1, 0, 1, 2, 0],
            'Rationale of Rating': [
                'Clear cardiac symptoms',
                'Neurological, not respiratory',
                'Respiratory infection symptoms',
                'Severe respiratory condition',
                'Neurological, not respiratory'
            ]
        })
        
        self.rating_mapping = {0: 0, 1: 1, 2: 2}
        self.topics = "respiratory infections, cardiac conditions, and neurological disorders"
    
    def test_initialization(self):
        """Test contextualizer initialization."""
        contextualizer = Contextualizer(
            debug_mode=True,
            max_rows_context=10,
            timeout=30,
            max_retries=3
        )
        
        self.assertEqual(contextualizer.max_rows_context, 10)
        self.assertEqual(contextualizer.timeout, 30)
        self.assertEqual(contextualizer.max_retries, 3)
        self.assertTrue(contextualizer.debug_mode)
    
    def test_sample_few_shot_examples(self):
        """Test few-shot example sampling."""
        # Test with valid data
        few_shot_df = self.contextualizer.sample_few_shot_examples(
            self.sample_data, self.rating_mapping, self.topics
        )
        
        self.assertIsInstance(few_shot_df, pd.DataFrame)
        self.assertLessEqual(len(few_shot_df), self.contextualizer.max_rows_context)
        self.assertIn('Standardized_Rating', few_shot_df.columns)
        
        # Test with empty data
        empty_df = pd.DataFrame(columns=self.sample_data.columns)
        with self.assertRaises(ContextualizerError):
            self.contextualizer.sample_few_shot_examples(
                empty_df, self.rating_mapping, self.topics
            )
    
    def test_build_schema_description(self):
        """Test schema description building."""
        description = self.contextualizer.build_schema_description(self.rating_mapping)
        
        self.assertIsInstance(description, str)
        self.assertIn("Use this rating schema:", description)
        self.assertIn("Rating 0:", description)
        self.assertIn("Rating 1:", description)
        self.assertIn("Rating 2:", description)
    
    def test_format_example_row(self):
        """Test example row formatting."""
        row = self.sample_data.iloc[0]
        formatted = self.contextualizer.format_example_row(row, 1)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("1. Record:", formatted)
        self.assertIn("Chief Complaint:", formatted)
        self.assertIn("Diagnosis:", formatted)
        self.assertIn("Rating:", formatted)
        self.assertIn("Rationale:", formatted)
    
    def test_construct_meta_prompt(self):
        """Test meta-prompt construction."""
        few_shot_df = self.contextualizer.sample_few_shot_examples(
            self.sample_data, self.rating_mapping, self.topics
        )
        query_row = self.sample_data.iloc[0]
        
        prompt = self.contextualizer.construct_meta_prompt(
            few_shot_df, query_row, self.topics, self.rating_mapping
        )
        
        self.assertIsInstance(prompt, str)
        self.assertIn(self.topics, prompt)
        self.assertIn("Examples:", prompt)
        self.assertIn("New record to evaluate:", prompt)
        self.assertIn("Respond with:", prompt)
    
    @patch('subprocess.run')
    def test_check_model_availability(self, mock_run):
        """Test model availability checking."""
        # Test available model
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "phi3:mini:latest"
        
        result = self.contextualizer.check_model_availability("phi3:mini")
        self.assertTrue(result)
        
        # Test unavailable model
        mock_run.return_value.stdout = "mistral:latest"
        
        result = self.contextualizer.check_model_availability("phi3:mini")
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_run_contextual_inference(self, mock_run):
        """Test contextual inference execution."""
        # Mock successful inference
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Rating: 1\nRationale: Clear respiratory symptoms"
        
        result = self.contextualizer.run_contextual_inference(
            "Test prompt", "phi3:mini"
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('rationale', result)
        self.assertEqual(result['prediction'], 1)
    
    def test_parse_response(self):
        """Test response parsing."""
        # Test valid response
        response = "Rating: 2\nRationale: Severe respiratory condition"
        result = self.contextualizer._parse_response(response)
        
        self.assertEqual(result['prediction'], 2)
        self.assertIn('Severe respiratory condition', result['rationale'])
        
        # Test invalid response
        invalid_response = "No rating here"
        with self.assertRaises(ModelResponseError):
            self.contextualizer._parse_response(invalid_response)
    
    @patch.object(Contextualizer, 'run_contextual_inference')
    def test_evaluate_single_row(self, mock_inference):
        """Test single row evaluation."""
        # Mock inference result
        mock_inference.return_value = {
            'prediction': 1,
            'confidence': None,
            'rationale': 'Test rationale'
        }
        
        result = self.contextualizer.evaluate_single_row(
            self.sample_data, 0, self.topics, self.rating_mapping, "phi3:mini"
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('C_BioSense_ID', result)
        self.assertIn('prediction', result)
        self.assertIn('rationale', result)
        self.assertEqual(result['prediction'], 1)
    
    def test_create_evaluation_metadata(self):
        """Test metadata creation."""
        results_df = pd.DataFrame({
            'C_BioSense_ID': ['patient_001', 'patient_002'],
            'prediction': [1, 0],
            'rationale': ['Test 1', 'Test 2']
        })
        
        metadata = self.contextualizer.create_evaluation_metadata(
            "phi3:mini", self.sample_data, results_df, self.rating_mapping,
            self.topics, 10.5
        )
        
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['evaluation_type'], 'contextual')
        self.assertEqual(metadata['model_name'], 'phi3:mini')
        self.assertEqual(metadata['topics'], self.topics)
        self.assertEqual(metadata['processing_time_seconds'], 10.5)
        self.assertEqual(metadata['success_rate'], 1.0)
    
    def test_error_handling_missing_rationale(self):
        """Test handling of missing rationales."""
        # Create data with missing rationales
        data_without_rationale = self.sample_data.copy()
        data_without_rationale.loc[0, 'Rationale of Rating'] = ''
        data_without_rationale.loc[1, 'Rationale of Rating'] = None
        
        # Should still work, just skip rows without rationales 