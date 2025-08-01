"""
Inference Runner Module for Ollama Fine-Tuning and Evaluation Suite

This module handles running inference on formatted dataset prompts using an Ollama model,
producing structured predictions and rationales for each case.
"""

import pandas as pd
import logging
import json
import re
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import requests
import subprocess
import sys


class ModelNotFoundError(Exception):
    """Raised when the specified Ollama model is not found."""
    pass


class ResponseParsingError(Exception):
    """Raised when model response cannot be parsed into required fields."""
    pass


class InferenceError(Exception):
    """Raised when inference operations fail."""
    pass


class InferenceRunner:
    """
    Handles running inference on formatted prompts using Ollama models.
    
    Supports both fine-tuned models and base models with contextual prompts.
    Produces structured predictions with confidence scores and rationales.
    """
    
    def __init__(self, debug_mode: bool = False, batch_size: int = 5, 
                 timeout: int = 30, max_retries: int = 3):
        """
        Initialize the InferenceRunner.
        
        Args:
            debug_mode: Enable verbose logging for debugging
            batch_size: Number of prompts to process in each batch
            timeout: Timeout for API calls in seconds
            max_retries: Maximum number of retry attempts for failed calls
        """
        self.debug_mode = debug_mode
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self._setup_logging()
        self.model_metadata = {}
        
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('inference_runner.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_model_availability(self, model_name: str) -> bool:
        """
        Check if the specified Ollama model is available.
        
        Args:
            model_name: Name of the Ollama model to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Try to list models using Ollama CLI
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                # Check if model name appears in the list
                return model_name in result.stdout
            else:
                self.logger.warning(f"Failed to list models: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            self.logger.error(f"Error checking model availability: {e}")
            return False
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Get metadata for the specified model.
        
        Args:
            model_name: Name of the Ollama model
            
        Returns:
            Dictionary containing model metadata
            
        Raises:
            ModelNotFoundError: If model is not found
        """
        try:
            result = subprocess.run(
                ['ollama', 'show', model_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                # Parse the show output to extract metadata
                metadata = {
                    'name': model_name,
                    'available': True,
                    'size': self._extract_model_size(result.stdout),
                    'parameters': self._extract_parameters(result.stdout)
                }
                self.model_metadata = metadata
                self.logger.info(f"Model metadata: {metadata}")
                return metadata
            else:
                raise ModelNotFoundError(f"Model '{model_name}' not found: {result.stderr}")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            raise ModelNotFoundError(f"Error getting model metadata: {e}")
    
    def _extract_model_size(self, show_output: str) -> Optional[str]:
        """Extract model size from ollama show output."""
        size_match = re.search(r'size:\s*([^\n]+)', show_output, re.IGNORECASE)
        return size_match.group(1).strip() if size_match else None
    
    def _extract_parameters(self, show_output: str) -> Optional[str]:
        """Extract parameter count from ollama show output."""
        param_match = re.search(r'parameters:\s*([^\n]+)', show_output, re.IGNORECASE)
        return param_match.group(1).strip() if param_match else None
    
    def run_inference(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """
        Run inference on a single prompt using the specified model.
        
        Args:
            prompt: Formatted prompt string
            model_name: Name of the Ollama model to use
            
        Returns:
            Dictionary containing prediction, rationale, and confidence
            
        Raises:
            InferenceError: If inference fails
            ResponseParsingError: If response cannot be parsed
        """
        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    ['ollama', 'run', model_name, prompt],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    response = result.stdout.strip()
                    return self._parse_response(response)
                else:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {result.stderr}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)  # Brief delay before retry
                    else:
                        raise InferenceError(f"All {self.max_retries} attempts failed: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Attempt {attempt + 1} timed out")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise InferenceError(f"All {self.max_retries} attempts timed out")
            except Exception as e:
                raise InferenceError(f"Unexpected error during inference: {e}")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response to extract prediction, rationale, and confidence.
        
        Args:
            response: Raw model response string
            
        Returns:
            Dictionary with parsed fields
            
        Raises:
            ResponseParsingError: If response cannot be parsed
        """
        # Try to parse as JSON first
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    'prediction': parsed.get('prediction'),
                    'rationale': parsed.get('rationale', ''),
                    'confidence': parsed.get('confidence', 0.0)
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback to regex parsing
        prediction = self._extract_prediction(response)
        rationale = self._extract_rationale(response)
        confidence = self._extract_confidence(response)
        
        # For unstructured responses, return with None prediction but don't raise error
        if prediction is None:
            return {
                'prediction': None,
                'rationale': rationale or response.strip(),
                'confidence': confidence or 0.0
            }
        
        return {
            'prediction': prediction,
            'rationale': rationale or 'No rationale provided',
            'confidence': confidence or 0.0
        }
    
    def _extract_prediction(self, response: str) -> Optional[int]:
        """Extract numeric prediction from response."""
        # Look for numeric values that could be predictions
        # Use a more robust pattern that handles negative numbers correctly
        numbers = re.findall(r'(?:^|\s)(-?\d+)(?:\s|$)', response)
        if numbers:
            # Convert to int and return the first one
            try:
                return int(numbers[0])
            except ValueError:
                pass
        return None
    
    def _extract_rationale(self, response: str) -> Optional[str]:
        """Extract rationale text from response."""
        # Look for text after common rationale indicators
        rationale_patterns = [
            r'rationale[:\s]+(.+)',
            r'reason[:\s]+(.+)',
            r'because[:\s]+(.+)',
            r'explanation[:\s]+(.+)'
        ]
        
        for pattern in rationale_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no specific rationale found, return the full response
        return response.strip()
    
    def _extract_confidence(self, response: str) -> Optional[float]:
        """Extract confidence score from response."""
        # Look for confidence values between 0 and 1
        confidence_match = re.search(r'confidence[:\s]+([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
            except ValueError:
                pass
        return None
    
    def run_batch_inference(self, prompts: List[str], model_name: str, 
                           row_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of prompts.
        
        Args:
            prompts: List of formatted prompt strings
            model_name: Name of the Ollama model to use
            row_ids: Optional list of row IDs to include in results
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                self.logger.info(f"Processing prompt {i + 1}/{len(prompts)}")
                result = self.run_inference(prompt, model_name)
                
                # Add row ID if provided
                if row_ids and i < len(row_ids):
                    result['C_BioSense_ID'] = row_ids[i]
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing prompt {i + 1}: {e}")
                # Add error result
                error_result = {
                    'prediction': None,
                    'rationale': f"Error: {str(e)}",
                    'confidence': 0.0
                }
                if row_ids and i < len(row_ids):
                    error_result['C_BioSense_ID'] = row_ids[i]
                results.append(error_result)
        
        return results
    
    def process_dataset(self, df: pd.DataFrame, model_name: str, 
                       prompt_column: str = 'formatted_prompt') -> pd.DataFrame:
        """
        Process entire dataset through inference.
        
        Args:
            df: DataFrame with formatted prompts
            model_name: Name of the Ollama model to use
            prompt_column: Name of column containing formatted prompts
            
        Returns:
            DataFrame with inference results added
        """
        if prompt_column not in df.columns:
            raise ValueError(f"Prompt column '{prompt_column}' not found in DataFrame")
        
        # Check model availability
        if not self.check_model_availability(model_name):
            raise ModelNotFoundError(f"Model '{model_name}' is not available")
        
        # Get model metadata
        self.get_model_metadata(model_name)
        
        # Extract prompts and row IDs
        prompts = df[prompt_column].tolist()
        row_ids = df['C_BioSense_ID'].tolist() if 'C_BioSense_ID' in df.columns else None
        
        # Process in batches
        all_results = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_row_ids = row_ids[i:i + self.batch_size] if row_ids else None
            
            self.logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(prompts) + self.batch_size - 1) // self.batch_size}")
            batch_results = self.run_batch_inference(batch_prompts, model_name, batch_row_ids)
            all_results.extend(batch_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Merge with original DataFrame
        if 'C_BioSense_ID' in df.columns and 'C_BioSense_ID' in results_df.columns:
            # Merge on row ID
            merged_df = df.merge(results_df, on='C_BioSense_ID', how='left')
        else:
            # Concatenate side by side
            merged_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        
        return merged_df
    
    def save_results(self, df: pd.DataFrame, output_path: str, 
                    format: str = 'csv') -> None:
        """
        Save inference results to file.
        
        Args:
            df: DataFrame with inference results
            output_path: Path to save results
            format: Output format ('csv' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Results saved to {output_path}")
    
    def create_inference_metadata(self, model_name: str, df: pd.DataFrame, 
                                 processing_time: float) -> Dict[str, Any]:
        """
        Create metadata for the inference run.
        
        Args:
            model_name: Name of the model used
            df: DataFrame with results
            processing_time: Total processing time in seconds
            
        Returns:
            Dictionary containing inference metadata
        """
        metadata = {
            'model_name': model_name,
            'model_metadata': self.model_metadata,
            'total_rows': len(df),
            'processing_time_seconds': processing_time,
            'batch_size': self.batch_size,
            'timestamp': pd.Timestamp.now().isoformat(),
            'columns_in_output': list(df.columns),
            'prediction_stats': self._calculate_prediction_stats(df)
        }
        
        return metadata
    
    def _calculate_prediction_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics on predictions."""
        if 'prediction' not in df.columns:
            return {}
        
        predictions = df['prediction'].dropna()
        if len(predictions) == 0:
            return {}
        
        stats = {
            'total_predictions': len(predictions),
            'unique_predictions': predictions.unique().tolist(),
            'prediction_counts': predictions.value_counts().to_dict(),
            'mean_confidence': df.get('confidence', pd.Series([0.0] * len(df))).mean()
        }
        
        return stats
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str) -> None:
        """
        Save inference metadata to file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Metadata saved to {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on formatted dataset')
    parser.add_argument('input_file', help='Path to formatted dataset CSV')
    parser.add_argument('model_name', help='Name of Ollama model to use')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = InferenceRunner(debug_mode=args.debug, batch_size=args.batch_size)
    
    try:
        # Load dataset
        df = pd.read_csv(args.input_file)
        
        # Run inference
        start_time = time.time()
        results_df = runner.process_dataset(df, args.model_name)
        processing_time = time.time() - start_time
        
        # Create metadata
        metadata = runner.create_inference_metadata(args.model_name, results_df, processing_time)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        runner.save_results(results_df, output_dir / 'inference_results.csv')
        runner.save_metadata(metadata, output_dir / 'inference_metadata.json')
        
        print(f"Inference completed successfully!")
        print(f"Processed {len(results_df)} rows in {processing_time:.2f} seconds")
        print(f"Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 