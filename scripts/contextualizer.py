"""
Contextualizer Module for Ollama Fine-Tuning and Evaluation Suite

This module provides a fallback approach for scenarios where fine-tuning is not available,
underperforms, or is not cost-effective. The contextualizer builds structured meta-prompts
using dataset rows, expert ratings, rationales, and schema mapping to guide the base Ollama
model to make improved predictions.
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


class ContextualizerError(Exception):
    """Raised when contextualizer operations fail."""
    pass


class PromptConstructionError(Exception):
    """Raised when meta-prompt construction fails."""
    pass


class ModelResponseError(Exception):
    """Raised when model response parsing fails."""
    pass


class Contextualizer:
    """
    Handles contextual prompting for base Ollama models without fine-tuning.
    
    Builds structured meta-prompts using few-shot examples from the dataset
    to guide the model in making predictions on new data.
    """
    
    def __init__(self, debug_mode: bool = False, max_rows_context: int = 10, 
                 timeout: int = 30, max_retries: int = 3):
        """
        Initialize the Contextualizer.
        
        Args:
            debug_mode: Enable verbose logging for debugging
            max_rows_context: Maximum number of example rows to include in context
            timeout: Timeout for API calls in seconds
            max_retries: Maximum number of retry attempts for failed calls
        """
        self.debug_mode = debug_mode
        self.max_rows_context = max_rows_context
        self.timeout = timeout
        self.max_retries = max_retries
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('contextualizer.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def sample_few_shot_examples(self, df: pd.DataFrame, rating_mapping: Dict[Any, int], 
                                topics: str) -> pd.DataFrame:
        """
        Select a subset of rows for few-shot examples.
        
        Args:
            df: DataFrame with expert ratings and rationales
            rating_mapping: Dictionary mapping original ratings to standardized values
            topics: Topics to evaluate alignment against
            
        Returns:
            DataFrame with selected few-shot examples
            
        Raises:
            ContextualizerError: If no suitable examples found
        """
        try:
            # Filter rows with clear expert ratings and rationales
            valid_examples = df[
                (df['Expert Rating'].notna()) & 
                (df['Rationale of Rating'].notna()) &
                (df['Rationale of Rating'].str.strip() != '')
            ].copy()
            
            if len(valid_examples) == 0:
                raise ContextualizerError("No rows with expert ratings and rationales found")
            
            # Add standardized rating column
            valid_examples['Standardized_Rating'] = valid_examples['Expert Rating'].map(rating_mapping)
            
            # Remove rows with unmapped ratings
            valid_examples = valid_examples[valid_examples['Standardized_Rating'].notna()]
            
            if len(valid_examples) == 0:
                raise ContextualizerError("No rows with valid rating mappings found")
            
            # Balance examples across different rating values
            balanced_examples = []
            rating_counts = valid_examples['Standardized_Rating'].value_counts()
            
            # Calculate examples per rating (ensure balanced representation)
            examples_per_rating = min(
                self.max_rows_context // len(rating_counts),
                rating_counts.min()
            )
            
            for rating in rating_counts.index:
                rating_examples = valid_examples[
                    valid_examples['Standardized_Rating'] == rating
                ]
                sampled_examples = rating_examples.sample(n=min(examples_per_rating, len(rating_examples)), random_state=42)
                balanced_examples.append(sampled_examples)
            
            # Combine and limit total examples
            few_shot_df = pd.concat(balanced_examples, ignore_index=True)
            few_shot_df = few_shot_df.head(self.max_rows_context)
            
            self.logger.info(f"Selected {len(few_shot_df)} few-shot examples across {len(rating_counts)} rating values")
            return few_shot_df
            
        except Exception as e:
            self.logger.error(f"Error sampling few-shot examples: {e}")
            raise ContextualizerError(f"Failed to sample few-shot examples: {e}")
    
    def build_schema_description(self, rating_mapping: Dict[Any, int]) -> str:
        """
        Build a description of the rating schema for the prompt.
        
        Args:
            rating_mapping: Dictionary mapping original ratings to standardized values
            
        Returns:
            String description of the rating schema
        """
        schema_lines = ["Use this rating schema:"]
        
        # Create reverse mapping for display
        reverse_mapping = {v: k for k, v in rating_mapping.items()}
        
        for rating_value in sorted(reverse_mapping.keys()):
            original_rating = reverse_mapping[rating_value]
            schema_lines.append(f"- Rating {rating_value}: {original_rating}")
        
        return "\n".join(schema_lines)
    
    def format_example_row(self, row: pd.Series, example_num: int) -> str:
        """
        Format a single example row for the prompt.
        
        Args:
            row: DataFrame row with patient data
            example_num: Example number for display
            
        Returns:
            Formatted string representation of the example
        """
        # Extract key information
        patient_id = row.get('C_BioSense_ID', 'Unknown')
        complaint = row.get('ChiefComplaintOrig', '')
        diagnosis = row.get('Discharge Diagnosis', '')
        triage_notes = row.get('TriageNotes', '')
        rating = row.get('Standardized_Rating', '')
        rationale = row.get('Rationale of Rating', '')
        
        # Build context string
        context_parts = []
        if complaint:
            context_parts.append(f"Chief Complaint: {complaint}")
        if diagnosis:
            context_parts.append(f"Diagnosis: {diagnosis}")
        if triage_notes:
            context_parts.append(f"Triage Notes: {triage_notes}")
        
        context_str = " | ".join(context_parts)
        
        # Format the example
        example = f"{example_num}. Record: {context_str} → Rating: {rating} | Rationale: {rationale}"
        
        return example
    
    def construct_meta_prompt(self, few_shot_df: pd.DataFrame, query_row: pd.Series, 
                            topics: str, rating_mapping: Dict[Any, int]) -> str:
        """
        Construct the meta-prompt for contextual inference.
        
        Args:
            few_shot_df: DataFrame with few-shot examples
            query_row: Row to evaluate (new data)
            topics: Topics to evaluate alignment against
            rating_mapping: Dictionary mapping original ratings to standardized values
            
        Returns:
            Constructed meta-prompt string
            
        Raises:
            PromptConstructionError: If prompt construction fails
        """
        try:
            # Build schema description
            schema_description = self.build_schema_description(rating_mapping)
            
            # Build examples section
            examples = []
            for idx, row in few_shot_df.iterrows():
                example = self.format_example_row(row, idx + 1)
                examples.append(example)
            
            examples_section = "\n".join(examples)
            
            # Format query row
            query_context = self.format_example_row(query_row, 0)  # 0 indicates it's the query
            query_context = query_context.replace("→ Rating: 0 | Rationale:", "→ [TO EVALUATE]")
            
            # Construct the full prompt
            prompt = f"""You are evaluating patient record alignment with topics: {topics}.
{schema_description}

Examples:
{examples_section}

New record to evaluate:
{query_context}

Respond with:
- Numeric rating (use the schema above)
- Brief rationale explaining your decision

Format your response as:
Rating: [number]
Rationale: [explanation]"""
            
            self.logger.debug(f"Constructed meta-prompt with {len(few_shot_df)} examples")
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error constructing meta-prompt: {e}")
            raise PromptConstructionError(f"Failed to construct meta-prompt: {e}")
    
    def check_model_availability(self, model_name: str) -> bool:
        """
        Check if the specified Ollama model is available.
        
        Args:
            model_name: Name of the Ollama model to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return model_name in result.stdout
            else:
                self.logger.warning(f"Failed to list models: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
            return False
    
    def run_contextual_inference(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """
        Run contextual inference using the constructed meta-prompt.
        
        Args:
            prompt: Constructed meta-prompt
            model_name: Name of the Ollama model to use
            
        Returns:
            Dictionary with prediction, confidence, and rationale
            
        Raises:
            ModelResponseError: If model response parsing fails
        """
        try:
            # Check model availability
            if not self.check_model_availability(model_name):
                raise ModelResponseError(f"Model {model_name} not available")
            
            # Run inference using Ollama CLI
            result = subprocess.run(
                ['ollama', 'run', model_name, prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                raise ModelResponseError(f"Ollama inference failed: {result.stderr}")
            
            response = result.stdout.strip()
            
            # Parse the response
            parsed_response = self._parse_response(response)
            
            return parsed_response
            
        except subprocess.TimeoutExpired:
            raise ModelResponseError(f"Inference timed out after {self.timeout} seconds")
        except Exception as e:
            self.logger.error(f"Error running contextual inference: {e}")
            raise ModelResponseError(f"Failed to run contextual inference: {e}")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the model response to extract rating and rationale.
        
        Args:
            response: Raw model response string
            
        Returns:
            Dictionary with prediction, confidence, and rationale
            
        Raises:
            ModelResponseError: If response parsing fails
        """
        try:
            # Extract rating
            rating_match = re.search(r'Rating:\s*(\d+)', response, re.IGNORECASE)
            if not rating_match:
                raise ModelResponseError("Could not extract rating from response")
            
            rating = int(rating_match.group(1))
            
            # Extract rationale
            rationale_match = re.search(r'Rationale:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
            if not rationale_match:
                # Try alternative patterns
                rationale_match = re.search(r'Rationale:\s*(.+)', response, re.IGNORECASE)
            
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided"
            
            return {
                "prediction": rating,
                "confidence": None,  # Contextualizer doesn't provide confidence scores
                "rationale": rationale
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            raise ModelResponseError(f"Failed to parse model response: {e}")
    
    def evaluate_single_row(self, df: pd.DataFrame, row_index: int, topics: str, 
                          rating_mapping: Dict[Any, int], model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single row using contextual prompting.
        
        Args:
            df: DataFrame with all data
            row_index: Index of the row to evaluate
            topics: Topics to evaluate alignment against
            rating_mapping: Dictionary mapping original ratings to standardized values
            model_name: Name of the Ollama model to use
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get the query row
            query_row = df.iloc[row_index].copy()
            
            # Sample few-shot examples (excluding the query row)
            df_without_query = df.drop(index=row_index).reset_index(drop=True)
            few_shot_df = self.sample_few_shot_examples(df_without_query, rating_mapping, topics)
            
            # Construct meta-prompt
            prompt = self.construct_meta_prompt(few_shot_df, query_row, topics, rating_mapping)
            
            # Run inference
            result = self.run_contextual_inference(prompt, model_name)
            
            # Add row identifier
            result["C_BioSense_ID"] = query_row.get('C_BioSense_ID', f"row_{row_index}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating row {row_index}: {e}")
            return {
                "C_BioSense_ID": df.iloc[row_index].get('C_BioSense_ID', f"row_{row_index}"),
                "prediction": None,
                "confidence": None,
                "rationale": f"Error: {str(e)}"
            }
    
    def process_dataset(self, df: pd.DataFrame, topics: str, rating_mapping: Dict[Any, int], 
                       model_name: str, output_dir: str = "outputs") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process entire dataset using contextual prompting.
        
        Args:
            df: DataFrame with all data
            topics: Topics to evaluate alignment against
            rating_mapping: Dictionary mapping original ratings to standardized values
            model_name: Name of the Ollama model to use
            output_dir: Directory to save results
            
        Returns:
            Tuple of (results DataFrame, metadata dictionary)
        """
        try:
            self.logger.info(f"Starting contextual evaluation of {len(df)} rows")
            start_time = time.time()
            
            results = []
            
            for idx, row in df.iterrows():
                self.logger.debug(f"Evaluating row {idx + 1}/{len(df)}")
                
                result = self.evaluate_single_row(df, idx, topics, rating_mapping, model_name)
                results.append(result)
                
                # Add small delay to avoid overwhelming the model
                time.sleep(0.1)
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = self.create_evaluation_metadata(
                model_name, df, results_df, rating_mapping, topics, processing_time
            )
            
            # Save results
            output_path = Path(output_dir) / "contextual_evaluation_results.csv"
            results_df.to_csv(output_path, index=False)
            
            # Save metadata
            metadata_path = Path(output_dir) / "contextual_evaluation_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Contextual evaluation complete. Results saved to {output_path}")
            
            return results_df, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {e}")
            raise ContextualizerError(f"Failed to process dataset: {e}")
    
    def create_evaluation_metadata(self, model_name: str, input_df: pd.DataFrame, 
                                 results_df: pd.DataFrame, rating_mapping: Dict[Any, int],
                                 topics: str, processing_time: float) -> Dict[str, Any]:
        """
        Create metadata for the contextual evaluation.
        
        Args:
            model_name: Name of the model used
            input_df: Input DataFrame
            results_df: Results DataFrame
            rating_mapping: Rating mapping used
            topics: Topics evaluated
            processing_time: Time taken for processing
            
        Returns:
            Metadata dictionary
        """
        # Calculate statistics
        successful_predictions = results_df[results_df['prediction'].notna()]
        failed_predictions = results_df[results_df['prediction'].isna()]
        
        metadata = {
            "evaluation_type": "contextual",
            "model_name": model_name,
            "topics": topics,
            "rating_mapping": rating_mapping,
            "processing_time_seconds": processing_time,
            "input_dataset_size": len(input_df),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(failed_predictions),
            "success_rate": len(successful_predictions) / len(results_df) if len(results_df) > 0 else 0,
            "prediction_distribution": successful_predictions['prediction'].value_counts().to_dict() if len(successful_predictions) > 0 else {},
            "timestamp": pd.Timestamp.now().isoformat(),
            "max_rows_context": self.max_rows_context,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
        
        return metadata


def main():
    """Main function for testing the Contextualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Contextualizer for Ollama models")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--topics", required=True, help="Topics to evaluate")
    parser.add_argument("--model", default="phi3:mini", help="Ollama model name")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max-context", type=int, default=10, help="Maximum context rows")
    
    args = parser.parse_args()
    
    try:
        # Initialize contextualizer
        contextualizer = Contextualizer(
            debug_mode=args.debug,
            max_rows_context=args.max_context
        )
        
        # Load dataset
        df = pd.read_csv(args.input)
        
        # Create a simple rating mapping (assuming standard 0, 1, 2 schema)
        rating_mapping = {0: 0, 1: 1, 2: 2}
        
        # Process dataset
        results_df, metadata = contextualizer.process_dataset(
            df, args.topics, rating_mapping, args.model, args.output
        )
        
        print(f"Evaluation complete! Results saved to {args.output}/")
        print(f"Success rate: {metadata['success_rate']:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 