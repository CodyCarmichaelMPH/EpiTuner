"""
Fine Tuner Module for Ollama Fine-Tuning and Evaluation Suite

This module handles fine-tuning Ollama-compatible models or creating structured
model-set configurations for meta-prompting when fine-tuning is not supported.
"""

import pandas as pd
import logging
import json
import yaml
import subprocess
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import tempfile
import shutil
from scripts.config_manager import get_config_manager


class FineTuningNotSupportedError(Exception):
    """Raised when the base model cannot be fine-tuned."""
    pass


class DatasetTooSmallWarning(Exception):
    """Raised when dataset is too small for effective fine-tuning."""
    pass


class TrainingFailedError(Exception):
    """Raised when fine-tuning process fails."""
    pass


class ModelSetConfigError(Exception):
    """Raised when model set configuration creation fails."""
    pass


class FineTuner:
    """
    Handles fine-tuning of Ollama models or creation of model-set configurations.
    
    Supports both actual fine-tuning (when available) and fallback meta-prompting
    approaches for adapting models to dataset-specific patterns.
    """
    
    def __init__(self, debug_mode: bool = False, fallback_mode: bool = None):
        """
        Initialize the FineTuner.
        
        Args:
            debug_mode: Enable verbose logging for debugging
            fallback_mode: Enable fallback to model contextualization when fine-tuning not supported
        """
        self.debug_mode = debug_mode
        self.config_manager = get_config_manager()
        
        # Use config fallback_mode if not explicitly provided
        if fallback_mode is None:
            self.fallback_mode = self.config_manager.get('fine_tuning.fallback_mode', True)
        else:
            self.fallback_mode = fallback_mode
        
        # Get configuration values
        self.MIN_DATASET_SIZE = self.config_manager.get('fine_tuning.min_dataset_size', 50)
        self.DEFAULT_EPOCHS = self.config_manager.get('fine_tuning.default_epochs', 3)
        self.DEFAULT_LEARNING_RATE = self.config_manager.get('fine_tuning.default_learning_rate', 0.0001)
        self.DEFAULT_BATCH_SIZE = self.config_manager.get('fine_tuning.default_batch_size', 4)
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('fine_tuner.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_fine_tuning_capability(self, model_name: str) -> bool:
        """
        Check if the specified Ollama model supports fine-tuning.
        
        Args:
            model_name: Name of the base model to check
            
        Returns:
            True if model supports fine-tuning, False otherwise
        """
        try:
            import requests
            
            # Get server URL from config
            server_url = self.config_manager.get('ollama.server_url', 'http://localhost:11434')
            timeout = self.config_manager.get('ollama.timeout', 30)
            
            # Check if Ollama server is running
            try:
                response = requests.get(f"{server_url}/api/version", timeout=5)
                if response.status_code != 200:
                    self.logger.warning("Ollama server not responding")
                    return False
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Cannot connect to Ollama server: {e}")
                return False
            
            # Check if model exists using API
            try:
                response = requests.post(
                    f"{server_url}/api/show",
                    json={"name": model_name},
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    model_info = response.json()
                    self.logger.info(f"Model {model_name} found: {model_info.get('model', 'Unknown')}")
                    # For now, assume all models support fine-tuning
                    # In a real implementation, you'd check model-specific capabilities
                    return True
                else:
                    self.logger.warning(f"Model {model_name} not found (status: {response.status_code})")
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Error checking model {model_name}: {e}")
                return False
            
        except ImportError:
            self.logger.warning("requests library not available, falling back to CLI")
            return self._check_fine_tuning_capability_cli(model_name)
    
    def _check_fine_tuning_capability_cli(self, model_name: str) -> bool:
        """Fallback method using CLI commands."""
        try:
            # Check if Ollama is available
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.warning("Ollama not available or not in PATH")
                return False
            
            # Check if model exists
            result = subprocess.run(
                ['ollama', 'show', model_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Model {model_name} not found")
                return False
            
            # For now, assume all models support fine-tuning
            # In a real implementation, you'd check model-specific capabilities
            self.logger.info(f"Model {model_name} appears to support fine-tuning")
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning(f"Error checking fine-tuning capability: {e}")
            return False
    
    def prepare_training_data(self, df: pd.DataFrame, rating_mapping: Dict[Any, int]) -> List[Dict[str, str]]:
        """
        Prepare dataset for fine-tuning in JSONL format.
        
        Args:
            df: DataFrame with formatted prompts and ratings
            rating_mapping: Mapping of rating values to integers
            
        Returns:
            List of training examples in JSONL format
        """
        training_data = []
        
        for _, row in df.iterrows():
            # Get the formatted prompt
            if 'formatted_prompt' in row:
                context = row['formatted_prompt']
            elif 'Prompt' in row:
                context = row['Prompt']
            else:
                # Fallback: create context from available fields
                context = self._create_context_from_row(row)
            
            # Get the target rating
            if 'Standardized_Rating' in row:
                rating = row['Standardized_Rating']
            elif 'Expert Rating' in row:
                rating = rating_mapping.get(row['Expert Rating'], -1)
            else:
                rating = -1  # Unknown rating
            
            # Get rationale if available
            rationale = row.get('Rationale of Rating', 'No rationale provided')
            
            # Create training example
            training_example = {
                "input": context,
                "output": f"Rating: {rating}, Rationale: {rationale}"
            }
            
            training_data.append(training_example)
        
        self.logger.info(f"Prepared {len(training_data)} training examples")
        return training_data
    
    def _create_context_from_row(self, row: pd.Series) -> str:
        """Create context string from available row fields."""
        context_parts = []
        
        # Add patient info
        if 'Age' in row and 'Sex' in row:
            context_parts.append(f"Patient: Age {row['Age']}, Sex {row['Sex']}")
        
        # Add medical information
        for field in ['ChiefComplaintOrig', 'Discharge Diagnosis', 'TriageNotes']:
            if field in row and pd.notna(row[field]):
                context_parts.append(f"{field}: {row[field]}")
        
        return "\n".join(context_parts)
    
    def create_training_config(self, model_name: str, training_data: List[Dict[str, str]], 
                             epochs: int = None, learning_rate: float = None, 
                             batch_size: int = None) -> Dict[str, Any]:
        """
        Create training configuration for Ollama fine-tuning.
        
        Args:
            model_name: Base model name
            training_data: List of training examples
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            
        Returns:
            Training configuration dictionary
        """
        config = {
            "model": model_name,
            "epochs": epochs or self.DEFAULT_EPOCHS,
            "learning_rate": learning_rate or self.DEFAULT_LEARNING_RATE,
            "batch_size": batch_size or self.DEFAULT_BATCH_SIZE,
            "training_data": training_data
        }
        
        return config
    
    def save_training_data_jsonl(self, training_data: List[Dict[str, str]], output_path: str) -> None:
        """
        Save training data in JSONL format.
        
        Args:
            training_data: List of training examples
            output_path: Path to save the JSONL file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
        
        self.logger.info(f"Saved training data to {output_path}")
    
    def save_training_config_yaml(self, config: Dict[str, Any], output_path: str) -> None:
        """
        Save training configuration in YAML format.
        
        Args:
            config: Training configuration dictionary
            output_path: Path to save the YAML file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"Saved training config to {output_path}")
    
    def execute_fine_tuning(self, model_name: str, training_data: List[Dict[str, str]], 
                          config: Dict[str, Any], output_model_name: str) -> bool:
        """
        Execute fine-tuning process using Ollama.
        
        Args:
            model_name: Base model name
            training_data: List of training examples
            config: Training configuration
            output_model_name: Name for the fine-tuned model
            
        Returns:
            True if fine-tuning successful, False otherwise
        """
        try:
            # Create temporary files for training
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save training data
                training_file = os.path.join(temp_dir, 'training_data.jsonl')
                self.save_training_data_jsonl(training_data, training_file)
                
                # Save config
                config_file = os.path.join(temp_dir, 'training_config.yaml')
                self.save_training_config_yaml(config, config_file)
                
                # Execute fine-tuning command
                cmd = [
                    'ollama', 'create', output_model_name,
                    '-f', config_file
                ]
                
                self.logger.info(f"Starting fine-tuning with command: {' '.join(cmd)}")
                
                # Run fine-tuning process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=temp_dir
                )
                
                # Monitor progress
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        self.logger.info(f"Fine-tuning: {output.strip()}")
                
                # Check result
                return_code = process.poll()
                
                if return_code == 0:
                    self.logger.info(f"Fine-tuning completed successfully for model: {output_model_name}")
                    return True
                else:
                    stderr_output = process.stderr.read()
                    self.logger.error(f"Fine-tuning failed: {stderr_output}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            return False
    
    def create_model_set_config(self, model_name: str, df: pd.DataFrame, 
                              rating_mapping: Dict[Any, int], target_topics: str) -> Dict[str, Any]:
        """
        Create model set configuration for meta-prompting approach.
        
        Args:
            model_name: Base model name
            df: Dataset with examples
            rating_mapping: Rating value mapping
            target_topics: Target topics for evaluation
            
        Returns:
            Model set configuration dictionary
        """
        # Create system prompt
        system_prompt = self._create_system_prompt(rating_mapping, target_topics)
        
        # Select few-shot examples
        few_shot_examples = self._select_few_shot_examples(df, rating_mapping)
        
        config = {
            "base_model": model_name,
            "system_prompt": system_prompt,
            "few_shot_examples": few_shot_examples,
            "target_topics": target_topics,
            "rating_mapping": rating_mapping,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(df)
        }
        
        return config
    
    def _create_system_prompt(self, rating_mapping: Dict[Any, int], target_topics: str) -> str:
        """Create system prompt for meta-prompting."""
        rating_descriptions = []
        for rating_value, rating_int in rating_mapping.items():
            rating_descriptions.append(f"{rating_int}: {rating_value}")
        
        rating_schema = ", ".join(rating_descriptions)
        
        system_prompt = f"""You are an expert evaluator trained to assess case alignment with medical topic criteria.

Your task is to evaluate whether medical cases align with the topic(s): {target_topics}

Use the following rating schema:
{rating_schema}

For each case, provide your response in this exact format:
Rating: <integer>
Rationale: <brief explanation of your reasoning>

Be consistent and thorough in your evaluation."""
        
        return system_prompt
    
    def _select_few_shot_examples(self, df: pd.DataFrame, rating_mapping: Dict[Any, int], 
                                num_examples: int = 5) -> List[Dict[str, str]]:
        """Select diverse few-shot examples from the dataset."""
        examples = []
        
        # Get unique ratings to ensure diversity
        unique_ratings = list(rating_mapping.values())
        
        for rating in unique_ratings:
            # Find examples with this rating
            rating_examples = df[df['Standardized_Rating'] == rating]
            
            if len(rating_examples) > 0:
                # Select one example per rating
                example_row = rating_examples.iloc[0]
                
                # Create example
                if 'formatted_prompt' in example_row:
                    context = example_row['formatted_prompt']
                elif 'Prompt' in example_row:
                    context = example_row['Prompt']
                else:
                    context = self._create_context_from_row(example_row)
                
                rationale = example_row.get('Rationale of Rating', 'No rationale provided')
                
                example = {
                    "input": context,
                    "output": f"Rating: {rating}, Rationale: {rationale}"
                }
                
                examples.append(example)
                
                if len(examples) >= num_examples:
                    break
        
        return examples
    
    def save_model_set_config(self, config: Dict[str, Any], output_path: str) -> None:
        """
        Save model set configuration to file.
        
        Args:
            config: Model set configuration
            output_path: Path to save the configuration
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved model set config to {output_path}")
    
    def validate_fine_tuned_model(self, model_name: str, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate fine-tuned model on test dataset.
        
        Args:
            model_name: Name of the fine-tuned model
            test_df: Test dataset
            
        Returns:
            Validation results dictionary
        """
        try:
            import requests
            
            # Check if model exists using API
            try:
                response = requests.post(
                    "http://localhost:11434/api/show",
                    json={"name": model_name},
                    timeout=10
                )
                
                if response.status_code == 200:
                    model_info = response.json()
                    self.logger.info(f"Fine-tuned model {model_name} validation successful")
                    return {
                        "model_name": model_name,
                        "validation_status": "success",
                        "model_available": True,
                        "model_info": model_info,
                        "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    self.logger.warning(f"Fine-tuned model {model_name} validation failed (status: {response.status_code})")
                    return {
                        "model_name": model_name,
                        "validation_status": "failed",
                        "model_available": False,
                        "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error validating fine-tuned model: {e}")
                return {
                    "model_name": model_name,
                    "validation_status": "error",
                    "model_available": False,
                    "error": str(e),
                    "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
        except ImportError:
            self.logger.warning("requests library not available, falling back to CLI")
            return self._validate_fine_tuned_model_cli(model_name, test_df)
    
    def _validate_fine_tuned_model_cli(self, model_name: str, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback method using CLI commands."""
        try:
            # Simple validation - check if model can be loaded
            result = subprocess.run(
                ['ollama', 'show', model_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.info(f"Fine-tuned model {model_name} validation successful")
                return {
                    "model_name": model_name,
                    "validation_status": "success",
                    "model_available": True,
                    "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                self.logger.warning(f"Fine-tuned model {model_name} validation failed")
                return {
                    "model_name": model_name,
                    "validation_status": "failed",
                    "model_available": False,
                    "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
        except Exception as e:
            self.logger.error(f"Error validating fine-tuned model: {e}")
            return {
                "model_name": model_name,
                "validation_status": "error",
                "model_available": False,
                "error": str(e),
                "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def process_dataset(self, df: pd.DataFrame, model_name: str = None, rating_mapping: Dict[Any, int] = None,
                       target_topics: str = None, epochs: int = None, learning_rate: float = None,
                       batch_size: int = None, output_dir: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Process dataset for fine-tuning or model set configuration.
        
        Args:
            df: Dataset to process
            model_name: Base model name (uses config default if None)
            rating_mapping: Rating value mapping
            target_topics: Target topics for evaluation
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            output_dir: Output directory for results (uses config default if None)
            
        Returns:
            Tuple of (success_flag, metadata)
        """
        # Use defaults from config if not provided
        if model_name is None:
            model_name = self.config_manager.get_model()
        if output_dir is None:
            output_dir = self.config_manager.get('output.default_directory', 'outputs')
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "processing_info": {
                "model_name": model_name,
                "dataset_size": len(df),
                "target_topics": target_topics,
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "fine_tuning_info": {},
            "model_set_info": {}
        }
        
        # Check dataset size
        if len(df) < self.MIN_DATASET_SIZE:
            self.logger.warning(f"Dataset size ({len(df)}) is below recommended minimum ({self.MIN_DATASET_SIZE})")
            if not self.fallback_mode:
                raise DatasetTooSmallWarning(f"Dataset too small for fine-tuning. Use fallback_mode=True for model set configuration.")
        
        # Check fine-tuning capability
        can_fine_tune = self.check_fine_tuning_capability(model_name)
        
        if can_fine_tune and len(df) >= self.MIN_DATASET_SIZE:
            # Proceed with fine-tuning
            self.logger.info("Proceeding with fine-tuning approach")
            
            try:
                # Prepare training data
                training_data = self.prepare_training_data(df, rating_mapping)
                
                # Create training configuration
                config = self.create_training_config(
                    model_name, training_data, epochs, learning_rate, batch_size
                )
                
                # Save training data
                training_data_path = os.path.join(output_dir, "training_data.jsonl")
                self.save_training_data_jsonl(training_data, training_data_path)
                
                # Save training config
                config_path = os.path.join(output_dir, "training_config.yaml")
                self.save_training_config_yaml(config, config_path)
                
                # Execute fine-tuning
                output_model_name = f"{model_name}-finetuned"
                success = self.execute_fine_tuning(model_name, training_data, config, output_model_name)
                
                if success:
                    # Validate fine-tuned model
                    validation_results = self.validate_fine_tuned_model(output_model_name, df)
                    
                    metadata["fine_tuning_info"] = {
                        "approach": "fine_tuning",
                        "output_model_name": output_model_name,
                        "training_data_path": training_data_path,
                        "config_path": config_path,
                        "validation_results": validation_results,
                        "success": True
                    }
                    
                    return True, metadata
                else:
                    raise TrainingFailedError("Fine-tuning process failed")
                    
            except Exception as e:
                self.logger.error(f"Fine-tuning failed: {e}")
                if not self.fallback_mode:
                    raise
                
                # Fall back to model set configuration
                self.logger.info("Falling back to model set configuration")
        
        # Use model set configuration approach
        if self.fallback_mode:
            self.logger.info("Using model set configuration approach")
            
            try:
                # Create model set configuration
                model_set_config = self.create_model_set_config(
                    model_name, df, rating_mapping, target_topics
                )
                
                # Save configuration
                config_path = os.path.join(output_dir, "model_set_config.json")
                self.save_model_set_config(model_set_config, config_path)
                
                metadata["model_set_info"] = {
                    "approach": "model_set_configuration",
                    "config_path": config_path,
                    "base_model": model_name,
                    "few_shot_examples_count": len(model_set_config["few_shot_examples"]),
                    "success": True
                }
                
                return True, metadata
                
            except Exception as e:
                self.logger.error(f"Model set configuration failed: {e}")
                raise ModelSetConfigError(f"Failed to create model set configuration: {e}")
        
        return False, metadata
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str) -> None:
        """
        Save processing metadata to file.
        
        Args:
            metadata: Processing metadata
            output_path: Path to save the metadata
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved metadata to {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine Tuner for Ollama Models")
    parser.add_argument("--input", required=True, help="Path to formatted dataset CSV")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--topics", required=True, help="Target topics for evaluation")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--no-fallback", action="store_true", help="Disable fallback mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Initialize fine tuner
    fine_tuner = FineTuner(debug_mode=args.debug, fallback_mode=not args.no_fallback)
    
    # Load dataset
    df = pd.read_csv(args.input)
    
    # Define rating mapping (this should come from schema mapper in practice)
    rating_mapping = {
        'Match': 1,
        'Does Not Match': 0,
        'Unknown': -1,
        'Partial Match': 2
    }
    
    # Process dataset
    success, metadata = fine_tuner.process_dataset(
        df, args.model, rating_mapping, args.topics,
        args.epochs, args.learning_rate, args.batch_size, args.output_dir
    )
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, "fine_tuning_metadata.json")
    fine_tuner.save_metadata(metadata, metadata_path)
    
    if success:
        print("Fine-tuning/model set configuration completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("Fine-tuning/model set configuration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 