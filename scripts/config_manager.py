"""
Configuration Manager Module for Ollama Fine-Tuning and Evaluation Suite

This module handles loading, saving, and updating user configuration settings
including model selection, fine-tuning parameters, and other user preferences.
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os


class ConfigManager:
    """
    Manages application configuration settings.
    
    Handles loading, saving, and updating user preferences including:
    - Ollama model selection
    - Fine-tuning parameters
    - Inference settings
    - Output preferences
    - Logging configuration
    """
    
    DEFAULT_CONFIG_PATH = "config/settings.json"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to configuration file (defaults to config/settings.json)
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._setup_logging()
        self.config = self._load_config()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found at {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "ollama": {
                "default_model": "llama2",
                "server_url": "http://localhost:11434",
                "timeout": 30
            },
            "fine_tuning": {
                "min_dataset_size": 50,
                "default_epochs": 3,
                "default_learning_rate": 0.0001,
                "default_batch_size": 4,
                "fallback_mode": True
            },
            "inference": {
                "default_batch_size": 5,
                "max_retries": 3,
                "timeout": 30
            },
            "contextualizer": {
                "max_rows_context": 10,
                "timeout": 30,
                "max_retries": 3,
                "default_prompt_template": None
            },
            "output": {
                "default_directory": "outputs",
                "save_metadata": True,
                "save_logs": True
            },
            "logging": {
                "level": "INFO",
                "save_to_file": True,
                "log_directory": "logs"
            }
        }
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure config directory exists
            config_dir = os.path.dirname(self.config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'ollama.default_model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'ollama.default_model')
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            self.logger.info(f"Set configuration {key} = {value}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting configuration {key}: {e}")
            return False
    
    def update_model(self, model_name: str) -> bool:
        """
        Update the default Ollama model.
        
        Args:
            model_name: Name of the model to set as default
            
        Returns:
            True if successful, False otherwise
        """
        success = self.set('ollama.default_model', model_name)
        if success:
            self.save_config()
        return success
    
    def get_model(self) -> str:
        """
        Get the current default model.
        
        Returns:
            Current default model name
        """
        return self.get('ollama.default_model', 'llama2')
    
    def get_available_models(self) -> list:
        """
        Get list of available models from Ollama server.
        
        Returns:
            List of available model names
        """
        try:
            import requests
            
            server_url = self.get('ollama.server_url', 'http://localhost:11434')
            response = requests.get(f"{server_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                self.logger.info(f"Found {len(models)} available models")
                return models
            else:
                self.logger.warning(f"Failed to get models: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []
    
    def validate_model(self, model_name: str) -> bool:
        """
        Validate if a model exists on the Ollama server.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            import requests
            
            server_url = self.get('ollama.server_url', 'http://localhost:11434')
            response = requests.post(
                f"{server_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error validating model {model_name}: {e}")
            return False
    
    def get_fine_tuning_config(self) -> Dict[str, Any]:
        """
        Get fine-tuning configuration.
        
        Returns:
            Fine-tuning configuration dictionary
        """
        return self.get('fine_tuning', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """
        Get inference configuration.
        
        Returns:
            Inference configuration dictionary
        """
        return self.get('inference', {})
    
    def get_contextualizer_config(self) -> Dict[str, Any]:
        """
        Get contextualizer configuration.
        
        Returns:
            Contextualizer configuration dictionary
        """
        return self.get('contextualizer', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output configuration.
        
        Returns:
            Output configuration dictionary
        """
        return self.get('output', {})
    
    def reload_config(self) -> bool:
        """
        Reload configuration from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config = self._load_config()
            self.logger.info("Configuration reloaded")
            return True
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")
            return False
    
    def export_config(self, export_path: str) -> bool:
        """
        Export current configuration to a file.
        
        Args:
            export_path: Path to export configuration to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration exported to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """
        Import configuration from a file.
        
        Args:
            import_path: Path to import configuration from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
            
            self.config = new_config
            self.save_config()
            self.logger.info(f"Configuration imported from {import_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False


# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Global ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Manager")
    parser.add_argument("--get", help="Get configuration value")
    parser.add_argument("--set", nargs=2, metavar=('KEY', 'VALUE'), help="Set configuration value")
    parser.add_argument("--model", help="Set default model")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--export", help="Export configuration to file")
    parser.add_argument("--import", dest="import_path", help="Import configuration from file")
    
    args = parser.parse_args()
    
    config_manager = get_config_manager()
    
    if args.get:
        value = config_manager.get(args.get)
        print(f"{args.get} = {value}")
    
    elif args.set:
        key, value = args.set
        success = config_manager.set(key, value)
        if success:
            config_manager.save_config()
            print(f"Set {key} = {value}")
        else:
            print(f"Failed to set {key}")
    
    elif args.model:
        success = config_manager.update_model(args.model)
        if success:
            print(f"Updated default model to {args.model}")
        else:
            print(f"Failed to update model")
    
    elif args.list_models:
        models = config_manager.get_available_models()
        print("Available models:")
        for model in models:
            print(f"  - {model}")
    
    elif args.export:
        success = config_manager.export_config(args.export)
        if success:
            print(f"Configuration exported to {args.export}")
        else:
            print("Failed to export configuration")
    
    elif args.import_path:
        success = config_manager.import_config(args.import_path)
        if success:
            print(f"Configuration imported from {args.import_path}")
        else:
            print("Failed to import configuration")
    
    else:
        print("Current configuration:")
        print(json.dumps(config_manager.config, indent=2))


if __name__ == "__main__":
    main() 