#!/usr/bin/env python3
"""
EpiTuner Training Script
Based on sft-play architecture with medical data focus
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import yaml
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Only import BitsAndBytesConfig if needed (and available)
try:
    import torch
    # Check PyTorch version first - BnB requires newer PyTorch
    torch_version = torch.__version__
    torch_major, torch_minor = map(int, torch_version.split('.')[:2])
    
    if torch_major < 2 or (torch_major == 2 and torch_minor < 1):
        print(f"Warning: PyTorch {torch_version} too old for BitsAndBytes. Disabling quantization.")
        HAS_BNB = False
        HAS_BNB_VALIDATION = False
    else:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            HAS_BNB = True
            print(f"BitsAndBytes available with PyTorch {torch_version}")
            
            # Check for validation function
            try:
                from transformers.integrations import validate_bnb_backend_availability
                HAS_BNB_VALIDATION = True
            except ImportError:
                HAS_BNB_VALIDATION = False
                # This is fine - validation is optional
        except ImportError as e:
            print(f"Warning: BitsAndBytes import failed: {e}")
            HAS_BNB = False
            HAS_BNB_VALIDATION = False
except ImportError:
    HAS_BNB = False
    HAS_BNB_VALIDATION = False

# Deferred PEFT import - will be imported only when needed and compatible
HAS_PEFT = False
PEFT_ERROR = None
PEFT_MODULES = {}

def try_import_peft():
    """Try to import PEFT modules safely"""
    global HAS_PEFT, PEFT_ERROR, PEFT_MODULES
    
    if HAS_PEFT:  # Already imported successfully
        return True
    
    if PEFT_ERROR:  # Already failed, don't try again
        return False
    
    try:
        print("Attempting to import PEFT modules...")
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
        
        PEFT_MODULES = {
            'LoraConfig': LoraConfig,
            'get_peft_model': get_peft_model,
            'TaskType': TaskType,
            'PeftModel': PeftModel
        }
        HAS_PEFT = True
        print("+ PEFT imported successfully")
        return True
        
    except Exception as e:
        PEFT_ERROR = str(e)
        HAS_PEFT = False
        print(f"X PEFT import failed: {e}")
        
        # Check if it's a BnB-related error
        if any(term in str(e).lower() for term in ['bitsandbytes', 'bnb', 'impl_abstract', '4bit', 'torch.library']):
            print("  This appears to be a BitsAndBytes compatibility issue")
            print("  Your PyTorch version may be incompatible with BitsAndBytes")
            print("  Will use fallback training without LoRA")
        else:
            print("  This appears to be a different PEFT issue")
            print("  Will use fallback training mode")
        
        return False

import pandas as pd
from jinja2 import Template
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MedicalDataProcessor:
    """Process medical CSV data for LoRA training"""
    
    def __init__(self, template_path: str, classification_topic: str):
        self.classification_topic = classification_topic
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = Template(f.read())
    
    def process_record(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a CSV row to training format"""
        # Create a narrative medical record with clear section headers
        narrative_record = f"""PATIENT DEMOGRAPHICS:
Sex: {row.get('Sex', 'N/A')}
Age: {row.get('Age', 'N/A')}
Ethnicity: {row.get('c_ethnicity', 'N/A')}
Race: {row.get('c_race', 'N/A')}

CHIEF COMPLAINT:
{row.get('ChiefComplaintOrig', 'Not documented')}

ADMISSION REASON:
{row.get('Admit_Reason_Combo', 'Not documented')}

TRIAGE NOTES:
{row.get('TriageNotes', row.get('TriageNotesOrig', 'Not documented'))}

DIAGNOSIS INFORMATION:
Discharge Diagnosis: {row.get('DischargeDiagnosis', 'Not documented')}
Diagnosis Codes: {row.get('Diagnosis_Combo', 'Not documented')}
CCDD Category: {row.get('CCDD Category', 'Not documented')}"""
        
        # Prepare the medical record content
        content = {
            'narrative_record': narrative_record,
            'chief_complaint': row.get('ChiefComplaintOrig', ''),
            'discharge_diagnosis': row.get('DischargeDiagnosis', ''),
            'demographics': f"Sex: {row.get('Sex', '')}, Age: {row.get('Age', '')}, Ethnicity: {row.get('c_ethnicity', '')}, Race: {row.get('c_race', '')}",
            'admit_reason': row.get('Admit_Reason_Combo', ''),
            'diagnosis_combo': row.get('Diagnosis_Combo', ''),
            'ccdd_category': row.get('CCDD Category', ''),
            'triage_notes': row.get('TriageNotes', row.get('TriageNotesOrig', ''))
        }
        
        # Create messages for the template
        messages = [
            {
                'role': 'user',
                'content': content
            },
            {
                'role': 'assistant',
                'content': {
                    'classification': row.get('Expert Rating', ''),
                    'confidence': 'Confident',  # Default confidence
                    'rationale': row.get('Rationale_of_Rating', '')
                }
            }
        ]
        
        # Render the template
        formatted_text = self.template.render(
            messages=messages,
            classification_topic=self.classification_topic
        )
        
        return {
            'text': formatted_text,
            'biosense_id': row.get('C_Biosense_ID', ''),
            'expert_rating': row.get('Expert Rating', ''),
            'rationale': row.get('Rationale_of_Rating', '')
        }
    
    def process_dataset(self, csv_path: str) -> Dataset:
        """Process entire CSV dataset"""
        df = pd.read_csv(csv_path)
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append(self.process_record(row))
        
        return Dataset.from_list(processed_data)


class LoRATrainer:
    """Main training class for LoRA fine-tuning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_model_and_tokenizer(self, model_name: str):
        """Initialize model and tokenizer"""
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config for QLoRA - with improved compatibility handling
        quantization_config = None
        if (self.config['tuning']['mode'] == 'qlora' and 
            self.config.get('quantization', {}).get('enabled', True) and 
            HAS_BNB):
            
            print("Setting up QLoRA quantization...")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.config['quantization']['load_in_4bit'],
                    bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype']),
                    bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
                    bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant']
                )
                print("[OK] QLoRA quantization configured successfully")
            except Exception as e:
                print(f"Warning: Failed to setup QLoRA quantization: {e}")
                print("Falling back to standard LoRA...")
                quantization_config = None
                
        elif self.config['tuning']['mode'] == 'qlora' and not HAS_BNB:
            print("Warning: QLoRA requested but BitsAndBytes not available. Using standard LoRA.")
            # Force standard LoRA mode
            self.config['tuning']['mode'] = 'lora'
        elif not self.config.get('quantization', {}).get('enabled', True):
            print("Quantization disabled in config. Using standard LoRA.")
            
        # If BnB not available, ensure we're in standard LoRA mode
        if not HAS_BNB and self.config['tuning']['mode'] == 'qlora':
            print("Forcing standard LoRA mode due to BitsAndBytes compatibility issues")
            self.config['tuning']['mode'] = 'lora'
        
        # Load model
        # Use float32 for CPU, float16 for GPU if requested
        if torch.cuda.is_available() and self.config['train']['fp16']:
            dtype = torch.float16
        elif torch.cuda.is_available() and self.config['train'].get('bf16', False):
            dtype = torch.bfloat16
        else:
            dtype = torch.float32  # CPU fallback
            
        model_kwargs = {
            'trust_remote_code': self.config['model']['trust_remote_code'],
            'torch_dtype': dtype,
        }
        
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
            model_kwargs['device_map'] = 'auto'
        
        # Load model with error handling for BnB compatibility issues
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            print("[OK] Model loaded successfully")
        except Exception as e:
            error_str = str(e).lower()
            bnb_errors = [
                "validate_bnb_backend_availability", 
                "bitsandbytes", 
                "bnb", 
                "4bit",
                "quantization_config"
            ]
            
            if quantization_config and any(error in error_str for error in bnb_errors):
                print(f"BnB/Quantization error detected: {e}")
                print("Retrying without quantization...")
                # Remove quantization and try again
                model_kwargs_fallback = {k: v for k, v in model_kwargs.items() 
                                       if k not in ['quantization_config', 'device_map']}
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs_fallback)
                print("[OK] Model loaded without quantization (standard LoRA mode)")
                # Update mode to standard LoRA since quantization failed
                self.config['tuning']['mode'] = 'lora'
            else:
                # Re-raise other errors
                raise e
        
        # Setup LoRA (if PEFT is available and compatible)
        if self.config['tuning']['mode'] in ['qlora', 'lora']:
            if try_import_peft():
                print("Setting up LoRA with PEFT...")
                try:
                    LoraConfig = PEFT_MODULES['LoraConfig']
                    get_peft_model = PEFT_MODULES['get_peft_model']
                    TaskType = PEFT_MODULES['TaskType']
                    
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=self.config['tuning']['lora_r'],
                        lora_alpha=self.config['tuning']['lora_alpha'],
                        lora_dropout=self.config['tuning']['lora_dropout'],
                        target_modules=self.config['tuning']['target_modules'],
                        bias=self.config['tuning']['bias']
                    )
                    self.model = get_peft_model(self.model, peft_config)
                    self.model.print_trainable_parameters()
                    print("[OK] LoRA setup completed successfully")
                    
                except Exception as e:
                    print(f"Warning: LoRA setup failed: {e}")
                    print("Falling back to full fine-tuning...")
                    self.config['tuning']['mode'] = 'full'
            else:
                print("PEFT not available - using full fine-tuning fallback")
                self.config['tuning']['mode'] = 'full'
        
        # For full fine-tuning mode, just unfreeze all parameters
        if self.config['tuning']['mode'] == 'full':
            print("Using full fine-tuning mode (all parameters trainable)")
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Count trainable parameters manually
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable params: {trainable_params:,} || All params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}")
            print("Warning: Full fine-tuning uses more memory and may be slower")
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset"""
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config['model']['max_seq_len'],
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def split_dataset(self, dataset: Dataset) -> DatasetDict:
        """Split dataset into train/val/test"""
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        # First split: train vs (val + test)
        train_val_split = dataset.train_test_split(
            test_size=1 - train_split,
            seed=self.config['data']['random_seed'],
            shuffle=self.config['data']['shuffle']
        )
        
        # Second split: val vs test
        val_test_split = train_val_split['test'].train_test_split(
            test_size=self.config['data']['test_split'] / (val_split + self.config['data']['test_split']),
            seed=self.config['data']['random_seed'],
            shuffle=self.config['data']['shuffle']
        )
        
        return DatasetDict({
            'train': train_val_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        
        # For now, just return loss-based metrics
        # TODO: Add classification-specific metrics
        return {}
    
    def train(self, dataset: Dataset, output_dir: str, model_name: str = None, ollama_name: str = None) -> str:
        """Main training function"""
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Split dataset
        dataset_splits = self.split_dataset(tokenized_dataset)
        
        # Setup training arguments with version compatibility
        training_kwargs = {
            'output_dir': output_dir,
            'learning_rate': self.config['train']['learning_rate'],
            'num_train_epochs': self.config['train']['num_epochs'],
            'per_device_train_batch_size': 1 if self.config['train']['batch_size'] == 'auto' else self.config['train']['batch_size'],
            'per_device_eval_batch_size': 1,
            'gradient_accumulation_steps': 8 if self.config['train']['gradient_accumulation_steps'] == 'auto' else self.config['train']['gradient_accumulation_steps'],
            'warmup_ratio': self.config['train']['warmup_ratio'],
            'weight_decay': self.config['train']['weight_decay'],
            'logging_steps': self.config['train']['logging_steps'],
            'eval_strategy': self.config['train']['evaluation_strategy'],
            'eval_steps': self.config['train']['eval_steps'],
            'save_strategy': self.config['train']['save_strategy'],
            'save_steps': self.config['train']['save_steps'],
            'load_best_model_at_end': self.config['train']['load_best_model_at_end'],
            'metric_for_best_model': self.config['train']['metric_for_best_model'],
            'greater_is_better': self.config['train']['greater_is_better'],
            'fp16': self.config['train']['fp16'],
            'bf16': self.config['train']['bf16'],
            'dataloader_num_workers': self.config['train']['dataloader_num_workers'],
            'save_total_limit': self.config['train']['save_total_limit'],
            'remove_unused_columns': False,
        }
        
        # Add report_to only if supported (newer transformers versions)
        try:
            from transformers import __version__ as transformers_version
            import packaging.version
            if packaging.version.parse(transformers_version) >= packaging.version.parse("4.21.0"):
                training_kwargs['report_to'] = self.config['train']['report_to']
        except (ImportError, AttributeError):
            # Fallback if version check fails
            pass
        
        training_args = TrainingArguments(**training_kwargs)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_splits['train'],
            eval_dataset=dataset_splits['validation'],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save the model and LoRA adapter
        print("Saving model and LoRA adapter...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save LoRA adapter separately if using LoRA/QLoRA
        if self.config['tuning']['mode'] in ['qlora', 'lora'] and HAS_PEFT:
            try:
                from peft import PeftModel
                # Save the LoRA adapter
                lora_output_dir = os.path.join(output_dir, 'lora_adapter')
                self.model.save_pretrained(lora_output_dir)
                print(f"LoRA adapter saved to: {lora_output_dir}")
                
                # Create Ollama-compatible model if requested
                if ollama_name:
                    self._create_ollama_model(ollama_name, output_dir, lora_output_dir)
                    
            except Exception as e:
                print(f"Warning: Could not save LoRA adapter separately: {e}")
        
        # Save model metadata
        model_metadata = {
            'model_name': model_name or 'epituner_medical_lora',
            'ollama_name': ollama_name or model_name or 'epituner_medical_lora',
            'training_mode': self.config['tuning']['mode'],
            'base_model': self.config['model']['name'],
            'classification_topic': self.config.get('classification_topic', ''),
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
        }
        
        # Evaluate on test set
        if len(dataset_splits['test']) > 0:
            eval_result = self.trainer.evaluate(dataset_splits['test'])
            model_metadata.update({f'test_{k}': v for k, v in eval_result.items()})
        
        # Save metadata
        with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"Training completed. Model saved to: {output_dir}")
        print(f"Model name: {model_metadata['model_name']}")
        if ollama_name:
            print(f"Ollama model name: {ollama_name}")
        
        return output_dir
    
    def _create_ollama_model(self, ollama_name: str, output_dir: str, lora_adapter_dir: str):
        """Create Ollama-compatible model files"""
        try:
            # Create Ollama Modelfile
            modelfile_path = os.path.join(output_dir, 'Modelfile')
            
            # Get base model name for Ollama
            base_model = self.config['model']['name']
            if '/' in base_model:
                base_model = base_model.split('/')[-1]  # Extract model name from path
            
            modelfile_content = f"""FROM {base_model}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"

# Medical classification LoRA adapter
ADAPTER {lora_adapter_dir}

# System prompt for medical classification
SYSTEM "You are an expert medical coder trained to classify medical records. You analyze patient data and provide accurate classifications based on the specified criteria."
"""
            
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            print(f"Ollama Modelfile created: {modelfile_path}")
            print(f"To use with Ollama, run: ollama create {ollama_name} -f {modelfile_path}")
            
        except Exception as e:
            print(f"Warning: Could not create Ollama model files: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA model for medical classification")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--topic", type=str, required=True, help="Classification topic description")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--template", type=str, default="chat_templates/medical_classification.jinja", help="Chat template path")
    parser.add_argument("--model-name", type=str, help="Custom name for the trained model/LoRA adapter")
    parser.add_argument("--ollama-name", type=str, help="Name for Ollama model (if different from model-name)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override model name with user selection
    config['model']['name'] = args.model
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process data
    print("Processing medical data...")
    processor = MedicalDataProcessor(args.template, args.topic)
    dataset = processor.process_dataset(args.data)
    
    print(f"Processed {len(dataset)} records")
    
    # Initialize trainer
    trainer = LoRATrainer(config)
    trainer.setup_model_and_tokenizer(args.model)
    
    # Train
    model_path = trainer.train(dataset, args.output, args.model_name, args.ollama_name)
    
    print(f"Training completed successfully! Model saved to: {model_path}")


if __name__ == "__main__":
    main()
