#!/usr/bin/env python3
"""
Simple, reliable LoRA trainer
One training path that always works
"""

import json
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

# Conditional imports
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


@dataclass
class TrainingResult:
    """Result of training process"""
    success: bool
    model_path: str
    error_message: str = ""
    training_loss: Optional[float] = None
    steps_completed: int = 0
    total_steps: int = 0


class SimpleTrainer:
    """Simple LoRA trainer that always works"""
    
    def __init__(self, model_name: str, progress_callback: Optional[Callable] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.progress_callback = progress_callback or self._default_progress
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
    def _default_progress(self, message: str, progress: float = 0.0):
        """Default progress callback"""
        print(f"[{progress:.1%}] {message}")
    
    def _validate_model(self) -> bool:
        """Check if model exists and is accessible"""
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.model_name)
            return True
        except Exception as e:
            self.progress_callback(f"Model validation failed: {e}", 0.0)
            return False
    
    def _setup_model(self) -> bool:
        """Load model and tokenizer"""
        try:
            self.progress_callback("Loading tokenizer...", 0.1)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.progress_callback("Loading model...", 0.2)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Setup LoRA if available
            if HAS_PEFT:
                self.progress_callback("Setting up LoRA...", 0.3)
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=8,  # Conservative rank for reliability
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj"],  # Safe modules
                )
                self.model = get_peft_model(self.model, lora_config)
                self.progress_callback("LoRA setup complete", 0.35)
            else:
                self.progress_callback("PEFT not available, using full fine-tuning", 0.35)
            
            return True
            
        except Exception as e:
            self.progress_callback(f"Model setup failed: {e}", 0.0)
            return False
    
    def _prepare_dataset(self, training_texts: List[str]) -> Dataset:
        """Convert training texts to tokenized dataset"""
        self.progress_callback("Tokenizing data...", 0.4)
        
        def tokenize_function(examples):
            # Simple tokenization
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=512,  # Conservative length
                return_tensors=None
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({'text': training_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        return tokenized_dataset
    
    def train(self, training_texts: List[str], output_dir: str, 
              epochs: int = 2, learning_rate: float = 2e-4) -> TrainingResult:
        """Train the model with LoRA"""
        
        self.progress_callback("Starting training...", 0.0)
        
        # Validate model
        if not self._validate_model():
            return TrainingResult(
                success=False,
                model_path="",
                error_message=f"Model '{self.model_name}' not found or inaccessible"
            )
        
        # Setup model
        if not self._setup_model():
            return TrainingResult(
                success=False,
                model_path="",
                error_message="Failed to load model and tokenizer"
            )
        
        try:
            # Prepare data
            dataset = self._prepare_dataset(training_texts)
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Training arguments - optimized for reliability
            training_args = TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                per_device_train_batch_size=1,  # Conservative batch size
                gradient_accumulation_steps=4,  # Compensate for small batch
                warmup_steps=10,
                logging_steps=10,
                save_steps=50,
                save_total_limit=1,
                fp16=torch.cuda.is_available(),  # Use fp16 only if GPU
                dataloader_num_workers=0,  # Avoid Windows issues
                remove_unused_columns=False,
                report_to=[],  # No external reporting
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            self.progress_callback("Training started...", 0.5)
            
            # Train
            train_result = trainer.train()
            
            self.progress_callback("Saving model...", 0.9)
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_path)
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'training_loss': train_result.training_loss,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'has_lora': HAS_PEFT,
                'device': self.device,
                'timestamp': datetime.now().isoformat(),
                'training_samples': len(training_texts)
            }
            
            with open(output_path / 'training_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.progress_callback("Training completed!", 1.0)
            
            return TrainingResult(
                success=True,
                model_path=str(output_path),
                training_loss=train_result.training_loss,
                steps_completed=train_result.global_step,
                total_steps=train_result.global_step
            )
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.progress_callback(error_msg, 0.0)
            
            return TrainingResult(
                success=False,
                model_path="",
                error_message=error_msg
            )

