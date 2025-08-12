#!/usr/bin/env python3
"""
EpiTuner Inference Script
Handle model predictions with confidence scoring
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

import torch
import yaml
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
import numpy as np
from jinja2 import Template
from scipy.special import softmax
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ConfidenceCalculator:
    """Calculate confidence scores for model predictions"""
    
    @staticmethod
    def calculate_token_confidence(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
        """Calculate confidence based on token probability distribution"""
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probabilities for the actual generated tokens
        token_probs = []
        for i, token_id in enumerate(token_ids):
            if i < logits.shape[0]:
                token_prob = probs[i, token_id].item()
                token_probs.append(token_prob)
        
        if not token_probs:
            return 0.0
        
        # Use geometric mean of token probabilities
        log_prob_sum = sum(np.log(max(p, 1e-10)) for p in token_probs)
        geometric_mean = np.exp(log_prob_sum / len(token_probs))
        
        return float(geometric_mean)
    
    @staticmethod
    def calculate_entropy_confidence(logits: torch.Tensor) -> float:
        """Calculate confidence based on entropy of probability distribution"""
        probs = F.softmax(logits, dim=-1)
        
        # Calculate entropy for each position
        entropies = []
        for i in range(logits.shape[0]):
            prob_dist = probs[i]
            entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10))
            entropies.append(entropy.item())
        
        if not entropies:
            return 0.0
        
        # Average entropy (lower entropy = higher confidence)
        avg_entropy = np.mean(entropies)
        max_entropy = np.log(logits.shape[-1])  # Maximum possible entropy
        
        # Convert to confidence score (0-1 scale)
        confidence = 1.0 - (avg_entropy / max_entropy)
        return max(0.0, min(1.0, confidence))
    
    @staticmethod
    def calculate_top_k_confidence(logits: torch.Tensor, k: int = 5) -> float:
        """Calculate confidence based on top-k probability mass"""
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k probabilities for each position
        top_k_masses = []
        for i in range(logits.shape[0]):
            top_k_probs, _ = torch.topk(probs[i], k)
            top_k_mass = torch.sum(top_k_probs).item()
            top_k_masses.append(top_k_mass)
        
        if not top_k_masses:
            return 0.0
        
        return float(np.mean(top_k_masses))
    
    @classmethod
    def calculate_combined_confidence(cls, logits: torch.Tensor, token_ids: torch.Tensor) -> float:
        """Calculate combined confidence score using multiple methods"""
        if logits is None or logits.shape[0] == 0:
            return 0.5  # Default moderate confidence
        
        # Calculate different confidence measures
        token_conf = cls.calculate_token_confidence(logits, token_ids)
        entropy_conf = cls.calculate_entropy_confidence(logits)
        top_k_conf = cls.calculate_top_k_confidence(logits)
        
        # Weighted combination
        weights = [0.4, 0.3, 0.3]  # Token, entropy, top-k
        combined = (weights[0] * token_conf + 
                   weights[1] * entropy_conf + 
                   weights[2] * top_k_conf)
        
        return float(combined)
    
    @staticmethod
    def confidence_to_level(confidence: float) -> str:
        """Convert numeric confidence to categorical level"""
        if confidence >= 0.85:
            return "Very Confident"
        elif confidence >= 0.70:
            return "Confident"
        elif confidence >= 0.55:
            return "Somewhat Confident"
        elif confidence >= 0.40:
            return "Not Very Confident"
        else:
            return "Not at all Confident"


class SyndromicSurveillanceClassificationInference:
    """Inference engine for medical classification with confidence scoring"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tokenizer = None
        self.model = None
        self.template = None
        
        self._load_model()
        self._load_template()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        # Check if model directory exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        # Get base model name from config or infer from adapter
        base_model_name = self.config.get('model', {}).get('name', 'microsoft/DialoGPT-medium')
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=self.config.get('model', {}).get('trust_remote_code', False)
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            # Try loading from model directory
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=self.config.get('model', {}).get('trust_remote_code', False)
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e2:
                raise Exception(f"Failed to load tokenizer from both base model and model directory: {e2}")
        
        # Setup quantization for inference if needed
        quantization_config = None
        if self.config.get('tuning', {}).get('mode') == 'qlora':
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except Exception as e:
                print(f"Warning: Could not setup quantization: {e}")
                quantization_config = None
        
        # Load base model
        model_kwargs = {
            'trust_remote_code': self.config.get('model', {}).get('trust_remote_code', False),
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
            model_kwargs['device_map'] = 'auto'
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        except Exception as e:
            print(f"Error loading base model: {e}")
            # Try loading from model directory
            try:
                base_model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **model_kwargs)
            except Exception as e2:
                raise Exception(f"Failed to load model from both base model and model directory: {e2}")
        
        # Load LoRA adapter if it exists
        adapter_path = self.model_path
        if (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists():
            print("Loading LoRA adapter...")
            try:
                self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            except Exception as e:
                print(f"Warning: Could not load LoRA adapter: {e}")
                print("Using base model without adapter...")
                self.model = base_model
        else:
            print("No LoRA adapter found, using base model")
            self.model = base_model
        
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        else:
            print("No GPU available, using CPU")
    
    def _load_template(self):
        """Load the chat template"""
        template_path = Path("chat_templates/medical_classification.jinja")
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                self.template = Template(f.read())
        else:
            # Fallback template
            self.template = Template("""
            Syndromic Surveillance Record Classification Task

            Classification Criteria: {{ classification_topic }}

            Syndromic Surveillance Record:
            - Chief Complaint: {{ chief_complaint }}
            - Discharge Diagnosis: {{ discharge_diagnosis }}
            - Demographics: {{ demographics }}
            - Triage Notes: {{ triage_notes }}

            Classification: [Match/Not a Match/Unknown]
            Confidence: [Very Confident/Confident/Somewhat Confident/Not Very Confident/Not at all Confident]
            Rationale: [Explanation]
            """)
    
    def format_record(self, record: Dict[str, Any], classification_topic: str) -> str:
        """Format a syndromic surveillance record for inference"""
        # Prepare the content
        content = {
            'chief_complaint': record.get('chief_complaint', record.get('ChiefComplaintOrig', '')),
            'discharge_diagnosis': record.get('discharge_diagnosis', record.get('DischargeDiagnosis', '')),
            'demographics': record.get('demographics', f"Sex: {record.get('Sex', '')}, Age: {record.get('Age', '')}, Ethnicity: {record.get('c_ethnicity', '')}, Race: {record.get('c_race', '')}"),
            'admit_reason': record.get('admit_reason', record.get('Admit_Reason_Combo', '')),
            'diagnosis_combo': record.get('diagnosis_combo', record.get('Diagnosis_Combo', '')),
            'ccdd_category': record.get('ccdd_category', record.get('CCDD Category', '')),
            'triage_notes': record.get('triage_notes', record.get('TriageNotes', record.get('TriageNotesOrig', '')))
        }
        
        # Create user message
        messages = [{
            'role': 'user',
            'content': content
        }]
        
        # Render template
        formatted_text = self.template.render(
            messages=messages,
            classification_topic=classification_topic
        )
        
        return formatted_text
    
    def predict_single(self, record: Dict[str, Any], classification_topic: str) -> Dict[str, Any]:
        """Make prediction for a single syndromic surveillance record"""
        
        try:
            # Format the input
            input_text = self.format_record(record, classification_topic)
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.get('model', {}).get('max_seq_len', 512),
                padding=True
            )
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('generation', {}).get('max_new_tokens', 256),
                    temperature=self.config.get('generation', {}).get('temperature', 0.7),
                    top_p=self.config.get('generation', {}).get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode the generated text
            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Calculate confidence
            if hasattr(outputs, 'scores') and outputs.scores:
                # Stack all logits
                all_logits = torch.stack(outputs.scores, dim=0)  # [seq_len, batch_size, vocab_size]
                all_logits = all_logits[:, 0, :]  # Remove batch dimension
                
                confidence_score = ConfidenceCalculator.calculate_combined_confidence(
                    all_logits, generated_ids
                )
            else:
                confidence_score = 0.5  # Default if no scores available
            
            # Parse the generated response
            parsed_response = self._parse_response(generated_text)
            
            # Add confidence information
            parsed_response['confidence_score'] = confidence_score
            parsed_response['confidence_level'] = ConfidenceCalculator.confidence_to_level(confidence_score)
            
            return parsed_response
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return a safe fallback response
            return {
                'classification': 'Unknown/Not able to determine',
                'confidence': 'Not at all Confident',
                'confidence_score': 0.0,
                'confidence_level': 'Not at all Confident',
                'rationale': f'Error during prediction: {str(e)}',
                'raw_response': '',
                'error': str(e)
            }
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the model's response to extract classification, confidence, and rationale"""
        
        # Initialize defaults
        classification = "Unknown"
        confidence = "Somewhat Confident"
        rationale = response_text.strip()
        
        # Try to extract structured information
        # Look for patterns like "Classification: Match"
        classification_match = re.search(r'Classification:\s*([^\n]+)', response_text, re.IGNORECASE)
        if classification_match:
            classification = classification_match.group(1).strip()
        
        # Look for confidence pattern
        confidence_match = re.search(r'Confidence:\s*([^\n]+)', response_text, re.IGNORECASE)
        if confidence_match:
            confidence = confidence_match.group(1).strip()
        
        # Look for rationale pattern
        rationale_match = re.search(r'Rationale:\s*([^\n]+(?:\n[^\n]+)*)', response_text, re.IGNORECASE)
        if rationale_match:
            rationale = rationale_match.group(1).strip()
        
        # Clean up classification
        classification = self._normalize_classification(classification)
        
        return {
            'classification': classification,
            'confidence': confidence,
            'rationale': rationale,
            'raw_response': response_text
        }
    
    def _normalize_classification(self, classification: str) -> str:
        """Normalize classification to standard values"""
        classification = classification.lower().strip()
        
        if 'match' in classification and 'not' not in classification:
            return "Match"
        elif 'not' in classification and 'match' in classification:
            return "Not a Match"
        elif 'unknown' in classification or 'unable' in classification:
            return "Unknown/Not able to determine"
        else:
            return "Unknown/Not able to determine"
    
    def predict_batch(self, records: List[Dict[str, Any]], classification_topic: str) -> List[Dict[str, Any]]:
        """Make predictions for a batch of syndromic surveillance records"""
        results = []
        
        for i, record in enumerate(records):
            print(f"Processing record {i+1}/{len(records)}: {record.get('C_Biosense_ID', 'Unknown')}")
            
            try:
                prediction = self.predict_single(record, classification_topic)
                prediction['biosense_id'] = record.get('C_Biosense_ID', f'record_{i}')
                results.append(prediction)
            except Exception as e:
                print(f"Error processing record {i+1}: {str(e)}")
                # Add error result
                results.append({
                    'biosense_id': record.get('C_Biosense_ID', f'record_{i}'),
                    'classification': 'Unknown/Not able to determine',
                    'confidence': 'Not at all Confident',
                    'confidence_score': 0.0,
                    'confidence_level': 'Not at all Confident',
                    'rationale': f'Error during processing: {str(e)}',
                    'raw_response': '',
                    'error': str(e)
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Run inference on syndromic surveillance records")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--topic", type=str, required=True, help="Classification topic")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    records = df.to_dict('records')
    
    # Initialize inference engine
    inference_engine = SyndromicSurveillanceClassificationInference(args.model, args.config)
    
    # Run inference
    print(f"Running inference on {len(records)} records...")
    results = inference_engine.predict_batch(records, args.topic)
    
    # Save results
    output_data = {
        'classification_topic': args.topic,
        'total_records': len(records),
        'predictions': results,
        'model_path': args.model,
        'config_path': args.config
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {args.output}")
    
    # Print summary
    classifications = [r['classification'] for r in results]
    confidence_levels = [r['confidence_level'] for r in results]
    
    print("\nSummary:")
    print(f"Match: {classifications.count('Match')}")
    print(f"Not a Match: {classifications.count('Not a Match')}")
    print(f"Unknown: {classifications.count('Unknown/Not able to determine')}")
    
    print(f"\nConfidence Distribution:")
    for level in ["Very Confident", "Confident", "Somewhat Confident", "Not Very Confident", "Not at all Confident"]:
        count = confidence_levels.count(level)
        print(f"{level}: {count}")


if __name__ == "__main__":
    main()
