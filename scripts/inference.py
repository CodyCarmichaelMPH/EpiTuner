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

# Import new modules for scoring and rationale
from epituner.inference_scoring import score_labels, pick_label
from epituner.rationale import extract_evidence, rationale_prompt, create_fallback_rationale


def load_temperature() -> float:
    """Load temperature from calibration.json, default to 1.0"""
    try:
        calibration_path = project_root / 'calibration.json'
        if calibration_path.exists():
            with open(calibration_path, 'r') as f:
                config = json.load(f)
                return config.get('temperature', 1.0)
    except Exception:
        pass
    return 1.0


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
        
        # Load LoRA adapter if it exists (exactly one adapter active)
        adapter_path = self.model_path
        if (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists():
            print("Loading LoRA adapter...")
            try:
                self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
                # Optional: merge adapter for deployment efficiency
                # self.model = self.model.merge_and_unload()
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
            # Fallback template - matches medical_classification.jinja structure
            self.template = Template("""{% for message in messages %}{% if message['role'] == 'user' %}**Medical Record Classification Task**

You are an expert medical coder tasked with classifying medical records. Your goal is to determine if the medical record matches the specified criteria.

**Classification Criteria:** {{ classification_topic }}

**Medical Record:**
- **Chief Complaint:** {{ message.content.chief_complaint }}
- **Discharge Diagnosis:** {{ message.content.discharge_diagnosis }}
- **Demographics:** {{ message.content.demographics }}
- **Triage Notes:** {{ message.content.triage_notes }}

Please provide your classification in the following format:
1. **Classification:** [Match/Not a Match/Unknown]
2. **Confidence:** [Very Confident/Confident/Somewhat Confident/Not Very Confident/Not at all Confident]
3. **Rationale:** [Provide a detailed, evidence-based explanation that includes:
   - Key medical indicators from the record that support your classification
   - Specific symptoms, diagnoses, or demographic factors that align with the criteria
   - Any conflicting or ambiguous information that influenced your decision
   - Clinical reasoning for why this case does or does not meet the classification criteria
   - Reference to specific sections of the medical record]

{% endif %}{% endfor %}""")
    
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
            # Format the input for classification
            input_text = self.format_record(record, classification_topic)
            
            # Build classification prompt suffix
            prompt_text = input_text + "\nAnswer (choose exactly one): "
            
            # Load temperature from calibration
            temperature = load_temperature()
            
            # Score the labels using the new approach
            scores = score_labels(self.model, self.tokenizer, prompt_text, labels=("Match", "Not a Match"))
            print(f"Debug: Raw scores = {scores}")
            
            label, confidence, unknown = pick_label(scores, tau=0.15, temperature=temperature)
            print(f"Debug: Label = {label}, Confidence = {confidence}, Unknown = {unknown}")
            
            if unknown:
                label = "Unknown/Not able to determine"
            
            # Generate rationale using quoted evidence
            evidence = extract_evidence(record)
            rationale_label = label if label != "Unknown/Not able to determine" else "Unknown"
            
            # Try to generate rationale with model
            rationale = None
            try:
                prompt = rationale_prompt(evidence, rationale_label)
                rationale_inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    rationale_inputs = {k: v.cuda() for k, v in rationale_inputs.items()}
                
                with torch.no_grad():
                    rationale_outputs = self.model.generate(
                        **rationale_inputs,
                        max_new_tokens=100,
                        do_sample=False,  # deterministic
                        temperature=0.1,  # Very low temperature for focused output
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                rationale_text = self.tokenizer.decode(
                    rationale_outputs[0][rationale_inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Check if the generated text looks like a proper rationale (not instructions)
                if (rationale_text and 
                    len(rationale_text) > 20 and 
                    not rationale_text.lower().startswith(('you will', 'write', 'facts you may', 'paraphrase'))):
                    rationale = rationale_text
                    
            except Exception as e:
                print(f"Warning: Rationale generation failed: {e}")
            
            # Use fallback if generation failed or returned instructions
            if not rationale:
                rationale = create_fallback_rationale(evidence, rationale_label)
            
            # Convert confidence to percentage and level
            confidence_score = confidence
            confidence_percentage = round(confidence * 100, 1)
            confidence_level = ConfidenceCalculator.confidence_to_level(confidence_score)
            
            return {
                'classification': label,
                'confidence': confidence_level,
                'confidence_score': confidence_score,
                'confidence_level': confidence_level,
                'confidence_percentage': confidence_percentage,
                'rationale': rationale,
                'raw_response': rationale_text,
                'scores': scores
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return a safe fallback response
            return {
                'classification': 'Unknown/Not able to determine',
                'confidence': 'Not at all Confident',
                'confidence_score': 0.0,
                'confidence_level': 'Not at all Confident',
                'confidence_percentage': 0.0,
                'rationale': f'Error during prediction: {str(e)}',
                'raw_response': '',
                'error': str(e)
            }
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the model's response to extract classification, confidence, and rationale"""
        
        # Initialize defaults
        classification = "Unknown/Not able to determine"
        confidence = "Somewhat Confident"
        rationale = "No detailed rationale provided by the model."
        
        # Try to extract structured information with multiple pattern variations
        # Look for classification patterns (more flexible)
        classification_patterns = [
            r'(?:Classification|1\.\s*\*?\*?Classification\*?\*?):\s*(.+?)(?:\n|\*|$)',
            r'(?:^|\n)\s*(?:Classification|Answer):\s*(.+?)(?:\n|$)',
            r'(?:^|\n)\s*(.+?)(?:\s*matches?|\s*does\s+not\s+match|\s*unknown)',
        ]
        
        for pattern in classification_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                classification = match.group(1).strip().rstrip('*').strip()
                break
        
        # Look for confidence patterns (more flexible)
        confidence_patterns = [
            r'(?:Confidence|2\.\s*\*?\*?Confidence\*?\*?):\s*(.+?)(?:\n|\*|$)',
            r'(?:^|\n)\s*(?:Confidence|Certainty):\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                confidence = match.group(1).strip().rstrip('*').strip()
                break
        
        # Look for rationale patterns (more flexible and comprehensive)
        rationale_patterns = [
            r'(?:Rationale|3\.\s*\*?\*?Rationale\*?\*?):\s*(.*?)(?:\n\n|\n\s*(?:\d+\.|\*\*|$))',
            r'(?:Explanation|Reasoning):\s*(.*?)(?:\n\n|\n\s*(?:\d+\.|\*\*|$))',
            r'(?:Because|This is because|The reason):\s*(.*?)(?:\n\n|\n\s*(?:\d+\.|\*\*|$))',
        ]
        
        for pattern in rationale_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                extracted_rationale = match.group(1).strip()
                if len(extracted_rationale) > 20:  # Only use if substantial
                    rationale = extracted_rationale
                    break
        
        # If no structured rationale found, try to extract meaningful content
        if rationale == "No detailed rationale provided by the model.":
            # Look for any substantive text after the classification/confidence
            remaining_text = response_text
            # Remove classification and confidence parts
            for term in ['Classification:', 'Confidence:', '1.', '2.', '3.']:
                if term in remaining_text:
                    remaining_text = remaining_text.split(term, 1)[-1]
            
            # Clean and extract meaningful content
            lines = [line.strip() for line in remaining_text.split('\n') if line.strip()]
            meaningful_lines = [line for line in lines if len(line) > 20 and not line.startswith(('*', '-', 'Classification', 'Confidence'))]
            
            if meaningful_lines:
                rationale = ' '.join(meaningful_lines[:3])  # Take first 3 meaningful lines
        
        # Clean up classification
        classification = self._normalize_classification(classification)
        
        # Clean up confidence
        confidence = self._normalize_confidence(confidence)
        
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
        elif 'unknown' in classification or 'unable' in classification or 'determine' in classification:
            return "Unknown/Not able to determine"
        else:
            return "Unknown/Not able to determine"
    
    def _normalize_confidence(self, confidence: str) -> str:
        """Normalize confidence to standard values"""
        confidence = confidence.lower().strip()
        
        if 'very confident' in confidence or 'extremely confident' in confidence:
            return "Very Confident"
        elif 'confident' in confidence and 'not' not in confidence and 'somewhat' not in confidence:
            return "Confident"
        elif 'somewhat' in confidence or 'moderately' in confidence:
            return "Somewhat Confident"
        elif 'not very' in confidence or 'low' in confidence:
            return "Not Very Confident"
        elif 'not at all' in confidence or 'no confidence' in confidence or 'uncertain' in confidence:
            return "Not at all Confident"
        else:
            return "Somewhat Confident"  # Default fallback
    
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
