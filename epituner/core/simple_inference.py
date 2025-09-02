#!/usr/bin/env python3
"""
Simple inference engine for medical classification
One prediction path that always works
"""

import json
import torch
import re
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Conditional imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


@dataclass
class PredictionResult:
    """Result of a single prediction"""
    classification: str
    confidence: str
    rationale: str
    confidence_score: float = 0.5
    error_message: str = ""


class SimpleInference:
    """Simple inference engine for medical classification"""
    
    def __init__(self, model_path: str):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required")
        
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load training metadata if available"""
        metadata_path = self.model_path / 'training_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            # Get base model name
            base_model = self.metadata.get('model_name', 'microsoft/DialoGPT-medium')
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Load LoRA adapter if available
            if HAS_PEFT and (self.model_path / "adapter_model.safetensors").exists():
                print("Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(self.model, str(self.model_path))
            
            self.model.eval()
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict(self, record: Dict[str, Any], topic: str) -> PredictionResult:
        """Make a prediction for a medical record"""
        if self.model is None or self.tokenizer is None:
            return PredictionResult(
                classification="Unknown/Not able to determine",
                confidence="Not at all Confident", 
                rationale="Model not loaded",
                error_message="Model not loaded"
            )
        
        try:
            # Create prediction prompt
            prompt = self._create_prompt(record, topic)
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Parse response
            return self._parse_response(response)
            
        except Exception as e:
            return PredictionResult(
                classification="Unknown/Not able to determine",
                confidence="Not at all Confident",
                rationale=f"Prediction error: {str(e)}",
                error_message=str(e)
            )
    
    def _create_prompt(self, record: Dict[str, Any], topic: str) -> str:
        """Create a simple, reliable prompt"""
        return f"""You are a medical expert. Analyze this syndromic surveillance record for: {topic}

Medical Record:
- Chief Complaint: {record.get('ChiefComplaintOrig', record.get('chief_complaint', 'Not documented'))}
- Discharge Diagnosis: {record.get('DischargeDiagnosis', record.get('discharge_diagnosis', 'Not documented'))}
- Demographics: Sex: {record.get('Sex', 'N/A')}, Age: {record.get('Age', 'N/A')}

Please provide:
1. Classification: Match/Not a Match/Unknown
2. Confidence: Very Confident/Confident/Somewhat Confident/Not Very Confident/Not at all Confident
3. Rationale: Your reasoning

Response:"""
    
    def _parse_response(self, response: str) -> PredictionResult:
        """Parse model response into structured result"""
        # Default values
        classification = "Unknown/Not able to determine"
        confidence = "Somewhat Confident"
        rationale = "Unable to parse model response"
        
        # Try to extract classification
        class_patterns = [
            r'(?:classification|1\.)\s*:?\s*(match|not a match|unknown)',
            r'\b(match|not a match|unknown)\b'
        ]
        
        for pattern in class_patterns:
            match = re.search(pattern, response.lower())
            if match:
                raw_class = match.group(1).lower()
                if 'not' in raw_class and 'match' in raw_class:
                    classification = "Not a Match"
                elif 'match' in raw_class:
                    classification = "Match"
                else:
                    classification = "Unknown/Not able to determine"
                break
        
        # Try to extract confidence
        conf_patterns = [
            r'(?:confidence|2\.)\s*:?\s*(very confident|confident|somewhat confident|not very confident|not at all confident)',
            r'\b(very confident|confident|somewhat confident|not very confident|not at all confident)\b'
        ]
        
        for pattern in conf_patterns:
            match = re.search(pattern, response.lower())
            if match:
                confidence = match.group(1).title()
                break
        
        # Try to extract rationale
        rationale_patterns = [
            r'(?:rationale|reasoning|3\.)\s*:?\s*(.+?)(?:\n\n|$)',
            r'(?:because|reason)\s*:?\s*(.+?)(?:\n\n|$)'
        ]
        
        for pattern in rationale_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                rationale = match.group(1).strip()
                break
        
        # If no structured rationale found, use the whole response
        if rationale == "Unable to parse model response" and len(response) > 10:
            rationale = response.strip()[:200] + "..." if len(response) > 200 else response.strip()
        
        # Calculate confidence score
        confidence_scores = {
            "Very Confident": 0.9,
            "Confident": 0.75,
            "Somewhat Confident": 0.6,
            "Not Very Confident": 0.4,
            "Not At All Confident": 0.2
        }
        confidence_score = confidence_scores.get(confidence, 0.5)
        
        return PredictionResult(
            classification=classification,
            confidence=confidence,
            rationale=rationale,
            confidence_score=confidence_score
        )
    
    def predict_batch(self, records: list, topic: str) -> list:
        """Make predictions for multiple records"""
        results = []
        for record in records:
            result = self.predict(record, topic)
            results.append(result)
        return results

