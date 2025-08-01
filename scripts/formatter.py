"""
Formatter Module for Ollama Fine-Tuning and Evaluation Suite

This module converts cleaned and mapped dataset rows into structured, context-rich prompts
that LLMs can use for both fine-tuning and inference.
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))

from debugging_logger import get_logger, LogLevel


class FormatterError(Exception):
    """Raised when formatting operations fail."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class Formatter:
    """
    Converts dataset rows into structured prompts for LLM training and inference.
    
    Supports:
    - Training prompt construction with few-shot examples
    - Inference prompt construction
    - Batch processing for efficiency
    - Configurable prompt templates
    - Rationale inclusion/exclusion
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the Formatter.
        
        Args:
            debug_mode: Enable verbose logging for debugging
        """
        self.debug_mode = debug_mode
        self.logger = get_logger(debug_mode=debug_mode)
        
        # Default prompt templates
        self.training_template = self._get_default_training_template()
        self.inference_template = self._get_default_inference_template()
        
    def _get_default_training_template(self) -> str:
        """Get the default training prompt template."""
        return """INPUT:
You are evaluating whether this case aligns with topic '{topic}'.

Patient Data:
{context_block}

OUTPUT (Expert Provided):
Rating: {rating}
Rationale: {rationale}"""
    
    def _get_default_inference_template(self) -> str:
        """Get the default inference prompt template."""
        return """INPUT:
You are evaluating whether this case aligns with topic '{topic}'.

Patient Data:
{context_block}

OUTPUT (Model Prediction):
Rating: <predicted integer>
Rationale: <model explanation>"""
    
    def create_context_block(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a Context_Block column from available data columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Context_Block column added
        """
        df_copy = df.copy()
        
        # Define potential context columns in order of preference
        context_columns = [
            'ChiefComplaintOrig', 'Discharge Diagnosis', 'TriageNotes',
            'Admit_Reason_Combo', 'Chief_Complaint_Combo', 'Diagnosis_Combo',
            'CCDD', 'CCDDCategory'
        ]
        
        # Find available context columns
        available_columns = [col for col in context_columns if col in df_copy.columns]
        
        if not available_columns:
            # If no specific context columns, use all non-ID and non-rating columns
            exclude_columns = ['C_BioSense_ID', 'Expert Rating', 'Standardized_Rating', 'Rationale of Rating']
            available_columns = [col for col in df_copy.columns if col not in exclude_columns]
        
        # Create context block by combining available columns
        def create_context(row):
            context_parts = []
            for col in available_columns:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    context_parts.append(f"{col}: {row[col]}")
            return " | ".join(context_parts) if context_parts else "No context available"
        
        df_copy['Context_Block'] = df_copy.apply(create_context, axis=1)
        return df_copy
    
    def validate_input_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input DataFrame for formatting.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_columns = ['C_BioSense_ID']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Create context block if not present
        if 'Context_Block' not in df.columns:
            df = self.create_context_block(df)
        
        # Check for training mode requirements
        if 'Standardized_Rating' in df.columns:
            # Training mode - check for rating column
            if df['Standardized_Rating'].isna().any():
                errors.append("Found missing values in Standardized_Rating column")
        
        # Check for context block content
        if 'Context_Block' in df.columns:
            empty_contexts = df['Context_Block'].isna().sum()
            if empty_contexts > 0:
                errors.append(f"Found {empty_contexts} empty context blocks")
        
        return len(errors) == 0, errors
    
    def create_training_prompts(self, 
                               df: pd.DataFrame,
                               topic: str,
                               include_rationale: bool = True,
                               custom_template: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create training prompts with few-shot examples.
        
        Args:
            df: DataFrame with training data
            topic: Target topic for alignment
            include_rationale: Whether to include expert rationale
            custom_template: Custom prompt template (optional)
            
        Returns:
            List of training prompt dictionaries
        """
        self.logger.log_function_entry("formatter", "create_training_prompts", 
                                     topic=topic, include_rationale=include_rationale)
        
        # Validate input and create context block if needed
        is_valid, errors = self.validate_input_data(df)
        if not is_valid:
            error_msg = f"Input validation failed: {'; '.join(errors)}"
            self.logger.error(error_msg, "formatter", "create_training_prompts")
            raise ValidationError(error_msg)
        
        # Ensure we have a context block
        if 'Context_Block' not in df.columns:
            df = self.create_context_block(df)
        
        # Check for required training columns
        if 'Standardized_Rating' not in df.columns:
            error_msg = "Standardized_Rating column required for training prompts"
            self.logger.error(error_msg, "formatter", "create_training_prompts")
            raise ValidationError(error_msg)
        
        template = custom_template or self.training_template
        prompts = []
        
        for idx, row in df.iterrows():
            try:
                # Extract data
                context_block = row.get('Context_Block', 'Data unavailable')
                rating = row.get('Standardized_Rating')
                rationale = row.get('Rationale of Rating', 'No rationale provided') if include_rationale else ''
                
                # Skip rows with missing ratings
                if pd.isna(rating):
                    self.logger.warning(f"Skipping row {idx} due to missing rating", 
                                      "formatter", "create_training_prompts")
                    continue
                
                # Create prompt
                prompt_text = template.format(
                    topic=topic,
                    context_block=context_block,
                    rating=int(rating),
                    rationale=rationale
                )
                
                prompt_data = {
                    'id': row.get('C_BioSense_ID', f'row_{idx}'),
                    'prompt': prompt_text,
                    'rating': int(rating),
                    'rationale': rationale if include_rationale else None,
                    'topic': topic,
                    'row_index': idx
                }
                
                prompts.append(prompt_data)
                
            except Exception as e:
                self.logger.error(f"Error creating prompt for row {idx}: {e}", 
                                "formatter", "create_training_prompts", error=e)
                continue
        
        self.logger.info(f"Created {len(prompts)} training prompts", 
                        "formatter", "create_training_prompts")
        self.logger.log_function_exit("formatter", "create_training_prompts", len(prompts))
        
        return prompts
    
    def create_inference_prompts(self, 
                                df: pd.DataFrame,
                                topic: str,
                                custom_template: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create inference prompts for model prediction.
        
        Args:
            df: DataFrame with inference data
            topic: Target topic for alignment
            custom_template: Custom prompt template (optional)
            
        Returns:
            List of inference prompt dictionaries
        """
        self.logger.log_function_entry("formatter", "create_inference_prompts", topic=topic)
        
        # Validate input and create context block if needed
        is_valid, errors = self.validate_input_data(df)
        if not is_valid:
            error_msg = f"Input validation failed: {'; '.join(errors)}"
            self.logger.error(error_msg, "formatter", "create_inference_prompts")
            raise ValidationError(error_msg)
        
        # Ensure we have a context block
        if 'Context_Block' not in df.columns:
            df = self.create_context_block(df)
        
        template = custom_template or self.inference_template
        prompts = []
        
        for idx, row in df.iterrows():
            try:
                # Extract data
                context_block = row.get('Context_Block', 'Data unavailable')
                
                # Create prompt
                prompt_text = template.format(
                    topic=topic,
                    context_block=context_block
                )
                
                prompt_data = {
                    'id': row.get('C_BioSense_ID', f'row_{idx}'),
                    'prompt': prompt_text,
                    'topic': topic,
                    'row_index': idx
                }
                
                prompts.append(prompt_data)
                
            except Exception as e:
                self.logger.error(f"Error creating inference prompt for row {idx}: {e}", 
                                "formatter", "create_inference_prompts", error=e)
                continue
        
        self.logger.info(f"Created {len(prompts)} inference prompts", 
                        "formatter", "create_inference_prompts")
        self.logger.log_function_exit("formatter", "create_inference_prompts", len(prompts))
        
        return prompts
    
    def create_mixed_prompts(self, 
                            df: pd.DataFrame,
                            topic: str,
                            include_rationale: bool = True,
                            custom_templates: Optional[Dict[str, str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create both training and inference prompts from the same dataset.
        
        Args:
            df: DataFrame with data
            topic: Target topic for alignment
            include_rationale: Whether to include expert rationale in training prompts
            custom_templates: Custom templates for training and inference
            
        Returns:
            Dictionary with 'training' and 'inference' prompt lists
        """
        self.logger.log_function_entry("formatter", "create_mixed_prompts", topic=topic)
        
        # Split data into training and inference sets
        has_ratings = 'Standardized_Rating' in df.columns
        training_df = df[has_ratings & df['Standardized_Rating'].notna()].copy()
        inference_df = df[~has_ratings | df['Standardized_Rating'].isna()].copy()
        
        self.logger.info(f"Split data: {len(training_df)} training rows, {len(inference_df)} inference rows", 
                        "formatter", "create_mixed_prompts")
        
        # Create prompts
        training_prompts = []
        inference_prompts = []
        
        if len(training_df) > 0:
            training_template = custom_templates.get('training') if custom_templates else None
            training_prompts = self.create_training_prompts(
                training_df, topic, include_rationale, training_template
            )
        
        if len(inference_df) > 0:
            inference_template = custom_templates.get('inference') if custom_templates else None
            inference_prompts = self.create_inference_prompts(
                inference_df, topic, inference_template
            )
        
        result = {
            'training': training_prompts,
            'inference': inference_prompts
        }
        
        self.logger.log_function_exit("formatter", "create_mixed_prompts", result)
        return result
    
    def save_prompts_to_jsonl(self, prompts: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save prompts to JSONL format file.
        
        Args:
            prompts: List of prompt dictionaries
            output_path: Output file path
        """
        self.logger.log_function_entry("formatter", "save_prompts_to_jsonl", output_path=output_path)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for prompt in prompts:
                    f.write(json.dumps(prompt) + '\n')
            
            self.logger.info(f"Saved {len(prompts)} prompts to {output_path}", 
                           "formatter", "save_prompts_to_jsonl")
            
        except Exception as e:
            error_msg = f"Failed to save prompts to {output_path}: {e}"
            self.logger.error(error_msg, "formatter", "save_prompts_to_jsonl", error=e)
            raise FormatterError(error_msg)
        
        self.logger.log_function_exit("formatter", "save_prompts_to_jsonl")
    
    def load_prompts_from_jsonl(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load prompts from JSONL format file.
        
        Args:
            input_path: Input file path
            
        Returns:
            List of prompt dictionaries
        """
        self.logger.log_function_entry("formatter", "load_prompts_from_jsonl", input_path=input_path)
        
        prompts = []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            prompt = json.loads(line)
                            prompts.append(prompt)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {line_num}: {e}", 
                                              "formatter", "load_prompts_from_jsonl")
                            continue
            
            self.logger.info(f"Loaded {len(prompts)} prompts from {input_path}", 
                           "formatter", "load_prompts_from_jsonl")
            
        except Exception as e:
            error_msg = f"Failed to load prompts from {input_path}: {e}"
            self.logger.error(error_msg, "formatter", "load_prompts_from_jsonl", error=e)
            raise FormatterError(error_msg)
        
        self.logger.log_function_exit("formatter", "load_prompts_from_jsonl", len(prompts))
        return prompts
    
    def get_prompt_statistics(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the prompts.
        
        Args:
            prompts: List of prompt dictionaries
            
        Returns:
            Dictionary with prompt statistics
        """
        if not prompts:
            return {
                'total_prompts': 0,
                'avg_prompt_length': 0,
                'rating_distribution': {},
                'topics': set()
            }
        
        # Calculate statistics
        total_prompts = len(prompts)
        prompt_lengths = [len(p.get('prompt', '')) for p in prompts]
        avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
        
        # Rating distribution (for training prompts)
        rating_distribution = {}
        topics = set()
        
        for prompt in prompts:
            topics.add(prompt.get('topic', 'unknown'))
            
            if 'rating' in prompt:
                rating = prompt['rating']
                rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
        
        return {
            'total_prompts': total_prompts,
            'avg_prompt_length': round(avg_prompt_length, 2),
            'min_prompt_length': min(prompt_lengths) if prompt_lengths else 0,
            'max_prompt_length': max(prompt_lengths) if prompt_lengths else 0,
            'rating_distribution': rating_distribution,
            'topics': list(topics)
        }
    
    def validate_prompt_format(self, prompt: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single prompt format.
        
        Args:
            prompt: Prompt dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        required_fields = ['id', 'prompt', 'topic']
        for field in required_fields:
            if field not in prompt:
                errors.append(f"Missing required field: {field}")
        
        # Check prompt content
        if 'prompt' in prompt and not prompt['prompt'].strip():
            errors.append("Prompt text is empty")
        
        # Check for training-specific fields
        if 'rating' in prompt:
            try:
                rating = int(prompt['rating'])
                if rating < 0:
                    errors.append("Rating must be non-negative")
            except (ValueError, TypeError):
                errors.append("Rating must be a valid integer")
        
        return len(errors) == 0, errors
    
    def batch_process(self, 
                     df: pd.DataFrame,
                     topic: str,
                     batch_size: int = 100,
                     include_rationale: bool = True) -> List[Dict[str, Any]]:
        """
        Process dataset in batches for memory efficiency.
        
        Args:
            df: Input DataFrame
            topic: Target topic
            batch_size: Number of rows to process per batch
            include_rationale: Whether to include rationale in training prompts
            
        Returns:
            List of all prompts
        """
        self.logger.log_function_entry("formatter", "batch_process", 
                                     batch_size=batch_size, include_rationale=include_rationale)
        
        all_prompts = []
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            self.logger.info(f"Processing batch {start_idx//batch_size + 1}: rows {start_idx}-{end_idx-1}", 
                           "formatter", "batch_process")
            
            try:
                # Determine if this batch has training data
                has_ratings = 'Standardized_Rating' in batch_df.columns
                
                if has_ratings:
                    batch_prompts = self.create_training_prompts(
                        batch_df, topic, include_rationale
                    )
                else:
                    batch_prompts = self.create_inference_prompts(batch_df, topic)
                
                all_prompts.extend(batch_prompts)
                
            except Exception as e:
                self.logger.error(f"Error processing batch {start_idx//batch_size + 1}: {e}", 
                                "formatter", "batch_process", error=e)
                continue
        
        self.logger.info(f"Batch processing completed: {len(all_prompts)} total prompts", 
                        "formatter", "batch_process")
        self.logger.log_function_exit("formatter", "batch_process", len(all_prompts))
        
        return all_prompts
    
    def set_custom_templates(self, 
                           training_template: Optional[str] = None,
                           inference_template: Optional[str] = None) -> None:
        """
        Set custom prompt templates.
        
        Args:
            training_template: Custom training template
            inference_template: Custom inference template
        """
        if training_template:
            self.training_template = training_template
            self.logger.info("Updated training template", "formatter", "set_custom_templates")
        
        if inference_template:
            self.inference_template = inference_template
            self.logger.info("Updated inference template", "formatter", "set_custom_templates")

    def extract_key_findings(self, df: pd.DataFrame, topic: str) -> str:
        """
        Extract key findings and patterns from training data to create a context summary.
        
        Args:
            df: DataFrame with training data (must have Standardized_Rating)
            topic: Target topic being evaluated
            
        Returns:
            Context summary string with key findings
        """
        if 'Standardized_Rating' not in df.columns:
            return f"Evaluate cases for alignment with topic: {topic}"
        
        # Separate positive and negative cases
        positive_cases = df[df['Standardized_Rating'] > 0]
        negative_cases = df[df['Standardized_Rating'] == 0]
        
        context_parts = [f"You are evaluating whether cases align with the topic: {topic}"]
        
        # Extract key patterns from positive cases
        if len(positive_cases) > 0:
            positive_patterns = self._extract_patterns_from_cases(positive_cases, "positive")
            if positive_patterns:
                context_parts.append(f"\nLook for these indicators of {topic}:")
                context_parts.extend([f"- {pattern}" for pattern in positive_patterns])
        
        # Extract key patterns from negative cases
        if len(negative_cases) > 0:
            negative_patterns = self._extract_patterns_from_cases(negative_cases, "negative")
            if negative_patterns:
                context_parts.append(f"\nThese patterns suggest the case does NOT align with {topic}:")
                context_parts.extend([f"- {pattern}" for pattern in negative_patterns])
        
        # Add evaluation instructions
        context_parts.append(f"""
Evaluation Instructions:
- Rate 0: Case does NOT align with {topic}
- Rate 1: Case partially aligns with {topic} (some indicators present)
- Rate 2: Case clearly aligns with {topic} (multiple strong indicators)

Provide your rating and explain your reasoning based on the patterns above.
""")
        
        return "\n".join(context_parts)
    
    def _extract_patterns_from_cases(self, cases_df: pd.DataFrame, case_type: str) -> List[str]:
        """
        Extract common patterns from cases.
        
        Args:
            cases_df: DataFrame with cases of a specific type
            case_type: "positive" or "negative"
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        # Look for common terms in context blocks
        if 'Context_Block' in cases_df.columns:
            context_texts = cases_df['Context_Block'].dropna().astype(str)
            
            # Extract common medical terms and symptoms
            common_terms = self._extract_common_terms(context_texts)
            if common_terms:
                patterns.extend(common_terms[:5])  # Top 5 most common
        
        # Look for common diagnoses
        if 'Discharge Diagnosis' in cases_df.columns:
            diagnoses = cases_df['Discharge Diagnosis'].dropna().astype(str)
            common_diagnoses = self._extract_common_terms(diagnoses)
            if common_diagnoses:
                patterns.extend([f"Diagnosis: {d}" for d in common_diagnoses[:3]])
        
        # Look for common chief complaints
        if 'ChiefComplaintOrig' in cases_df.columns:
            complaints = cases_df['ChiefComplaintOrig'].dropna().astype(str)
            common_complaints = self._extract_common_terms(complaints)
            if common_complaints:
                patterns.extend([f"Complaint: {c}" for c in common_complaints[:3]])
        
        return patterns
    
    def _extract_common_terms(self, text_series: pd.Series) -> List[str]:
        """
        Extract common terms from text series.
        
        Args:
            text_series: Series of text strings
            
        Returns:
            List of common terms
        """
        import re
        from collections import Counter
        
        # Combine all text
        all_text = " ".join(text_series.str.lower())
        
        # Extract medical terms (words with 3+ characters)
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'been',
            'from', 'they', 'were', 'been', 'their', 'said', 'each', 'which',
            'there', 'were', 'time', 'very', 'into', 'just', 'only', 'know',
            'take', 'than', 'them', 'well', 'some', 'over', 'think', 'also',
            'back', 'after', 'work', 'first', 'want', 'because', 'any', 'these',
            'give', 'most', 'other', 'about', 'many', 'then', 'them', 'these',
            'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time',
            'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first',
            'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day',
            'did', 'get', 'come', 'made', 'may', 'part', 'patient', 'history',
            'present', 'complaint', 'diagnosis', 'treatment', 'medication',
            'hospital', 'emergency', 'department', 'admission', 'discharge'
        }
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Filter out stop words and get top terms
        filtered_words = [(word, count) for word, count in word_counts.items() 
                         if word not in stop_words and count > 1]
        
        # Sort by frequency and return top terms
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, count in filtered_words[:10]]

    def create_context_summary_prompts(self, 
                                     df: pd.DataFrame,
                                     topic: str,
                                     custom_template: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create prompts using context summary approach.
        
        Args:
            df: DataFrame with data
            topic: Target topic for alignment
            custom_template: Custom prompt template (optional)
            
        Returns:
            List of prompt dictionaries with context summary
        """
        self.logger.log_function_entry("formatter", "create_context_summary_prompts", topic=topic)
        
        # Validate input and create context block if needed
        is_valid, errors = self.validate_input_data(df)
        if not is_valid:
            error_msg = f"Input validation failed: {'; '.join(errors)}"
            self.logger.error(error_msg, "formatter", "create_context_summary_prompts")
            raise ValidationError(error_msg)
        
        # Ensure we have a context block
        if 'Context_Block' not in df.columns:
            df = self.create_context_block(df)
        
        # Extract key findings from training data
        context_summary = self.extract_key_findings(df, topic)
        
        # Use custom template or default
        template = custom_template or """CONTEXT:
{context_summary}

CASE TO EVALUATE:
{context_block}

Please evaluate this case based on the context above.
Rating: <0, 1, or 2>
Rationale: <your explanation>"""
        
        prompts = []
        
        for idx, row in df.iterrows():
            try:
                # Extract data
                context_block = row.get('Context_Block', 'Data unavailable')
                
                # Create prompt
                prompt_text = template.format(
                    context_summary=context_summary,
                    context_block=context_block
                )
                
                prompt_data = {
                    'id': row.get('C_BioSense_ID', f'row_{idx}'),
                    'prompt': prompt_text,
                    'topic': topic,
                    'row_index': idx,
                    'context_summary': context_summary
                }
                
                # Add rating if available (for training data)
                if 'Standardized_Rating' in df.columns and not pd.isna(row.get('Standardized_Rating')):
                    prompt_data['rating'] = int(row.get('Standardized_Rating'))
                    prompt_data['rationale'] = row.get('Rationale of Rating', 'No rationale provided')
                
                prompts.append(prompt_data)
                
            except Exception as e:
                self.logger.error(f"Error creating prompt for row {idx}: {e}", 
                                "formatter", "create_context_summary_prompts", error=e)
                continue
        
        self.logger.info(f"Created {len(prompts)} context summary prompts", 
                        "formatter", "create_context_summary_prompts")
        self.logger.log_function_exit("formatter", "create_context_summary_prompts", len(prompts))
        
        return prompts


def main():
    """Demo function to test the formatter."""
    import tempfile
    import os
    
    # Create sample data
    sample_data = pd.DataFrame({
        'C_BioSense_ID': ['P001', 'P002', 'P003'],
        'Context_Block': [
            'Chief Complaint: Fever, cough\nDischarge Diagnosis: Viral pneumonia\nTriage Notes: Persistent cough, oxygen saturation low',
            'Chief Complaint: Chest pain\nDischarge Diagnosis: Angina\nTriage Notes: Severe chest pain, treated with nitroglycerin',
            'Chief Complaint: Headache\nDischarge Diagnosis: Migraine\nTriage Notes: Throbbing headache, photophobia'
        ],
        'Standardized_Rating': [1, 2, 0],
        'Rationale of Rating': [
            'Clear respiratory infection',
            'Cardiac symptoms present',
            'Neurological condition'
        ]
    })
    
    # Initialize formatter
    formatter = Formatter(debug_mode=True)
    
    # Test training prompts
    print("Creating training prompts...")
    training_prompts = formatter.create_training_prompts(
        sample_data, "Respiratory Issues", include_rationale=True
    )
    
    print(f"Created {len(training_prompts)} training prompts")
    for i, prompt in enumerate(training_prompts[:2]):  # Show first 2
        print(f"\nPrompt {i+1}:")
        print(prompt['prompt'])
    
    # Test inference prompts (without ratings)
    print("\nCreating inference prompts...")
    inference_data = sample_data.drop(columns=['Standardized_Rating', 'Rationale of Rating'])
    inference_prompts = formatter.create_inference_prompts(inference_data, "Respiratory Issues")
    
    print(f"Created {len(inference_prompts)} inference prompts")
    for i, prompt in enumerate(inference_prompts[:2]):  # Show first 2
        print(f"\nPrompt {i+1}:")
        print(prompt['prompt'])
    
    # Test saving and loading
    print("\nTesting save/load functionality...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name
    
    try:
        formatter.save_prompts_to_jsonl(training_prompts, temp_file)
        loaded_prompts = formatter.load_prompts_from_jsonl(temp_file)
        print(f"Successfully saved and loaded {len(loaded_prompts)} prompts")
        
        # Test statistics
        stats = formatter.get_prompt_statistics(training_prompts)
        print(f"Prompt statistics: {stats}")
        
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    main() 