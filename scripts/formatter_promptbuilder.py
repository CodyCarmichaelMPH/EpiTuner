"""
Formatter PromptBuilder Module for Ollama Fine-Tuning and Evaluation Suite

This module transforms structured dataset rows into optimized, consistent prompts 
for the Ollama model (or fine-tuned model set). It ensures that each row of data 
is presented with the necessary context, formatted predictably for reliable inference.
"""

import pandas as pd
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


class FormattingError(Exception):
    """Raised when prompt formatting operations fail."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class PromptBuilder:
    """
    Handles the transformation of dataset rows into structured prompts for LLM inference.
    
    Creates consistent, well-formatted prompts that include:
    - Patient context information
    - Medical complaint and diagnosis data
    - Task instructions with rating schema
    - Expected response format
    """
    
    DEFAULT_PROMPT_TEMPLATE = """
Context:
- Patient Info: Age {age}, Sex {sex}
- Chief Complaint: {chief_complaint}
- Discharge Diagnosis: {discharge_diagnosis}
- Admit Reason: {admit_reason}
- Combined Complaints: {chief_complaint_combo}
- Diagnosis Combo: {diagnosis_combo}
- CCDD: {ccdd}, Category: {ccdd_category}
- Triage Notes: {triage_notes}

Task:
Based on the above information, evaluate whether this record aligns with the topic(s): {target_topics}.
Use the following rating schema:
{schema_description}

Respond with:
- Numeric rating (from schema).
- Brief rationale (1–3 sentences).
"""
    
    REQUIRED_FIELDS = [
        'C_BioSense_ID',
        'ChiefComplaintOrig',
        'Discharge Diagnosis', 
        'Sex',
        'Age',
        'Admit_Reason_Combo',
        'Chief_Complaint_Combo',
        'Diagnosis_Combo',
        'CCDD',
        'CCDDCategory',
        'TriageNotes'
    ]
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the PromptBuilder.
        
        Args:
            debug_mode: Enable verbose logging for debugging
        """
        self.debug_mode = debug_mode
        self._setup_logging()
        self.prompt_template = self.DEFAULT_PROMPT_TEMPLATE
        
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('formatter_promptbuilder.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def set_prompt_template(self, template: str) -> None:
        """
        Set a custom prompt template.
        
        Args:
            template: Custom prompt template string with placeholders
        """
        self.prompt_template = template
        self.logger.info("Custom prompt template set")
    
    def validate_row_data(self, row: pd.Series) -> Tuple[bool, List[str]]:
        """
        Validate that a row has required context fields.
        
        Args:
            row: DataFrame row to validate
            
        Returns:
            Tuple of (is_valid, missing_fields)
        """
        missing_fields = []
        
        for field in self.REQUIRED_FIELDS:
            if field not in row.index:
                missing_fields.append(field)
            elif pd.isna(row[field]) or str(row[field]).strip() == '':
                missing_fields.append(field)
        
        is_valid = len(missing_fields) == 0
        
        if not is_valid:
            self.logger.warning(f"Row {row.get('C_BioSense_ID', 'Unknown')} missing fields: {missing_fields}")
        
        return is_valid, missing_fields
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text fields.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text is None:
            return "N/A"
        
        # Convert to string and strip whitespace
        cleaned = str(text).strip()
        
        # Replace empty strings with N/A
        if cleaned == '':
            return "N/A"
        
        return cleaned
    
    def create_schema_description(self, rating_mapping: Dict[Any, int]) -> str:
        """
        Create a human-readable description of the rating schema.
        
        Args:
            rating_mapping: Dictionary mapping rating values to meanings
            
        Returns:
            Formatted schema description
        """
        descriptions = []
        for meaning, value in rating_mapping.items():
            descriptions.append(f"- {meaning} ({value})")
        
        return "\n".join(descriptions)
    
    def build_prompt(self, row: pd.Series, target_topics: str, 
                    rating_mapping: Dict[Any, int]) -> str:
        """
        Build a structured prompt from a dataset row.
        
        Args:
            row: DataFrame row containing patient data
            target_topics: Topics to evaluate alignment with
            rating_mapping: Dictionary mapping rating values to meanings
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValidationError: If required fields are missing
        """
        # Validate row data
        is_valid, missing_fields = self.validate_row_data(row)
        if not is_valid:
            self.logger.warning(f"Row {row.get('C_BioSense_ID', 'Unknown')} has missing fields: {missing_fields}")
        
        # Preprocess text fields
        age = self.preprocess_text(row.get('Age', 'N/A'))
        sex = self.preprocess_text(row.get('Sex', 'N/A'))
        chief_complaint = self.preprocess_text(row.get('ChiefComplaintOrig', 'N/A'))
        discharge_diagnosis = self.preprocess_text(row.get('Discharge Diagnosis', 'N/A'))
        admit_reason = self.preprocess_text(row.get('Admit_Reason_Combo', 'N/A'))
        chief_complaint_combo = self.preprocess_text(row.get('Chief_Complaint_Combo', 'N/A'))
        diagnosis_combo = self.preprocess_text(row.get('Diagnosis_Combo', 'N/A'))
        ccdd = self.preprocess_text(row.get('CCDD', 'N/A'))
        ccdd_category = self.preprocess_text(row.get('CCDDCategory', 'N/A'))
        triage_notes = self.preprocess_text(row.get('TriageNotes', 'N/A'))
        
        # Create schema description
        schema_description = self.create_schema_description(rating_mapping)
        
        # Build prompt using template
        prompt = self.prompt_template.format(
            age=age,
            sex=sex,
            chief_complaint=chief_complaint,
            discharge_diagnosis=discharge_diagnosis,
            admit_reason=admit_reason,
            chief_complaint_combo=chief_complaint_combo,
            diagnosis_combo=diagnosis_combo,
            ccdd=ccdd,
            ccdd_category=ccdd_category,
            triage_notes=triage_notes,
            target_topics=target_topics,
            schema_description=schema_description
        )
        
        return prompt.strip()
    
    def format_dataset(self, df: pd.DataFrame, target_topics: str, 
                      rating_mapping: Dict[Any, int]) -> pd.DataFrame:
        """
        Format an entire dataset with prompts.
        
        Args:
            df: DataFrame with patient data
            target_topics: Topics to evaluate alignment with
            rating_mapping: Dictionary mapping rating values to meanings
            
        Returns:
            DataFrame with added 'Prompt' column
        """
        self.logger.info(f"Formatting dataset with {len(df)} rows")
        
        prompts = []
        row_ids = []
        
        for idx, row in df.iterrows():
            try:
                prompt = self.build_prompt(row, target_topics, rating_mapping)
                prompts.append(prompt)
                row_ids.append(row.get('C_BioSense_ID', f'row_{idx}'))
            except Exception as e:
                self.logger.error(f"Error formatting row {idx}: {e}")
                prompts.append("ERROR: Could not format prompt")
                row_ids.append(row.get('C_BioSense_ID', f'row_{idx}'))
        
        # Create new DataFrame with prompts
        result_df = df.copy()
        result_df['Prompt'] = prompts
        result_df['Row_ID'] = row_ids
        
        self.logger.info(f"Successfully formatted {len(prompts)} prompts")
        
        return result_df
    
    def save_formatted_dataset(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save formatted dataset to file.
        
        Args:
            df: DataFrame with prompts
            output_path: Path to save the file
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            df.to_csv(output_path, index=False)
            self.logger.info(f"Formatted dataset saved to {output_path}")
            
        except Exception as e:
            raise FormattingError(f"Failed to save formatted dataset: {e}")
    
    def save_prompts_jsonl(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save prompts in JSONL format for LLM training/inference.
        
        Args:
            df: DataFrame with prompts
            output_path: Path to save the JSONL file
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for idx, row in df.iterrows():
                    prompt_data = {
                        "C_BioSense_ID": row.get('C_BioSense_ID', f'row_{idx}'),
                        "prompt": row.get('Prompt', ''),
                        "target_rating": row.get('Standardized_Rating', None),
                        "original_rating": row.get('Expert Rating', None)
                    }
                    f.write(json.dumps(prompt_data) + '\n')
            
            self.logger.info(f"Prompts saved in JSONL format to {output_path}")
            
        except Exception as e:
            raise FormattingError(f"Failed to save JSONL prompts: {e}")
    
    def create_prompt_metadata(self, df: pd.DataFrame, target_topics: str, 
                             rating_mapping: Dict[Any, int]) -> Dict[str, Any]:
        """
        Create metadata about the prompt formatting process.
        
        Args:
            df: DataFrame that was formatted
            target_topics: Topics used for evaluation
            rating_mapping: Rating mapping used
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "formatting_info": {
                "total_rows": len(df),
                "target_topics": target_topics,
                "rating_mapping": rating_mapping,
                "prompt_template_used": self.prompt_template[:200] + "..." if len(self.prompt_template) > 200 else self.prompt_template
            },
            "data_quality": {
                "rows_with_missing_fields": 0,
                "rows_with_na_values": 0
            }
        }
        
        # Count data quality issues
        for idx, row in df.iterrows():
            is_valid, missing_fields = self.validate_row_data(row)
            if not is_valid:
                metadata["data_quality"]["rows_with_missing_fields"] += 1
            
            na_count = row.isna().sum()
            if na_count > 0:
                metadata["data_quality"]["rows_with_na_values"] += 1
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str) -> None:
        """
        Save formatting metadata to file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save the metadata
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Formatting metadata saved to {output_path}")
            
        except Exception as e:
            raise FormattingError(f"Failed to save metadata: {e}")
    
    def process_dataset(self, df: pd.DataFrame, target_topics: str, 
                       rating_mapping: Dict[Any, int], output_dir: str = "outputs") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete processing pipeline for formatting a dataset.
        
        Args:
            df: Input DataFrame
            target_topics: Topics to evaluate alignment with
            rating_mapping: Dictionary mapping rating values to meanings
            output_dir: Directory to save outputs
            
        Returns:
            Tuple of (formatted_dataframe, metadata)
        """
        self.logger.info("Starting dataset formatting process")
        
        # Format the dataset
        formatted_df = self.format_dataset(df, target_topics, rating_mapping)
        
        # Create metadata
        metadata = self.create_prompt_metadata(df, target_topics, rating_mapping)
        
        # Save outputs
        output_path = Path(output_dir)
        formatted_df.to_csv(output_path / "formatted_dataset.csv", index=False)
        self.save_prompts_jsonl(formatted_df, output_path / "prompts.jsonl")
        self.save_metadata(metadata, output_path / "formatting_metadata.json")
        
        self.logger.info("Dataset formatting process completed")
        
        return formatted_df, metadata


def main():
    """Demo function to test the PromptBuilder."""
    import sys
    
    # Create sample data
    sample_data = {
        'C_BioSense_ID': ['P001', 'P002'],
        'ChiefComplaintOrig': ['Fever', 'Chest pain'],
        'Discharge Diagnosis': ['Viral infection', 'Angina'],
        'Sex': ['M', 'F'],
        'Age': [25, 45],
        'Admit_Reason_Combo': ['Fever', 'Chest pain'],
        'Chief_Complaint_Combo': ['Fever', 'Chest pain'],
        'Diagnosis_Combo': ['Infection', 'Cardiac'],
        'CCDD': ['Fever', 'Chest pain'],
        'CCDDCategory': ['Viral', 'Cardiac'],
        'TriageNotes': ['High fever with chills', 'Severe chest pain'],
        'Expert Rating': [1, 2],
        'Standardized_Rating': [1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Sample rating mapping
    rating_mapping = {
        'Match': 1,
        'Does Not Match': 0,
        'Unknown': -1
    }
    
    # Initialize formatter
    formatter = PromptBuilder(debug_mode=True)
    
    # Process dataset
    target_topics = "respiratory infections and cardiac conditions"
    formatted_df, metadata = formatter.process_dataset(df, target_topics, rating_mapping)
    
    print("Sample formatted prompt:")
    print(formatted_df['Prompt'].iloc[0])
    print("\nMetadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main() 