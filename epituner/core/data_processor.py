#!/usr/bin/env python3
"""
Simple data processor for medical CSV files
Converts medical records to training format
"""

import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


class MedicalDataProcessor:
    """Process medical CSV data for training"""
    
    REQUIRED_COLUMNS = [
        'C_Biosense_ID',
        'ChiefComplaintOrig', 
        'DischargeDiagnosis',
        'Expert Rating',
        'Rationale_of_Rating'
    ]
    
    @classmethod
    def validate_csv(cls, csv_path: str) -> Tuple[bool, List[str]]:
        """Validate CSV has required columns"""
        try:
            df = pd.read_csv(csv_path)
            errors = []
            
            # Check required columns
            missing = set(cls.REQUIRED_COLUMNS) - set(df.columns)
            if missing:
                errors.append(f"Missing columns: {', '.join(missing)}")
            
            # Check for empty data
            if len(df) == 0:
                errors.append("File is empty")
                
            # Check expert ratings
            if 'Expert Rating' in df.columns:
                valid_ratings = {'Match', 'Not a Match', 'Unknown/Not able to determine'}
                invalid = set(df['Expert Rating'].dropna()) - valid_ratings
                if invalid:
                    errors.append(f"Invalid Expert Rating values: {', '.join(invalid)}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error reading CSV: {str(e)}"]
    
    @classmethod
    def load_data(cls, csv_path: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        is_valid, errors = cls.validate_csv(csv_path)
        if not is_valid:
            raise ValueError(f"Invalid CSV file: {'; '.join(errors)}")
        
        return pd.read_csv(csv_path)
    
    @classmethod
    def create_training_text(cls, record: Dict, topic: str) -> str:
        """Convert medical record to training text"""
        return f"""Medical Classification Task: {topic}

Patient Record:
- Chief Complaint: {record.get('ChiefComplaintOrig', 'Not documented')}
- Discharge Diagnosis: {record.get('DischargeDiagnosis', 'Not documented')}
- Demographics: Sex: {record.get('Sex', 'N/A')}, Age: {record.get('Age', 'N/A')}
- Additional Notes: {record.get('TriageNotes', record.get('TriageNotesOrig', 'None'))}

Classification: {record.get('Expert Rating', 'Unknown')}
Reasoning: {record.get('Rationale_of_Rating', 'No reasoning provided')}"""

    @classmethod
    def prepare_for_training(cls, df: pd.DataFrame, topic: str) -> List[str]:
        """Convert DataFrame to list of training texts"""
        training_texts = []
        
        for _, row in df.iterrows():
            text = cls.create_training_text(row.to_dict(), topic)
            training_texts.append(text)
            
        return training_texts
    
    @classmethod  
    def get_stats(cls, df: pd.DataFrame) -> Dict:
        """Get basic statistics about the dataset"""
        stats = {
            'total_records': len(df),
            'rating_distribution': df['Expert Rating'].value_counts().to_dict() if 'Expert Rating' in df.columns else {},
            'complete_records': len(df.dropna(subset=cls.REQUIRED_COLUMNS))
        }
        return stats

