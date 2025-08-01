"""
Data Loader Module for Ollama Fine-Tuning and Evaluation Suite

This module handles loading and cleaning datasets, validating schema,
preparing rating mappings, and bundling text context for LLM training and inference.
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


class SchemaError(Exception):
    """Raised when dataset schema validation fails."""
    pass


class DataTypeError(Exception):
    """Raised when data type conversion fails."""
    pass


class FileError(Exception):
    """Raised when file operations fail."""
    pass


class DataLoader:
    """
    Handles dataset loading, validation, cleaning, and context preparation.
    
    Expected schema:
    - C_BioSense_ID (string) - Required
    - ChiefComplaintOrig (string) - Required
    - Discharge Diagnosis (string) - Required
    - Sex (string) - Required
    - Age (integer) - Required
    - Admit_Reason_Combo (string) - Required
    - Chief_Complaint_Combo (string) - Required
    - Diagnosis_Combo (string) - Required
    - CCDD (string) - Required
    - CCDDCategory (string) - Required
    - TriageNotes (string) - Required
    - Expert Rating (integer or string) - Required
    - Rationale of Rating (string) - Optional
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
        'TriageNotes',
        'Expert Rating'
    ]
    
    OPTIONAL_FIELDS = [
        'Rationale of Rating'
    ]
    
    TEXT_FIELDS_FOR_CONTEXT = [
        'ChiefComplaintOrig',
        'Discharge Diagnosis', 
        'TriageNotes',
        'Admit_Reason_Combo',
        'Chief_Complaint_Combo',
        'Diagnosis_Combo',
        'CCDD',
        'CCDDCategory'
    ]
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the DataLoader.
        
        Args:
            debug_mode: Enable verbose logging for debugging
        """
        self.debug_mode = debug_mode
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('data_loader.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileError: If file cannot be read
        """
        try:
            self.logger.info(f"Loading dataset from: {file_path}")
            
            if not Path(file_path).exists():
                raise FileError(f"File not found: {file_path}")
                
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except pd.errors.EmptyDataError:
            raise FileError(f"File is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise FileError(f"Failed to parse CSV file: {e}")
        except Exception as e:
            raise FileError(f"Unexpected error loading file: {e}")
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, str]]:
        """
        Validate dataset schema against expected fields.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, missing_fields, column_mapping_suggestions)
        """
        self.logger.info("Validating dataset schema")
        
        missing_fields = []
        column_mapping_suggestions = {}
        
        # Check for required fields
        for field in self.REQUIRED_FIELDS:
            if field not in df.columns:
                missing_fields.append(field)
                
                # Suggest potential column mappings
                potential_matches = self._find_potential_column_matches(field, df.columns)
                if potential_matches:
                    column_mapping_suggestions[field] = potential_matches
        
        is_valid = len(missing_fields) == 0
        
        if is_valid:
            self.logger.info("Schema validation passed")
        else:
            self.logger.warning(f"Schema validation failed. Missing fields: {missing_fields}")
            
        return is_valid, missing_fields, column_mapping_suggestions
    
    def _find_potential_column_matches(self, target_field: str, available_columns: List[str]) -> List[str]:
        """
        Find potential column matches for missing fields.
        
        Args:
            target_field: The field we're looking for
            available_columns: Available column names
            
        Returns:
            List of potential matches
        """
        target_lower = target_field.lower().replace(' ', '').replace('_', '')
        matches = []
        
        for col in available_columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            
            # Exact match
            if col_lower == target_lower:
                matches.append(col)
            # Contains target
            elif target_lower in col_lower or col_lower in target_lower:
                matches.append(col)
            # Common abbreviations
            elif self._check_abbreviations(target_field, col):
                matches.append(col)
                
        return matches[:3]  # Limit to top 3 matches
    
    def _check_abbreviations(self, target_field: str, column: str) -> bool:
        """Check for common abbreviations between target field and column."""
        abbreviations = {
            'C_BioSense_ID': ['id', 'biosense', 'patient_id'],
            'ChiefComplaintOrig': ['chief', 'complaint', 'cc'],
            'Discharge Diagnosis': ['diagnosis', 'discharge', 'dx'],
            'Expert Rating': ['rating', 'expert', 'score'],
            'Rationale of Rating': ['rationale', 'reason', 'explanation']
        }
        
        if target_field in abbreviations:
            target_abbrevs = abbreviations[target_field]
            col_lower = column.lower()
            return any(abbrev in col_lower for abbrev in target_abbrevs)
            
        return False
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning dataset")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Type casting
        df_clean = self._cast_data_types(df_clean)
        
        # Drop rows with missing C_BioSense_ID
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['C_BioSense_ID'])
        dropped_count = initial_count - len(df_clean)
        
        if dropped_count > 0:
            self.logger.warning(f"Dropped {dropped_count} rows with missing C_BioSense_ID")
        
        self.logger.info(f"Dataset cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill missing values in text fields with empty string
        text_fields = [col for col in df.columns if col in self.TEXT_FIELDS_FOR_CONTEXT]
        df[text_fields] = df[text_fields].fillna('')
        
        # Fill missing values in other string fields
        string_fields = ['Sex', 'CCDD', 'CCDDCategory']
        for field in string_fields:
            if field in df.columns:
                df[field] = df[field].fillna('')
        
        # Fill missing Age with 0 (will be handled in type casting)
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(0)
            
        return df
    
    def _cast_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast data types appropriately."""
        try:
            # Cast Age to integer
            if 'Age' in df.columns:
                df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0).astype(int)
                self.logger.info("Successfully cast Age to integer")
        except Exception as e:
            self.logger.warning(f"Failed to cast Age to integer: {e}")
            
        try:
            # Try to cast Expert Rating to integer
            if 'Expert Rating' in df.columns:
                # First, try direct conversion
                df['Expert Rating'] = pd.to_numeric(df['Expert Rating'], errors='coerce')
                
                # Check if we have any non-numeric values
                non_numeric_mask = df['Expert Rating'].isna()
                if non_numeric_mask.any():
                    self.logger.warning(f"Found {non_numeric_mask.sum()} non-numeric Expert Rating values")
                    # Keep as string for these cases
                    df.loc[non_numeric_mask, 'Expert Rating'] = df.loc[non_numeric_mask, 'Expert Rating'].astype(str)
                else:
                    df['Expert Rating'] = df['Expert Rating'].astype(int)
                    self.logger.info("Successfully cast Expert Rating to integer")
                    
        except Exception as e:
            self.logger.warning(f"Failed to cast Expert Rating: {e}")
            
        return df
    
    def extract_unique_ratings(self, df: pd.DataFrame) -> List[Any]:
        """
        Extract unique values from Expert Rating column.
        
        Args:
            df: DataFrame with Expert Rating column
            
        Returns:
            List of unique rating values
        """
        if 'Expert Rating' not in df.columns:
            raise SchemaError("Expert Rating column not found")
            
        unique_ratings = df['Expert Rating'].unique().tolist()
        self.logger.info(f"Found {len(unique_ratings)} unique rating values: {unique_ratings}")
        
        return unique_ratings
    
    def create_context_block(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create context block by merging key text fields.
        
        Args:
            df: DataFrame with text fields
            
        Returns:
            DataFrame with added Context_Block column
        """
        self.logger.info("Creating context blocks")
        
        def create_context_string(row):
            """Create a formatted context string from text fields."""
            context_parts = []
            
            # Add patient info
            if pd.notna(row.get('Age')) and row.get('Age', 0) > 0:
                context_parts.append(f"Age: {row.get('Age')}")
            if row.get('Sex'):
                context_parts.append(f"Sex: {row.get('Sex')}")
                
            # Add medical information
            if row.get('ChiefComplaintOrig'):
                context_parts.append(f"Chief Complaint: {row.get('ChiefComplaintOrig')}")
            if row.get('Discharge Diagnosis'):
                context_parts.append(f"Discharge Diagnosis: {row.get('Discharge Diagnosis')}")
            if row.get('TriageNotes'):
                context_parts.append(f"Triage Notes: {row.get('TriageNotes')}")
            if row.get('Admit_Reason_Combo'):
                context_parts.append(f"Admit Reason: {row.get('Admit_Reason_Combo')}")
            if row.get('Chief_Complaint_Combo'):
                context_parts.append(f"Chief Complaint Combo: {row.get('Chief_Complaint_Combo')}")
            if row.get('Diagnosis_Combo'):
                context_parts.append(f"Diagnosis Combo: {row.get('Diagnosis_Combo')}")
            if row.get('CCDD'):
                context_parts.append(f"CCDD: {row.get('CCDD')}")
            if row.get('CCDDCategory'):
                context_parts.append(f"Category: {row.get('CCDDCategory')}")
                
            return "\n".join(context_parts) if context_parts else "No context available"
        
        # Apply context creation to each row
        df['Context_Block'] = df.apply(create_context_string, axis=1)
        
        self.logger.info("Context blocks created successfully")
        return df
    
    def add_standardized_rating_column(self, df: pd.DataFrame, rating_mapping: Dict[Any, int]) -> pd.DataFrame:
        """
        Add standardized rating column based on user mapping.
        
        Args:
            df: DataFrame with Expert Rating column
            rating_mapping: Dictionary mapping original ratings to standardized values
            
        Returns:
            DataFrame with added Standardized_Rating column
        """
        self.logger.info("Adding standardized rating column")
        
        def map_rating(rating):
            """Map original rating to standardized value."""
            return rating_mapping.get(rating, -1)  # Default to -1 (Unknown)
        
        df['Standardized_Rating'] = df['Expert Rating'].apply(map_rating)
        
        # Log mapping summary
        mapping_summary = df.groupby(['Expert Rating', 'Standardized_Rating']).size()
        self.logger.info(f"Rating mapping summary:\n{mapping_summary}")
        
        return df
    
    def process_dataset(self, file_path: str, rating_mapping: Optional[Dict[Any, int]] = None) -> Tuple[pd.DataFrame, List[Any], Dict[str, Any]]:
        """
        Complete dataset processing pipeline.
        
        Args:
            file_path: Path to CSV file
            rating_mapping: Optional mapping for rating standardization
            
        Returns:
            Tuple of (cleaned_dataframe, unique_ratings, metadata)
        """
        self.logger.info("Starting dataset processing pipeline")
        
        # Step 1: Load and validate
        df = self.load_dataset(file_path)
        is_valid, missing_fields, column_mapping_suggestions = self.validate_schema(df)
        
        if not is_valid:
            raise SchemaError(f"Schema validation failed. Missing fields: {missing_fields}")
        
        # Step 2: Clean dataset
        df_clean = self.clean_dataset(df)
        
        # Step 3: Extract unique ratings
        unique_ratings = self.extract_unique_ratings(df_clean)
        
        # Step 4: Create context blocks
        df_clean = self.create_context_block(df_clean)
        
        # Step 5: Add standardized rating if mapping provided
        if rating_mapping:
            df_clean = self.add_standardized_rating_column(df_clean, rating_mapping)
        else:
            # Add placeholder column
            df_clean['Standardized_Rating'] = -1
            self.logger.info("Added placeholder Standardized_Rating column (set to -1)")
        
        # Prepare metadata
        metadata = {
            'original_shape': df.shape,
            'cleaned_shape': df_clean.shape,
            'unique_ratings': unique_ratings,
            'rating_mapping': rating_mapping,
            'missing_fields': missing_fields,
            'column_mapping_suggestions': column_mapping_suggestions
        }
        
        self.logger.info("Dataset processing completed successfully")
        return df_clean, unique_ratings, metadata
    
    def process_dataset_from_dataframe(self, df: pd.DataFrame, rating_mapping: Optional[Dict[Any, int]] = None) -> Tuple[pd.DataFrame, List[Any], Dict[str, Any]]:
        """
        Process dataset from existing DataFrame (for testing and integration).
        
        Args:
            df: Input DataFrame
            rating_mapping: Optional mapping for rating standardization
            
        Returns:
            Tuple of (cleaned_dataframe, unique_ratings, metadata)
        """
        self.logger.info("Starting dataset processing from DataFrame")
        
        # Step 1: Validate
        is_valid, missing_fields, column_mapping_suggestions = self.validate_schema(df)
        
        if not is_valid:
            raise SchemaError(f"Schema validation failed. Missing fields: {missing_fields}")
        
        # Step 2: Clean dataset
        df_clean = self.clean_dataset(df)
        
        # Step 3: Extract unique ratings
        unique_ratings = self.extract_unique_ratings(df_clean)
        
        # Step 4: Create context blocks
        df_clean = self.create_context_block(df_clean)
        
        # Step 5: Add standardized rating if mapping provided
        if rating_mapping:
            df_clean = self.add_standardized_rating_column(df_clean, rating_mapping)
        else:
            # Add placeholder column
            df_clean['Standardized_Rating'] = -1
            self.logger.info("Added placeholder Standardized_Rating column (set to -1)")
        
        # Prepare metadata
        metadata = {
            'original_shape': df.shape,
            'cleaned_shape': df_clean.shape,
            'unique_ratings': unique_ratings,
            'rating_mapping': rating_mapping,
            'missing_fields': missing_fields,
            'column_mapping_suggestions': column_mapping_suggestions
        }
        
        self.logger.info("Dataset processing from DataFrame completed successfully")
        return df_clean, unique_ratings, metadata
    
    def save_processed_dataset(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed dataset to file.
        
        Args:
            df: Processed DataFrame
            output_path: Output file path
        """
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Processed dataset saved to: {output_path}")
        except Exception as e:
            raise FileError(f"Failed to save processed dataset: {e}")
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str) -> None:
        """
        Save processing metadata to JSON file.
        
        Args:
            metadata: Processing metadata
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.info(f"Metadata saved to: {output_path}")
        except Exception as e:
            raise FileError(f"Failed to save metadata: {e}")


def main():
    """Example usage of DataLoader."""
    # Example usage
    loader = DataLoader(debug_mode=True)
    
    try:
        # Process dataset
        df, unique_ratings, metadata = loader.process_dataset("data/sample_dataset.csv")
        
        # Save results
        loader.save_processed_dataset(df, "outputs/processed_dataset.csv")
        loader.save_metadata(metadata, "outputs/processing_metadata.json")
        
        print(f"Processing completed. Found {len(unique_ratings)} unique ratings: {unique_ratings}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 