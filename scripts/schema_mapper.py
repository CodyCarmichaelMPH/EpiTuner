"""
Schema Mapper Module for Ollama Fine-Tuning and Evaluation Suite

This module handles the transformation of non-standard or mixed expert ratings 
into a standardized schema for consistent model training and inference.
"""

import pandas as pd
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


class MappingError(Exception):
    """Raised when rating mapping operations fail."""
    pass


class DataTypeError(Exception):
    """Raised when data type conversion fails."""
    pass


class SchemaMapper:
    """
    Handles the mapping of expert ratings to standardized values.
    
    Standard rating schema:
    - Match (1): Clear match between complaint/diagnosis and target condition
    - Does Not Match (0): No match between complaint/diagnosis and target condition  
    - Unknown (-1): Uncertain or unclear relationship
    """
    
    STANDARD_RATINGS = {
        'Match': 1,
        'Does Not Match': 0, 
        'Unknown': -1,
        'Partial Match': 2  # Optional intermediate rating
    }
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the SchemaMapper.
        
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
                logging.FileHandler('schema_mapper.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_unique_ratings(self, df: pd.DataFrame) -> List[Any]:
        """
        Extract unique values from Expert Rating column.
        
        Args:
            df: DataFrame with Expert Rating column
            
        Returns:
            List of unique rating values
            
        Raises:
            MappingError: If Expert Rating column is missing
        """
        if 'Expert Rating' not in df.columns:
            raise MappingError("Expert Rating column not found in DataFrame")
            
        unique_ratings = df['Expert Rating'].unique().tolist()
        self.logger.info(f"Found {len(unique_ratings)} unique rating values: {unique_ratings}")
        
        return unique_ratings
    
    def validate_mapping(self, df: pd.DataFrame, rating_mapping: Dict[Any, int]) -> Tuple[bool, List[Any]]:
        """
        Validate that all unique ratings in the dataset have mappings.
        
        Args:
            df: DataFrame with Expert Rating column
            rating_mapping: Dictionary mapping original ratings to standardized values
            
        Returns:
            Tuple of (is_valid, unmapped_values)
        """
        unique_ratings = self.extract_unique_ratings(df)
        unmapped_values = [rating for rating in unique_ratings if rating not in rating_mapping]
        
        is_valid = len(unmapped_values) == 0
        
        if not is_valid:
            self.logger.warning(f"Found {len(unmapped_values)} unmapped values: {unmapped_values}")
        else:
            self.logger.info("All rating values have valid mappings")
            
        return is_valid, unmapped_values
    
    def apply_mapping(self, df: pd.DataFrame, rating_mapping: Dict[Any, int]) -> pd.DataFrame:
        """
        Apply rating mapping to create Standardized_Rating column.
        
        Args:
            df: DataFrame with Expert Rating column
            rating_mapping: Dictionary mapping original ratings to standardized values
            
        Returns:
            DataFrame with added Standardized_Rating column
            
        Raises:
            MappingError: If mapping validation fails
        """
        self.logger.info("Applying rating mapping to dataset")
        
        # Validate mapping first
        is_valid, unmapped_values = self.validate_mapping(df, rating_mapping)
        
        if not is_valid:
            raise MappingError(f"Missing mappings for values: {unmapped_values}")
        
        # Create a copy to avoid modifying original
        df_mapped = df.copy()
        
        # Apply mapping
        def map_rating(rating):
            """Map original rating to standardized value."""
            return rating_mapping.get(rating, -99)  # -99 for unmapped values
        
        df_mapped['Standardized_Rating'] = df_mapped['Expert Rating'].apply(map_rating)
        
        # Log mapping summary
        mapping_summary = df_mapped.groupby(['Expert Rating', 'Standardized_Rating']).size()
        self.logger.info(f"Rating mapping summary:\n{mapping_summary}")
        
        # Check for any -99 values (shouldn't happen after validation, but just in case)
        unmapped_count = (df_mapped['Standardized_Rating'] == -99).sum()
        if unmapped_count > 0:
            self.logger.warning(f"Found {unmapped_count} rows with unmapped ratings (set to -99)")
        
        self.logger.info("Rating mapping applied successfully")
        return df_mapped
    
    def create_mapping_metadata(self, rating_mapping: Dict[Any, int], 
                               original_values: List[Any]) -> Dict[str, Any]:
        """
        Create metadata dictionary for the rating mapping.
        
        Args:
            rating_mapping: Dictionary mapping original ratings to standardized values
            original_values: List of original unique rating values
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "original_values": original_values,
            "mapped_values": list(rating_mapping.values()),
            "schema": rating_mapping,
            "standard_ratings": self.STANDARD_RATINGS,
            "mapping_summary": {
                "total_mappings": len(rating_mapping),
                "unique_mapped_values": list(set(rating_mapping.values())),
                "mapping_coverage": f"{len(rating_mapping)}/{len(original_values)} values mapped"
            }
        }
        
        self.logger.info(f"Created mapping metadata: {metadata['mapping_summary']}")
        return metadata
    
    def save_mapping_metadata(self, metadata: Dict[str, Any], output_path: str) -> None:
        """
        Save mapping metadata to JSON file.
        
        Args:
            metadata: Mapping metadata dictionary
            output_path: Output file path
            
        Raises:
            MappingError: If file save fails
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert metadata to JSON-serializable format
            json_metadata = {}
            for key, value in metadata.items():
                if key == 'schema':
                    # Convert schema keys to strings for JSON serialization
                    json_metadata[key] = {str(k): v for k, v in value.items()}
                else:
                    json_metadata[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(json_metadata, f, indent=2, default=str)
            
            self.logger.info(f"Mapping metadata saved to: {output_path}")
            
        except Exception as e:
            raise MappingError(f"Failed to save mapping metadata: {e}")
    
    def load_mapping_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Load mapping metadata from JSON file.
        
        Args:
            file_path: Path to mapping metadata file
            
        Returns:
            Mapping metadata dictionary
            
        Raises:
            MappingError: If file load fails
        """
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Mapping metadata loaded from: {file_path}")
            return metadata
            
        except Exception as e:
            raise MappingError(f"Failed to load mapping metadata: {e}")
    
    def suggest_mapping(self, unique_ratings: List[Any]) -> Dict[Any, int]:
        """
        Suggest a default mapping based on common patterns.
        
        Args:
            unique_ratings: List of unique rating values
            
        Returns:
            Suggested mapping dictionary
        """
        self.logger.info("Generating suggested mapping for rating values")
        
        suggested_mapping = {}
        
        for rating in unique_ratings:
            rating_str = str(rating).lower().strip()
            
            # Common patterns for negative matches - check for "no match" first to avoid conflicts
            if any(pattern in rating_str for pattern in ['no match', 'does not match', 'negative']):
                suggested_mapping[rating] = 0
            # Common patterns for positive matches (but not "no match" which was already handled)
            elif any(pattern in rating_str for pattern in ['match', 'yes', '1', 'positive', 'clear', 'strong']):
                suggested_mapping[rating] = 1
            # Common patterns for partial/uncertain matches
            elif any(pattern in rating_str for pattern in ['partial', 'weak', '2', 'maybe', 'uncertain']):
                suggested_mapping[rating] = 2
            # Check for simple "no" or "0" (but not "no match" which was already handled)
            elif rating_str in ['no', '0']:
                suggested_mapping[rating] = 0
            # Default to unknown
            else:
                suggested_mapping[rating] = -1
        
        self.logger.info(f"Suggested mapping: {suggested_mapping}")
        return suggested_mapping
    
    def process_mapping(self, df: pd.DataFrame, rating_mapping: Dict[Any, int], 
                       output_dir: str = "outputs") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete mapping processing pipeline.
        
        Args:
            df: DataFrame with Expert Rating column
            rating_mapping: Dictionary mapping original ratings to standardized values
            output_dir: Directory to save mapping metadata
            
        Returns:
            Tuple of (mapped_dataframe, mapping_metadata)
        """
        self.logger.info("Starting schema mapping processing pipeline")
        
        # Step 1: Extract unique ratings
        unique_ratings = self.extract_unique_ratings(df)
        
        # Step 2: Apply mapping
        df_mapped = self.apply_mapping(df, rating_mapping)
        
        # Step 3: Create metadata
        metadata = self.create_mapping_metadata(rating_mapping, unique_ratings)
        
        # Step 4: Save metadata
        metadata_path = Path(output_dir) / "rating_mapping_metadata.json"
        self.save_mapping_metadata(metadata, str(metadata_path))
        
        self.logger.info("Schema mapping processing completed successfully")
        return df_mapped, metadata
    
    def get_mapping_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the current rating distribution.
        
        Args:
            df: DataFrame with Expert Rating column
            
        Returns:
            Statistics dictionary
        """
        if 'Expert Rating' not in df.columns:
            raise MappingError("Expert Rating column not found")
        
        rating_counts = df['Expert Rating'].value_counts()
        total_rows = len(df)
        
        stats = {
            "total_rows": total_rows,
            "unique_ratings": len(rating_counts),
            "rating_distribution": rating_counts.to_dict(),
            "rating_percentages": (rating_counts / total_rows * 100).to_dict()
        }
        
        self.logger.info(f"Rating statistics: {stats}")
        return stats


def main():
    """Example usage of SchemaMapper."""
    from data_loader import DataLoader
    
    # Example usage
    mapper = SchemaMapper(debug_mode=True)
    loader = DataLoader(debug_mode=True)
    
    try:
        # Load and process dataset
        df, unique_ratings, _ = loader.process_dataset("data/sample_dataset.csv")
        
        # Get suggested mapping
        suggested_mapping = mapper.suggest_mapping(unique_ratings)
        print(f"Suggested mapping: {suggested_mapping}")
        
        # Apply mapping (using suggested mapping as example)
        df_mapped, metadata = mapper.process_mapping(df, suggested_mapping)
        
        # Save mapped dataset
        df_mapped.to_csv("outputs/mapped_dataset.csv", index=False)
        
        print(f"Mapping completed. Dataset saved with {len(df_mapped)} rows")
        print(f"Mapping metadata: {metadata['mapping_summary']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 