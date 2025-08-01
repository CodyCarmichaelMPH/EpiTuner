# Data Loader Module Documentation

## Overview

The `DataLoader` class is the core component responsible for loading, validating, cleaning, and preparing medical datasets for LLM training and inference. It handles CSV files with epidemiological data and transforms them into a format suitable for machine learning workflows.

**File Location**: `scripts/data_loader.py`

## Class Structure

```python
class DataLoader:
    """
    Handles dataset loading, validation, cleaning, and context preparation.
    """
```

## Class Constants

### REQUIRED_FIELDS
```python
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
```
**Purpose**: Defines the 12 required fields that must be present in the input dataset.

### OPTIONAL_FIELDS
```python
OPTIONAL_FIELDS = [
    'Rationale of Rating'
]
```
**Purpose**: Defines fields that are not required but may be present.

### TEXT_FIELDS_FOR_CONTEXT
```python
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
```
**Purpose**: Defines which text fields should be merged into the context block for LLM training.

## Custom Exception Classes

### SchemaError
```python
class SchemaError(Exception):
    """Raised when dataset schema validation fails."""
```
**Purpose**: Custom exception for schema validation failures.

### DataTypeError
```python
class DataTypeError(Exception):
    """Raised when data type conversion fails."""
```
**Purpose**: Custom exception for data type conversion failures.

### FileError
```python
class FileError(Exception):
    """Raised when file operations fail."""
```
**Purpose**: Custom exception for file operation failures.

## Constructor

### `__init__(self, debug_mode: bool = False)`

**Purpose**: Initializes the DataLoader instance and sets up logging.

**Parameters**:
- `debug_mode` (bool): Enable verbose logging for debugging. Default: False

**Behavior**:
- Sets the debug_mode attribute
- Calls `_setup_logging()` to configure logging

**Example**:
```python
loader = DataLoader(debug_mode=True)  # Enable debug logging
```

## Private Methods

### `_setup_logging(self)`

**Purpose**: Configures logging with appropriate handlers and format.

**Behavior**:
- Sets log level based on debug_mode (DEBUG if True, INFO if False)
- Configures console and file handlers
- Creates logger instance for the class

**Log Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

**Output**: Logs are written to both console and `data_loader.log` file

### `_find_potential_column_matches(self, target_field: str, available_columns: List[str]) -> List[str]`

**Purpose**: Finds potential column matches for missing fields by comparing field names.

**Parameters**:
- `target_field` (str): The field name we're looking for
- `available_columns` (List[str]): List of available column names

**Returns**: List[str] - Up to 3 potential matches

**Algorithm**:
1. Normalizes field names (lowercase, remove spaces/underscores)
2. Checks for exact matches
3. Checks for substring matches
4. Checks for common abbreviations
5. Returns top 3 matches

**Example**:
```python
matches = loader._find_potential_column_matches("Expert Rating", ["rating", "expert_score", "assessment"])
# Returns: ["rating", "expert_score"]
```

### `_check_abbreviations(self, target_field: str, column: str) -> bool`

**Purpose**: Checks for common abbreviations between target field and column name.

**Parameters**:
- `target_field` (str): Target field name
- `column` (str): Column name to check

**Returns**: bool - True if abbreviation match found

**Supported Abbreviations**:
- `C_BioSense_ID`: ['id', 'biosense', 'patient_id']
- `ChiefComplaintOrig`: ['chief', 'complaint', 'cc']
- `Discharge Diagnosis`: ['diagnosis', 'discharge', 'dx']
- `Expert Rating`: ['rating', 'expert', 'score']
- `Rationale of Rating`: ['rationale', 'reason', 'explanation']

### `_handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Handles missing values in the dataset by filling them appropriately.

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame

**Returns**: pd.DataFrame - DataFrame with missing values filled

**Behavior**:
- Text fields: Filled with empty string
- Age field: Filled with 0 (will be handled in type casting)
- String fields (Sex, CCDD, CCDDCategory): Filled with empty string

**Example**:
```python
# Before: Age column has NaN values
# After: Age column has 0 values
df_clean = loader._handle_missing_values(df)
```

### `_cast_data_types(self, df: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Casts data types appropriately for Age and Expert Rating fields.

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame

**Returns**: pd.DataFrame - DataFrame with proper data types

**Behavior**:
- **Age**: Converts to integer using `pd.to_numeric()` with error handling
- **Expert Rating**: 
  - Attempts to convert to numeric
  - If non-numeric values found, keeps them as strings
  - Logs warning for non-numeric values

**Example**:
```python
# Age: "25" -> 25 (integer)
# Expert Rating: "1" -> 1 (integer), "High" -> "High" (string)
df_typed = loader._cast_data_types(df)
```

## Public Methods

### `load_dataset(self, file_path: str) -> pd.DataFrame`

**Purpose**: Loads a CSV dataset from file with comprehensive error handling.

**Parameters**:
- `file_path` (str): Path to the CSV file

**Returns**: pd.DataFrame - Loaded dataset

**Error Handling**:
- `FileError`: File not found
- `FileError`: Empty file
- `FileError`: CSV parsing errors
- `FileError`: Unexpected errors

**Example**:
```python
try:
    df = loader.load_dataset("data/sample_dataset.csv")
    print(f"Loaded {len(df)} rows")
except FileError as e:
    print(f"Error loading file: {e}")
```

### `validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, str]]`

**Purpose**: Validates dataset schema against expected fields and suggests mappings.

**Parameters**:
- `df` (pd.DataFrame): DataFrame to validate

**Returns**: Tuple[bool, List[str], Dict[str, str]]
- bool: True if schema is valid
- List[str]: Missing field names
- Dict[str, str]: Column mapping suggestions

**Behavior**:
1. Checks for all required fields
2. For missing fields, suggests potential column matches
3. Returns validation status and suggestions

**Example**:
```python
is_valid, missing_fields, suggestions = loader.validate_schema(df)
if not is_valid:
    print(f"Missing fields: {missing_fields}")
    print(f"Suggestions: {suggestions}")
```

### `clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Cleans and standardizes the dataset.

**Parameters**:
- `df` (pd.DataFrame): Raw DataFrame

**Returns**: pd.DataFrame - Cleaned DataFrame

**Process**:
1. Handles missing values via `_handle_missing_values()`
2. Casts data types via `_cast_data_types()`
3. Drops rows with missing C_BioSense_ID
4. Logs cleaning statistics

**Example**:
```python
df_clean = loader.clean_dataset(df)
print(f"Cleaned dataset shape: {df_clean.shape}")
```

### `extract_unique_ratings(self, df: pd.DataFrame) -> List[Any]`

**Purpose**: Extracts unique values from the Expert Rating column.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with Expert Rating column

**Returns**: List[Any] - List of unique rating values

**Error Handling**:
- `SchemaError`: If Expert Rating column not found

**Example**:
```python
unique_ratings = loader.extract_unique_ratings(df)
print(f"Unique ratings: {unique_ratings}")
# Output: [1, 2, 0]
```

### `create_context_block(self, df: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Creates context blocks by merging key text fields for LLM training.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with text fields

**Returns**: pd.DataFrame - DataFrame with added Context_Block column

**Process**:
1. Defines `create_context_string()` function for each row
2. Merges patient info (Age, Sex) and medical details
3. Formats as structured text for LLM consumption
4. Adds Context_Block column to DataFrame

**Context Format**:
```
Age: 25
Sex: M
Chief Complaint: Fever
Discharge Diagnosis: Viral infection
Triage Notes: High fever with chills
Admit Reason: Fever
Chief Complaint Combo: Fever
Diagnosis Combo: Infection
CCDD: Fever
Category: Viral
```

**Example**:
```python
df_with_context = loader.create_context_block(df)
print(df_with_context['Context_Block'].iloc[0])
```

### `add_standardized_rating_column(self, df: pd.DataFrame, rating_mapping: Dict[Any, int]) -> pd.DataFrame`

**Purpose**: Adds standardized rating column based on user-defined mapping.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with Expert Rating column
- `rating_mapping` (Dict[Any, int]): Mapping from original to standardized values

**Returns**: pd.DataFrame - DataFrame with added Standardized_Rating column

**Behavior**:
- Maps original ratings to standardized values
- Unmapped values default to -1 (Unknown)
- Logs mapping summary statistics

**Example**:
```python
rating_mapping = {1: 1, 2: 0, 0: -1}  # 1=Match, 2=Does Not Match, 0=Unknown
df_mapped = loader.add_standardized_rating_column(df, rating_mapping)
```

### `process_dataset(self, file_path: str, rating_mapping: Optional[Dict[Any, int]] = None) -> Tuple[pd.DataFrame, List[Any], Dict[str, Any]]`

**Purpose**: Complete dataset processing pipeline - the main entry point.

**Parameters**:
- `file_path` (str): Path to CSV file
- `rating_mapping` (Optional[Dict[Any, int]]): Optional rating standardization mapping

**Returns**: Tuple[pd.DataFrame, List[Any], Dict[str, Any]]
- pd.DataFrame: Processed dataset
- List[Any]: Unique rating values
- Dict[str, Any]: Processing metadata

**Pipeline Steps**:
1. Load dataset via `load_dataset()`
2. Validate schema via `validate_schema()`
3. Clean dataset via `clean_dataset()`
4. Extract unique ratings via `extract_unique_ratings()`
5. Create context blocks via `create_context_block()`
6. Add standardized rating column (if mapping provided)

**Metadata Includes**:
- Original and cleaned dataset shapes
- Unique ratings found
- Rating mapping used
- Missing fields and suggestions

**Example**:
```python
df, unique_ratings, metadata = loader.process_dataset("data/sample.csv")
print(f"Processed {len(df)} rows")
print(f"Found ratings: {unique_ratings}")
```

### `save_processed_dataset(self, df: pd.DataFrame, output_path: str) -> None`

**Purpose**: Saves processed dataset to CSV file.

**Parameters**:
- `df` (pd.DataFrame): Processed DataFrame to save
- `output_path` (str): Output file path

**Error Handling**:
- `FileError`: If save operation fails

**Example**:
```python
loader.save_processed_dataset(df, "outputs/processed_dataset.csv")
```

### `save_metadata(self, metadata: Dict[str, Any], output_path: str) -> None`

**Purpose**: Saves processing metadata to JSON file.

**Parameters**:
- `metadata` (Dict[str, Any]): Processing metadata
- `output_path` (str): Output file path

**Error Handling**:
- `FileError`: If save operation fails

**Example**:
```python
loader.save_metadata(metadata, "outputs/processing_metadata.json")
```

## Usage Examples

### Basic Usage
```python
from scripts.data_loader import DataLoader

# Initialize
loader = DataLoader(debug_mode=True)

# Process dataset
df, unique_ratings, metadata = loader.process_dataset("data/sample.csv")

# Save results
loader.save_processed_dataset(df, "outputs/processed.csv")
loader.save_metadata(metadata, "outputs/metadata.json")
```

### With Rating Standardization
```python
# Define rating mapping
rating_mapping = {
    1: 1,    # 1 = Match
    2: 0,    # 2 = Does Not Match
    0: -1    # 0 = Unknown
}

# Process with mapping
df, unique_ratings, metadata = loader.process_dataset(
    "data/sample.csv", 
    rating_mapping=rating_mapping
)
```

### Error Handling
```python
try:
    df, unique_ratings, metadata = loader.process_dataset("data/sample.csv")
except SchemaError as e:
    print(f"Schema validation failed: {e}")
except FileError as e:
    print(f"File operation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Output Format

The processed dataset includes:
- **All original columns** (13 fields)
- **Context_Block**: Merged text context for LLM training
- **Standardized_Rating**: Mapped rating values (-1 for unmapped)

### Context Block Structure
```
Age: [age]
Sex: [sex]
Chief Complaint: [chief_complaint]
Discharge Diagnosis: [diagnosis]
Triage Notes: [triage_notes]
Admit Reason: [admit_reason]
Chief Complaint Combo: [chief_combo]
Diagnosis Combo: [diagnosis_combo]
CCDD: [ccdd]
Category: [category]
```

### Rating Standardization
- **1**: Match
- **0**: Does Not Match
- **-1**: Unknown (unmapped values)

## Performance Characteristics

- **Memory**: Efficient pandas operations, minimal memory overhead
- **Speed**: Optimized for datasets of 5-500+ rows
- **Scalability**: Handles larger datasets with appropriate memory management
- **Logging**: Configurable verbosity for debugging

## Dependencies

- **pandas**: Data manipulation and CSV handling
- **logging**: Debug and error logging
- **pathlib**: File path operations
- **json**: Metadata serialization
- **typing**: Type hints for better code documentation

## Testing

The module includes comprehensive unit tests covering:
- All public methods
- Error conditions
- Edge cases
- Data type handling
- Schema validation
- File operations

Run tests with:
```bash
pytest tests/test_data_loader.py -v
``` 