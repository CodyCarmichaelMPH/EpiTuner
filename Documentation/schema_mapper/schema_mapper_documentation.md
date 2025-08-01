# Schema Mapper Module - Complete Documentation

## Overview

The Schema Mapper module (`scripts/schema_mapper.py`) is responsible for transforming non-standard or mixed expert ratings into a standardized schema for consistent model training and inference. It ensures that all expert ratings are mapped to consistent numeric values that can be used reliably in machine learning models.

## Class Structure

### SchemaMapper

The main class that handles all rating mapping operations.

**Location**: `scripts/schema_mapper.py`

**Dependencies**:
- `pandas` for data manipulation
- `logging` for structured logging
- `json` for metadata storage
- `pathlib` for file operations
- `typing` for type hints

## Custom Exception Classes

### MappingError
Raised when rating mapping operations fail due to invalid mappings or missing data.

### DataTypeError
Raised when data type conversion fails during mapping operations.

## Public Methods

### `__init__(debug_mode=False)`

**Purpose**: Initialize the SchemaMapper instance.

**Parameters**:
- `debug_mode` (bool): Enable verbose logging for debugging

**Returns**: None

**Example**:
```python
mapper = SchemaMapper(debug_mode=True)
```

### `extract_unique_ratings(df)`

**Purpose**: Extract unique values from Expert Rating column.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with Expert Rating column

**Returns**: List[Any] - List of unique rating values

**Raises**: MappingError if Expert Rating column is missing

**Example**:
```python
unique_ratings = mapper.extract_unique_ratings(df)
print(f"Found {len(unique_ratings)} unique ratings: {unique_ratings}")
```

### `validate_mapping(df, rating_mapping)`

**Purpose**: Validate that all unique ratings in the dataset have mappings.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with Expert Rating column
- `rating_mapping` (Dict[Any, int]): Dictionary mapping original ratings to standardized values

**Returns**: Tuple[bool, List[Any]] - (is_valid, unmapped_values)

**Example**:
```python
is_valid, unmapped = mapper.validate_mapping(df, rating_mapping)
if not is_valid:
    print(f"Unmapped values: {unmapped}")
```

### `apply_mapping(df, rating_mapping)`

**Purpose**: Apply rating mapping to DataFrame.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with Expert Rating column
- `rating_mapping` (Dict[Any, int]): Dictionary mapping original ratings to standardized values

**Returns**: pd.DataFrame - DataFrame with added Standardized_Rating column

**Raises**: MappingError if mapping fails

**Example**:
```python
df_mapped = mapper.apply_mapping(df, rating_mapping)
print(f"Added Standardized_Rating column with values: {df_mapped['Standardized_Rating'].unique()}")
```

### `create_mapping_metadata(rating_mapping, original_values)`

**Purpose**: Create comprehensive metadata for the mapping operation.

**Parameters**:
- `rating_mapping` (Dict[Any, int]): Dictionary mapping original ratings to standardized values
- `original_values` (List[Any]): List of original unique rating values

**Returns**: Dict[str, Any] - Dictionary containing mapping metadata

**Metadata Structure**:
```python
{
    'original_values': [1, 2, 0, 'Match', 'No Match'],
    'mapped_values': [1, 2, 0, 1, 0],
    'schema': {'1': 1, '2': 2, '0': 0, 'Match': 1, 'No Match': 0},
    'standard_ratings': {'Match': 1, 'Does Not Match': 0, 'Unknown': -1, 'Partial Match': 2},
    'mapping_summary': {
        'total_mappings': 5,
        'unique_mapped_values': [0, 1, 2],
        'mapping_coverage': '5/5 values mapped'
    }
}
```

**Example**:
```python
metadata = mapper.create_mapping_metadata(rating_mapping, original_values)
print(f"Mapping coverage: {metadata['mapping_summary']['mapping_coverage']}")
```

### `save_mapping_metadata(metadata, output_path)`

**Purpose**: Save mapping metadata to file.

**Parameters**:
- `metadata` (Dict[str, Any]): Mapping metadata dictionary
- `output_path` (str): Path to save metadata

**Returns**: None

**Example**:
```python
mapper.save_mapping_metadata(metadata, 'outputs/mapping_metadata.json')
```

### `load_mapping_metadata(file_path)`

**Purpose**: Load mapping metadata from file.

**Parameters**:
- `file_path` (str): Path to metadata file

**Returns**: Dict[str, Any] - Loaded mapping metadata

**Raises**: FileNotFoundError if file doesn't exist

**Example**:
```python
metadata = mapper.load_mapping_metadata('outputs/mapping_metadata.json')
rating_mapping = metadata['schema']
```

### `suggest_mapping(unique_ratings)`

**Purpose**: Automatically suggest mappings for unique rating values.

**Parameters**:
- `unique_ratings` (List[Any]): List of unique rating values

**Returns**: Dict[Any, int] - Suggested mapping dictionary

**Mapping Logic**:
- Numeric values: Direct mapping (1→1, 0→0, 2→2, -1→-1)
- Text values: Pattern matching
  - "Match", "Yes", "Strong" → 1
  - "No Match", "No", "Weak" → 0
  - "Partial", "Maybe", "Uncertain" → 2
  - Unknown values → -1

**Example**:
```python
suggested_mapping = mapper.suggest_mapping(unique_ratings)
print(f"Suggested mapping: {suggested_mapping}")
```

### `process_mapping(df, rating_mapping, output_dir="outputs")`

**Purpose**: Complete mapping process with metadata creation and saving.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with Expert Rating column
- `rating_mapping` (Dict[Any, int]): Dictionary mapping original ratings to standardized values
- `output_dir` (str): Directory to save output files

**Returns**: Tuple[pd.DataFrame, Dict[str, Any]] - (mapped_dataframe, metadata)

**Behavior**:
- Validates mapping completeness
- Applies mapping to DataFrame
- Creates comprehensive metadata
- Saves metadata to file
- Returns mapped DataFrame and metadata

**Example**:
```python
df_mapped, metadata = mapper.process_mapping(df, rating_mapping)
print(f"Processing complete. Coverage: {metadata['mapping_summary']['mapping_coverage']}")
```

### `get_mapping_statistics(df)`

**Purpose**: Calculate statistics on the mapping results.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with Standardized_Rating column

**Returns**: Dict[str, Any] - Statistics dictionary

**Statistics Include**:
- Total records processed
- Distribution of standardized ratings
- Mapping coverage percentage
- Unique value counts

**Example**:
```python
stats = mapper.get_mapping_statistics(df_mapped)
print(f"Rating distribution: {stats['rating_distribution']}")
```

## Private Methods

### `_setup_logging()`

**Purpose**: Setup logging configuration.

**Behavior**:
- Configures logging level based on debug_mode
- Sets up both console and file handlers
- Creates logger instance

### `_validate_dataframe(df)`

**Purpose**: Validate DataFrame structure and required columns.

**Parameters**:
- `df` (pd.DataFrame): DataFrame to validate

**Returns**: bool - True if valid, False otherwise

**Validation Checks**:
- DataFrame is not empty
- Expert Rating column exists
- Expert Rating column has data

### `_convert_to_standard_rating(value)`

**Purpose**: Convert a single value to standardized rating.

**Parameters**:
- `value` (Any): Value to convert

**Returns**: int - Standardized rating value

**Conversion Logic**:
- Direct numeric mapping for known values
- Pattern matching for text values
- Default to -1 for unknown values

### `_create_mapping_summary(rating_mapping, original_values)`

**Purpose**: Create summary statistics for mapping operation.

**Parameters**:
- `rating_mapping` (Dict[Any, int]): Mapping dictionary
- `original_values` (List[Any]): Original unique values

**Returns**: Dict[str, Any] - Summary statistics

## Usage Examples

### Basic Usage

```python
from scripts.schema_mapper import SchemaMapper

# Initialize mapper
mapper = SchemaMapper(debug_mode=True)

# Extract unique ratings
unique_ratings = mapper.extract_unique_ratings(df)

# Generate automatic mapping
suggested_mapping = mapper.suggest_mapping(unique_ratings)

# Apply mapping
df_mapped, metadata = mapper.process_mapping(df, suggested_mapping)
```

### Custom Mapping

```python
# Define custom mapping
custom_mapping = {
    1: 1,           # Keep as Match
    2: 0,           # Treat as Does Not Match
    0: 0,           # Keep as Does Not Match
    "Strong": 1,    # Custom text mapping
    "Weak": 2       # Partial Match
}

# Validate mapping
is_valid, unmapped = mapper.validate_mapping(df, custom_mapping)
if not is_valid:
    print(f"Missing mappings for: {unmapped}")

# Apply custom mapping
df_mapped, metadata = mapper.process_mapping(df, custom_mapping)
```

### Integration with Data Loader

```python
from scripts.data_loader import DataLoader
from scripts.schema_mapper import SchemaMapper

# Load and process dataset
loader = DataLoader()
df, unique_ratings, _ = loader.process_dataset("data/dataset.csv")

# Create mapper and apply mapping
mapper = SchemaMapper()
suggested_mapping = mapper.suggest_mapping(unique_ratings)
df_mapped, metadata = mapper.process_mapping(df, suggested_mapping)

# Save results
df_mapped.to_csv("outputs/mapped_dataset.csv", index=False)
```

### Error Handling

```python
try:
    df_mapped, metadata = mapper.process_mapping(df, rating_mapping)
except MappingError as e:
    print(f"Mapping failed: {e}")
except DataTypeError as e:
    print(f"Data type error: {e}")
```

## Performance Characteristics

### Time Complexity
- **Extract unique ratings**: O(n) where n is number of rows
- **Apply mapping**: O(n) where n is number of rows
- **Validate mapping**: O(m) where m is number of unique ratings
- **Metadata creation**: O(m) where m is number of unique ratings

### Space Complexity
- **Memory usage**: O(n) for DataFrame operations
- **Storage**: O(m) for metadata where m is number of unique ratings

### Performance Considerations
- Large datasets processed efficiently with pandas operations
- Metadata creation scales linearly with unique rating count
- File I/O operations optimized for typical dataset sizes

## Integration Points

### Input Dependencies
- DataFrame with Expert Rating column from data loader
- Optional custom mapping dictionary from user input
- Existing metadata files for reproducibility

### Output Dependencies
- Mapped DataFrame for formatter/prompt builder
- Metadata files for tracking and reproducibility
- Logs for debugging and audit purposes

### Error Handling Integration
- Logging integration with debugging logger
- Error propagation to calling modules
- Graceful degradation for partial failures 