# Schema Mapper - Quick Reference

## Key Methods

| Method | Purpose | Parameters | Returns |
|--------|---------|------------|---------|
| `__init__()` | Initialize mapper | `debug_mode` | None |
| `extract_unique_ratings()` | Get unique ratings | `df` | List[Any] |
| `validate_mapping()` | Validate mapping | `df`, `rating_mapping` | Tuple[bool, List] |
| `apply_mapping()` | Apply mapping | `df`, `rating_mapping` | DataFrame |
| `suggest_mapping()` | Auto-suggest mappings | `unique_ratings` | Dict[Any, int] |
| `process_mapping()` | Complete mapping process | `df`, `rating_mapping`, `output_dir` | Tuple[DataFrame, Dict] |
| `create_mapping_metadata()` | Create metadata | `rating_mapping`, `original_values` | Dict |
| `save_mapping_metadata()` | Save metadata | `metadata`, `output_path` | None |
| `load_mapping_metadata()` | Load metadata | `file_path` | Dict |
| `get_mapping_statistics()` | Get statistics | `df` | Dict |

## Standard Rating Schema

| Rating | Meaning | Description |
|--------|---------|-------------|
| 1 | Match | Clear match between complaint/diagnosis and target condition |
| 0 | Does Not Match | No match between complaint/diagnosis and target condition |
| 2 | Partial Match | Weak or uncertain match (optional intermediate rating) |
| -1 | Unknown | Uncertain or unclear relationship |

## Configuration Options

### Initialization Parameters
```python
mapper = SchemaMapper(
    debug_mode=False      # Enable verbose logging
)
```

### Mapping Dictionary Structure
```python
rating_mapping = {
    1: 1,                # Original → Standardized
    "Match": 1,          # Text → Numeric
    "No Match": 0,       # Text → Numeric
    2: 2,                # Partial Match
    "Unknown": -1        # Unknown values
}
```

## Automatic Mapping Suggestions

### Numeric Values
- `1`, `1.0` → 1 (Match)
- `0`, `0.0` → 0 (Does Not Match)
- `2`, `2.0` → 2 (Partial Match)
- `-1`, `-1.0` → -1 (Unknown)

### Text Values
- `"Match"`, `"Yes"`, `"Strong"` → 1
- `"No Match"`, `"No"`, `"Weak"` → 0
- `"Partial"`, `"Maybe"`, `"Uncertain"` → 2
- Unknown text → -1

## Common Usage Patterns

### Basic Mapping
```python
# Initialize and extract ratings
mapper = SchemaMapper(debug_mode=True)
unique_ratings = mapper.extract_unique_ratings(df)

# Auto-suggest and apply mapping
suggested_mapping = mapper.suggest_mapping(unique_ratings)
df_mapped, metadata = mapper.process_mapping(df, suggested_mapping)
```

### Custom Mapping
```python
# Define custom mapping
custom_mapping = {
    1: 1,           # Keep as Match
    2: 0,           # Treat as Does Not Match
    "Strong": 1,    # Custom text mapping
    "Weak": 2       # Partial Match
}

# Validate and apply
is_valid, unmapped = mapper.validate_mapping(df, custom_mapping)
if is_valid:
    df_mapped, metadata = mapper.process_mapping(df, custom_mapping)
```

### Integration with Data Loader
```python
# Complete pipeline
loader = DataLoader()
mapper = SchemaMapper()

# Load and process
df, unique_ratings, _ = loader.process_dataset("data/dataset.csv")
suggested_mapping = mapper.suggest_mapping(unique_ratings)
df_mapped, metadata = mapper.process_mapping(df, suggested_mapping)
```

## Error Handling Reference

### Exception Types
- `MappingError` - Rating mapping operations fail
- `DataTypeError` - Data type conversion fails

### Error Handling Pattern
```python
try:
    df_mapped, metadata = mapper.process_mapping(df, rating_mapping)
except MappingError as e:
    print(f"Mapping failed: {e}")
except DataTypeError as e:
    print(f"Data type error: {e}")
```

### Validation Pattern
```python
# Check mapping completeness
is_valid, unmapped = mapper.validate_mapping(df, rating_mapping)
if not is_valid:
    print(f"Missing mappings for: {unmapped}")
    # Add missing mappings or use auto-suggestion
```

## Metadata Structure

### Output Metadata
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

## Performance Tips

### Large Datasets
- Use pandas operations for efficient processing
- Metadata creation scales with unique rating count
- File I/O optimized for typical dataset sizes

### Memory Management
- DataFrame operations are memory-efficient
- Metadata storage minimal for typical datasets
- No temporary file creation during processing

## Integration Points

### Input Requirements
- DataFrame with `Expert Rating` column
- Optional custom mapping dictionary
- Existing metadata files for reproducibility

### Output Structure
- Original DataFrame + `Standardized_Rating` column
- Comprehensive metadata JSON file
- Processing logs for debugging

### File Outputs
- Mapped DataFrame CSV file
- Mapping metadata JSON file
- Log file (`schema_mapper.log`)

## Command Line Usage

```bash
# Run demo
python demo_schema_mapper.py

# Run tests
pytest tests/test_schema_mapper.py -v
```

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/test_schema_mapper.py

# Run specific test
pytest tests/test_schema_mapper.py::TestSchemaMapper::test_suggest_mapping
```

### Demo Script
```bash
# Run demo
python demo_schema_mapper.py
```

## Troubleshooting

### Common Issues
1. **Missing Expert Rating column** - Check DataFrame structure
2. **Unmapped values** - Use auto-suggestion or add custom mappings
3. **Data type errors** - Ensure consistent data types in mapping
4. **File not found** - Check output directory permissions

### Debug Mode
```python
mapper = SchemaMapper(debug_mode=True)
# Enables verbose logging to console and file
``` 