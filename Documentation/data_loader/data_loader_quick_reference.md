# Data Loader Quick Reference Guide

## Quick Start

```python
from scripts.data_loader import DataLoader

# Initialize
loader = DataLoader(debug_mode=True)

# Process dataset
df, unique_ratings, metadata = loader.process_dataset("data/sample.csv")

# Save results
loader.save_processed_dataset(df, "outputs/processed.csv")
```

## Key Methods

| Method | Purpose | Parameters | Returns |
|--------|---------|------------|---------|
| `process_dataset()` | Main entry point | `file_path`, `rating_mapping=None` | `(df, ratings, metadata)` |
| `load_dataset()` | Load CSV file | `file_path` | `pd.DataFrame` |
| `validate_schema()` | Check required fields | `df` | `(bool, missing, suggestions)` |
| `clean_dataset()` | Clean and standardize | `df` | `pd.DataFrame` |
| `extract_unique_ratings()` | Get unique ratings | `df` | `List[Any]` |
| `create_context_block()` | Merge text fields | `df` | `pd.DataFrame` |
| `add_standardized_rating_column()` | Map ratings | `df`, `rating_mapping` | `pd.DataFrame` |

## Required Dataset Fields

```
C_BioSense_ID          (string)  - Unique patient identifier
ChiefComplaintOrig     (string)  - Original chief complaint
Discharge Diagnosis    (string)  - Final diagnosis
Sex                    (string)  - Patient sex
Age                    (integer) - Patient age
Admit_Reason_Combo     (string)  - Admission reason
Chief_Complaint_Combo  (string)  - Combined chief complaint
Diagnosis_Combo        (string)  - Combined diagnosis
CCDD                   (string)  - CCDD code
CCDDCategory           (string)  - CCDD category
TriageNotes            (string)  - Triage notes
Expert Rating          (int/str) - Expert assessment rating
```

**Optional**: `Rationale of Rating` (string)

## Rating Standardization

```python
# Standard mapping
rating_mapping = {
    1: 1,    # 1 = Match
    2: 0,    # 2 = Does Not Match
    0: -1    # 0 = Unknown
}
```

## Output Columns

**Original columns** + 2 new columns:
- `Context_Block`: Merged text context for LLM
- `Standardized_Rating`: Mapped rating values

## Context Block Format

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

## Error Handling

| Exception | When Raised | Solution |
|-----------|-------------|----------|
| `FileError` | File not found/unreadable | Check file path and permissions |
| `SchemaError` | Missing required fields | Add missing columns or use column mapping |
| `DataTypeError` | Type conversion fails | Check data format in problematic columns |

## Common Usage Patterns

### Basic Processing
```python
loader = DataLoader()
df, ratings, meta = loader.process_dataset("data.csv")
```

### With Rating Mapping
```python
mapping = {1: 1, 2: 0, 0: -1}
df, ratings, meta = loader.process_dataset("data.csv", mapping)
```

### Save Results
```python
loader.save_processed_dataset(df, "output.csv")
loader.save_metadata(meta, "metadata.json")
```

### Error Handling
```python
try:
    df, ratings, meta = loader.process_dataset("data.csv")
except SchemaError as e:
    print(f"Schema issue: {e}")
except FileError as e:
    print(f"File issue: {e}")
```

## Debug Mode

```python
loader = DataLoader(debug_mode=True)  # Verbose logging
```

Logs are written to:
- Console output
- `data_loader.log` file

## Performance Notes

- **Memory**: Efficient for 5-500+ row datasets
- **Speed**: Optimized pandas operations
- **Scalability**: Handles larger datasets with appropriate memory management 