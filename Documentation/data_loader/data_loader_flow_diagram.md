# Data Loader Flow Diagram

## Overview Flow

```
CSV File Input
      ↓
[1] Load Dataset
      ↓
[2] Validate Schema
      ↓
[3] Clean Dataset
      ↓
[4] Extract Unique Ratings
      ↓
[5] Create Context Blocks
      ↓
[6] Add Standardized Ratings (Optional)
      ↓
Processed Dataset Output
```

## Detailed Process Flow

### 1. Load Dataset
```
File Path Input
      ↓
Check File Exists
      ↓
Read CSV with pandas
      ↓
Handle Errors:
  ├─ File not found → FileError
  ├─ Empty file → FileError
  ├─ Parse error → FileError
  └─ Success → DataFrame
```

### 2. Validate Schema
```
DataFrame Input
      ↓
Check Required Fields (12 fields)
      ↓
For Each Missing Field:
  ├─ Find potential matches
  ├─ Check abbreviations
  └─ Build suggestions
      ↓
Return: (is_valid, missing_fields, suggestions)
```

### 3. Clean Dataset
```
Raw DataFrame
      ↓
Handle Missing Values:
  ├─ Text fields → empty string
  ├─ Age → 0
  └─ Other strings → empty string
      ↓
Cast Data Types:
  ├─ Age → integer
  └─ Expert Rating → integer/string
      ↓
Drop Rows Missing C_BioSense_ID
      ↓
Cleaned DataFrame
```

### 4. Extract Unique Ratings
```
DataFrame with Expert Rating Column
      ↓
Get unique values: df['Expert Rating'].unique()
      ↓
Convert to list
      ↓
Log unique ratings found
      ↓
List of unique ratings
```

### 5. Create Context Blocks
```
DataFrame with Text Fields
      ↓
For Each Row:
  ├─ Extract patient info (Age, Sex)
  ├─ Extract medical details (8 text fields)
  └─ Format as structured text
      ↓
Add Context_Block Column
      ↓
DataFrame with Context Blocks
```

### 6. Add Standardized Ratings (Optional)
```
DataFrame + Rating Mapping
      ↓
For Each Rating:
  ├─ Look up in mapping
  ├─ Apply standardized value
  └─ Default to -1 if unmapped
      ↓
Add Standardized_Rating Column
      ↓
Log mapping summary
      ↓
Final DataFrame
```

## Error Handling Flow

### Schema Validation Errors
```
Schema Validation Fails
      ↓
Identify Missing Fields
      ↓
Generate Column Suggestions
      ↓
Raise SchemaError with details
      ↓
User can:
  ├─ Add missing columns
  ├─ Use column mapping
  └─ Fix field names
```

### File Operation Errors
```
File Operation Fails
      ↓
Identify Error Type:
  ├─ File not found
  ├─ Permission denied
  ├─ Empty file
  └─ Parse error
      ↓
Raise FileError with details
      ↓
User can:
  ├─ Check file path
  ├─ Fix permissions
  └─ Validate file format
```

### Data Type Errors
```
Type Conversion Fails
      ↓
Log Warning
      ↓
Handle Gracefully:
  ├─ Convert to string
  ├─ Fill with default
  └─ Continue processing
      ↓
Data continues with mixed types
```

## Context Block Creation Flow

### Text Field Processing
```
Input Text Fields:
  ├─ ChiefComplaintOrig
  ├─ Discharge Diagnosis
  ├─ TriageNotes
  ├─ Admit_Reason_Combo
  ├─ Chief_Complaint_Combo
  ├─ Diagnosis_Combo
  ├─ CCDD
  └─ CCDDCategory
      ↓
For Each Field:
  ├─ Check if not empty
  ├─ Format as "Field: Value"
  └─ Add to context list
      ↓
Join with newlines
      ↓
Structured Context Block
```

### Patient Info Processing
```
Input Patient Fields:
  ├─ Age
  └─ Sex
      ↓
Format as:
  ├─ "Age: [age]" (if age > 0)
  └─ "Sex: [sex]" (if not empty)
      ↓
Add to context block
```

## Rating Standardization Flow

### Mapping Process
```
Input: Original Rating Values
      ↓
User-Defined Mapping:
  ├─ 1 → 1 (Match)
  ├─ 2 → 0 (Does Not Match)
  └─ 0 → -1 (Unknown)
      ↓
Apply to Each Rating:
  ├─ Look up in mapping
  ├─ Apply standardized value
  └─ Default to -1 if not found
      ↓
Output: Standardized Rating Values
```

### Validation Process
```
Rating Mapping Applied
      ↓
Count Each Standardized Value
      ↓
Log Summary:
  ├─ Original → Standardized counts
  └─ Unmapped values (if any)
      ↓
Return Mapping Statistics
```

## Output Generation Flow

### Dataset Output
```
Processed DataFrame
      ↓
Include:
  ├─ All original columns (13)
  ├─ Context_Block column
  └─ Standardized_Rating column
      ↓
Save to CSV
      ↓
Processed Dataset File
```

### Metadata Output
```
Processing Information
      ↓
Collect:
  ├─ Original dataset shape
  ├─ Cleaned dataset shape
  ├─ Unique ratings found
  ├─ Rating mapping used
  ├─ Missing fields
  └─ Column suggestions
      ↓
Save to JSON
      ↓
Metadata File
```

## Performance Considerations

### Memory Management
```
Large Dataset Input
      ↓
Stream Processing:
  ├─ Load in chunks if needed
  ├─ Process row by row
  └─ Release memory after use
      ↓
Optimized Memory Usage
```

### Speed Optimization
```
Dataset Processing
      ↓
Efficient Operations:
  ├─ Vectorized pandas operations
  ├─ Minimal DataFrame copies
  └─ Optimized string operations
      ↓
Fast Processing Time
```

## Debug Mode Flow

### Logging Configuration
```
Debug Mode Enabled
      ↓
Set Log Level to DEBUG
      ↓
Configure Handlers:
  ├─ Console output
  └─ File output (data_loader.log)
      ↓
Detailed Logging Active
```

### Log Output
```
Processing Steps
      ↓
Log Each Step:
  ├─ Method entry/exit
  ├─ Data transformations
  ├─ Error conditions
  └─ Performance metrics
      ↓
Comprehensive Debug Information
``` 