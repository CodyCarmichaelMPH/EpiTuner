# Formatter PromptBuilder Flow Diagram

## Overview
This document illustrates the data flow and processing steps in the Formatter PromptBuilder module.

## Main Processing Flow

```mermaid
graph TD
    A[Input: DataFrame + Rating Mapping + Target Topics] --> B[Initialize PromptBuilder]
    B --> C[Validate Row Data]
    C --> D{All Required Fields Present?}
    D -->|Yes| E[Preprocess Text Fields]
    D -->|No| F[Log Warning & Continue]
    F --> E
    E --> G[Create Schema Description]
    G --> H[Build Prompt from Template]
    H --> I[Add Prompt to Dataset]
    I --> J{More Rows?}
    J -->|Yes| C
    J -->|No| K[Generate Metadata]
    K --> L[Save Outputs]
    L --> M[CSV: formatted_dataset.csv]
    L --> N[JSONL: prompts.jsonl]
    L --> O[JSON: formatting_metadata.json]
```

## Detailed Processing Steps

### 1. Data Validation
```mermaid
graph LR
    A[Row Data] --> B[Check Required Fields]
    B --> C{Missing Fields?}
    C -->|No| D[Valid Row]
    C -->|Yes| E[Log Warning]
    E --> F[Mark as Incomplete]
    F --> D
```

### 2. Text Preprocessing
```mermaid
graph LR
    A[Raw Text] --> B{Is Null/NaN?}
    B -->|Yes| C[Replace with 'N/A']
    B -->|No| D[Convert to String]
    D --> E[Strip Whitespace]
    E --> F{Empty String?}
    F -->|Yes| C
    F -->|No| G[Return Cleaned Text]
    C --> G
```

### 3. Prompt Construction
```mermaid
graph TD
    A[Preprocessed Fields] --> B[Load Template]
    B --> C[Format Patient Info]
    C --> D[Format Medical Data]
    D --> E[Format Task Instructions]
    E --> F[Format Rating Schema]
    F --> G[Combine into Final Prompt]
    G --> H[Return Structured Prompt]
```

## Error Handling Flow

```mermaid
graph TD
    A[Processing Error] --> B{Error Type?}
    B -->|Missing Field| C[Log Warning]
    B -->|Invalid Data| D[Log Error]
    B -->|File I/O| E[Raise FormattingError]
    C --> F[Use Placeholder Value]
    D --> G[Skip Row]
    E --> H[Stop Processing]
    F --> I[Continue Processing]
    G --> I
```

## Output Generation Flow

```mermaid
graph TD
    A[Formatted Dataset] --> B[Save CSV]
    A --> C[Save JSONL]
    A --> D[Generate Metadata]
    D --> E[Save JSON Metadata]
    B --> F[formatted_dataset.csv]
    C --> G[prompts.jsonl]
    E --> H[formatting_metadata.json]
```

## Integration Points

### Input Integration
- **data_loader**: Provides cleaned DataFrame with required fields
- **schema_mapper**: Provides rating mapping dictionary

### Output Integration
- **inference_runner**: Uses prompts.jsonl for LLM calls
- **fine_tuner**: Uses prompts.jsonl for model training
- **contextualizer**: Uses prompts for few-shot examples

## Data Transformation Examples

### Input Row
```json
{
  "C_BioSense_ID": "P001",
  "ChiefComplaintOrig": "Fever",
  "Discharge Diagnosis": "Viral infection",
  "Sex": "M",
  "Age": 25,
  "Admit_Reason_Combo": "Fever",
  "Chief_Complaint_Combo": "Fever",
  "Diagnosis_Combo": "Infection",
  "CCDD": "Fever",
  "CCDDCategory": "Viral",
  "TriageNotes": "High fever with chills"
}
```

### Output Prompt
```
Context:
- Patient Info: Age 25, Sex M
- Chief Complaint: Fever
- Discharge Diagnosis: Viral infection
- Admit Reason: Fever
- Combined Complaints: Fever
- Diagnosis Combo: Infection
- CCDD: Fever, Category: Viral
- Triage Notes: High fever with chills

Task:
Based on the above information, evaluate whether this record aligns with the topic(s): respiratory infections.
Use the following rating schema:
- Match (1)
- Does Not Match (0)
- Unknown (-1)

Respond with:
- Numeric rating (from schema).
- Brief rationale (1–3 sentences).
```

## Performance Considerations

### Memory Usage
- Processes rows sequentially to minimize memory footprint
- No large intermediate data structures

### Processing Speed
- Text preprocessing is O(n) where n is number of fields
- Template formatting is O(1) per row
- Overall complexity: O(rows × fields)

### Scalability
- Handles datasets from 5 to 500+ rows efficiently
- Output file sizes scale linearly with input size 