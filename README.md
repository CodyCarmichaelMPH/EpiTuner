# EpiTuner - Ollama Fine-Tuning and Evaluation Suite

A comprehensive Python suite for fine-tuning and evaluating Ollama models with epidemiological data, featuring schema-guided evaluation and context-aware inference.

## System Flowcharts

### **Interactive HTML Flowchart** (Recommended)
View the complete interactive system flowchart:
- **Main Overview**: Click on any section to see detailed sub-processes
- **Inputs & Outputs**: Each section shows what it needs and produces
- **Detailed Steps**: Sub-flowcharts show the complete process for each component

```bash
# Open in your browser
start flowcharts.html
# or double-click the file
```

### **Detailed Mermaid Flowcharts**
For advanced viewing with Mermaid support:
- **[Master Flowchart](Documentation/master_flowchart.md)**: Complete pipeline overview
- **[Flowchart Index](Documentation/flowchart_index.md)**: All available flowcharts

## Project Overview

EpiTuner is designed to handle medical datasets with expert ratings, providing tools for:
- Dataset loading and validation
- Schema mapping and normalization
- Context bundling for LLM training
- Fine-tuning and inference with Ollama
- Comprehensive evaluation and debugging

## Project Structure

```
EpiTuner/
├── Docs/
│   ├── master_functional_overview.txt
│   ├── .cursorrules
│   └── queries/
│       ├── 01_data_loader_query.txt
│       ├── 02_schema_mapper_query.txt
│       ├── 03_formatter_promptbuilder_query.txt
│       ├── 04_inference_runner_query.txt
│       ├── 05_fine_tuner_query.txt
│       ├── 06_contextualizer_query.txt
│       ├── 07_debugging_logger_query.txt
│       ├── 08_gui_query.txt
│       └── 09_formatter_query.txt
├── scripts/
│   ├── data_loader.py          # Implemented
│   ├── schema_mapper.py        # Implemented
│   ├── formatter_promptbuilder.py # Implemented
│   ├── inference_runner.py     # Next
│   ├── fine_tuner.py           # Planned
│   ├── contextualizer.py       # Planned
│   ├── debugging_logger.py     # Planned
│   └── utils.py                # Planned
├── gui/
│   └── app.py                  # Planned
├── tests/
│   ├── test_data_loader.py     # Implemented
│   ├── test_schema_mapper.py   # Implemented
│   ├── test_formatter_promptbuilder.py # Implemented
│   ├── test_inference_runner.py # Next
│   └── test_fine_tuner.py      # Planned
├── data/
│   └── sample_dataset.csv      # Created
├── outputs/                    # Generated during processing
├── requirements.txt            # Created
└── README.md                   # This file
```

## Current Status: Complete EpiTuner Suite with Context Summary and Dynamic Model Selection

### What's Implemented

1. **Data Loader Module** (`scripts/data_loader.py`)
   - CSV file loading with error handling
   - Schema validation against expected fields
   - Data cleaning and type casting
   - Missing value handling
   - Context block creation for LLM training
   - Rating standardization preparation
   - Comprehensive logging and debugging

2. **Schema Mapper Module** (`scripts/schema_mapper.py`)
   - Rating standardization and normalization
   - Automatic mapping suggestions based on common patterns
   - Custom user-defined mapping support
   - Validation and error handling
   - Metadata storage for reproducibility
   - Integration with data loader

3. **Formatter Module** (`scripts/formatter.py`) - **ENHANCED**
   - **Context Summary Approach**: Extract key patterns from training data
   - **Pattern-Based Evaluation**: "Look for respiratory symptoms: fever, cough, etc."
   - **Transfer Learning**: Patterns transfer to new case evaluation
   - **Traditional Prompts**: Individual case prompts for fine-tuning
   - **Smart Context Creation**: Automatic pattern extraction and summarization
   - **Multiple Output Formats**: CSV, JSONL, metadata with context summaries

4. **GUI Interface** (`gui/epituner_gui.py`) - **NEW**
   - **Clean Interface**: Modern, intuitive design
   - **Dynamic Model Selection**: Dropdown with all available Ollama models
   - **Context Summary Integration**: Default prompt formatting approach
   - **Low Power Mode**: Optimized for tablet/limited hardware
   - **Step-by-Step Workflow**: Intuitive navigation through all processes

5. **Complete Integration Pipeline**
   - Seamless data flow from raw CSV to LLM-ready prompts
   - Context Summary approach for efficient pattern-based evaluation
   - Dynamic model selection and configuration
   - Comprehensive integration testing
   - Performance optimization for large datasets
   - Complete error handling and logging

5. **Unit Tests** 
   - `tests/test_data_loader.py` (25 tests, all passing)
   - `tests/test_schema_mapper.py` (14 tests, all passing)
   - `tests/test_formatter_promptbuilder.py` (16 tests, all passing)
   - Integration tests (3 test suites, all passing)

5. **Sample Dataset** (`data/sample_dataset.csv`)
   - 10-row sample with all required fields
   - Varied expert ratings (0, 1, 2)
   - Realistic medical scenarios

6. **Comprehensive Documentation**
   - **Data Loader**: `Documentation/data_loader/` (4 files)
   - **Schema Mapper**: `Documentation/schema_mapper/` (1 file)
   - **Formatter PromptBuilder**: `Documentation/formatter_promptbuilder/` (3 files)
   - **Integration Guide**: `Documentation/integration_guide.md`
   - **Complete function-by-function breakdowns**
   - **Quick reference guides**
   - **Process flow diagrams**
   - **Implementation summaries**

### Expected Dataset Schema

The data loader expects CSV files with the following schema:

**Required Fields:**
- `C_BioSense_ID` (string) - Unique patient identifier
- `ChiefComplaintOrig` (string) - Original chief complaint
- `Discharge Diagnosis` (string) - Final diagnosis
- `Sex` (string) - Patient sex
- `Age` (integer) - Patient age
- `Admit_Reason_Combo` (string) - Admission reason
- `Chief_Complaint_Combo` (string) - Combined chief complaint
- `Diagnosis_Combo` (string) - Combined diagnosis
- `CCDD` (string) - CCDD code
- `CCDDCategory` (string) - CCDD category
- `TriageNotes` (string) - Triage notes
- `Expert Rating` (integer/string) - Expert assessment rating

**Optional Fields:**
- `Rationale of Rating` (string) - Expert's reasoning

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EpiTuner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Data Loader Usage

```python
from scripts.data_loader import DataLoader

# Initialize data loader
loader = DataLoader(debug_mode=True)

# Process dataset
df, unique_ratings, metadata = loader.process_dataset("data/sample_dataset.csv")

# Print results
print(f"Processed {len(df)} rows")
print(f"Unique ratings: {unique_ratings}")
print(f"Metadata: {metadata}")
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
    "data/sample_dataset.csv", 
    rating_mapping=rating_mapping
)

# Save results
loader.save_processed_dataset(df, "outputs/processed_dataset.csv")
loader.save_metadata(metadata, "outputs/processing_metadata.json")
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_loader.py

# Run with verbose output
pytest tests/test_data_loader.py -v

# Run with coverage
pytest tests/test_data_loader.py --cov=scripts.data_loader
```

## Data Loader Features

### 1. Schema Validation
- Validates presence of all required fields
- Suggests column mappings for missing fields
- Handles common field name variations

### 2. Data Cleaning
- Handles missing values appropriately
- Type casting (Age to integer, Expert Rating handling)
- Drops rows with missing C_BioSense_ID

### 3. Context Block Creation
- Merges key text fields into structured context
- Format: Patient info + Medical details
- Ready for LLM training and inference

### 4. Rating Standardization
- Extracts unique rating values
- Supports user-defined mapping to standard values
- Handles mixed rating types gracefully

### 5. Error Handling
- FileError: File not found or unreadable
- SchemaError: Missing required fields
- DataTypeError: Type conversion failures
- Comprehensive logging for debugging

## Output Format

The processed dataset includes:
- All original columns
- `Context_Block`: Merged text context for LLM
- `Standardized_Rating`: Mapped rating values (-1 for unmapped)

Example context block:
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

## Next Steps

1. **Schema Mapper Implementation** (Next)
   - Interactive rating mapping
   - GUI for user input
   - Validation and error handling

2. **Formatter & Prompt Builder**
   - Context optimization
   - Prompt template creation
   - Training data preparation

3. **Inference Runner**
   - Ollama model integration
   - Batch processing
   - Result formatting

4. **Fine-Tuner**
   - Model fine-tuning pipeline
   - .gguf file generation
   - Training validation

## Contributing

1. Follow the coding standards in `Docs/.cursorrules`
2. Write unit tests for all new functionality
3. Ensure 90%+ test coverage
4. Update documentation as needed

## License

[Add your license information here]

## Support

For issues and questions, please refer to the documentation in the `Docs/` folder or create an issue in the repository. 