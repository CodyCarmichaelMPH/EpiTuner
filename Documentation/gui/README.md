# Streamlit GUI Module

## Overview

The Streamlit GUI module provides a comprehensive web-based interface for the entire EpiTuner suite, enabling non-technical users to perform all operations through an intuitive, step-by-step workflow.

## Purpose

- Provide a user-friendly web interface for all EpiTuner operations
- Guide users through the complete data processing pipeline
- Enable interactive data upload and validation
- Support visual schema mapping and rating standardization
- Provide real-time prompt formatting and review
- Execute model operations with progress monitoring
- Visualize results and enable data export

## Key Features

### Step-by-Step Workflow
- **Step 1**: Data Upload & Validation
- **Step 2**: Schema Mapping & Rating Standardization
- **Step 3**: Prompt Formatting & Preparation
- **Step 4**: Model Operations (Fine-tuning, Inference, Contextualization)
- **Step 5**: Results Visualization & Export

### Interactive Data Management
- File upload with automatic validation
- Real-time data preview and statistics
- Interactive schema mapping interface
- Visual rating standardization tools

### Model Operations
- Model selection and configuration
- Fine-tuning execution with progress tracking
- Inference processing with real-time updates
- Contextualizer creation and management

### Results and Export
- Interactive results visualization
- Performance metrics and analysis
- Multiple export formats (CSV, JSON)
- Session report generation

### Real-time Logging
- Live log display in the interface
- Error handling with user-friendly messages
- Progress tracking for all operations

## Installation

### Prerequisites

```bash
# Install required dependencies
pip install -r requirements.txt

# Streamlit is included in requirements.txt
```

### Running the GUI

#### Method 1: Using the Launcher Script

```bash
python run_gui.py
```

#### Method 2: Direct Streamlit Command

```bash
streamlit run gui/epituner_gui.py
```

#### Method 3: Custom Configuration

```bash
streamlit run gui/epituner_gui.py --server.port 8501 --server.address localhost
```

## User Interface Overview

### Main Layout

The GUI is organized into several key areas:

1. **Header**: Progress indicator showing current step
2. **Sidebar**: Navigation, settings, and system information
3. **Main Content**: Step-specific interface and controls
4. **Status Bar**: Real-time status updates and error messages

### Navigation

The sidebar provides:
- Quick navigation between steps
- Settings configuration
- Model configuration
- System information
- Session management

## Step-by-Step Guide

### Step 1: Data Upload & Validation

#### File Upload
- Click "Browse files" to select your CSV dataset
- Supported formats: CSV files
- File size limits: Up to 200MB

#### Data Validation
The system automatically validates:
- Required columns presence
- Data format and types
- Missing or invalid values
- File encoding and structure

#### Data Preview
- View first 10 rows of your dataset
- Check data statistics (rows, columns, memory usage)
- Verify column names and data types

#### Required Columns
Your CSV must contain:
- `C_BioSense_ID`: Unique identifier for each row
- `ChiefComplaintOrig`: Original chief complaint
- `Discharge Diagnosis`: Discharge diagnosis
- `Sex`: Patient sex
- `Age`: Patient age
- `Admit_Reason_Combo`: Admission reason
- `Chief_Complaint_Combo`: Chief complaint combination
- `Diagnosis_Combo`: Diagnosis combination
- `CCDD`: CCDD code
- `CCDDCategory`: CCDD category
- `TriageNotes`: Triage notes
- `Expert Rating`: Expert rating (optional)
- `Rationale of Rating`: Expert rationale (optional)

### Step 2: Schema Mapping & Rating Standardization

#### Expert Rating Detection
- System automatically detects if expert ratings are present
- Shows unique rating values found in the dataset
- Displays rating distribution

#### Rating Mapping Methods

**Manual Mapping**
- Map each unique rating to a standardized value
- Choose from predefined rating scales (0-4)
- Customize mapping for your specific needs

**Predefined Mapping**
- Choose from common rating scales:
  - Binary (0/1): No Match / Match
  - Three-level (0/1/2): No Match / Partial Match / Full Match
  - Five-level (0-4): No Match / Weak Match / Partial Match / Strong Match / Full Match

**Auto-detect**
- System attempts to automatically map ratings
- Handles numeric and string ratings
- Provides fallback mapping for non-numeric values

#### Topic Specification
- Enter the analysis topic (e.g., "Respiratory Issues")
- This topic will be used in all generated prompts
- Examples: "Cardiac Conditions", "Neurological Disorders", "Infectious Diseases"

### Step 3: Prompt Formatting & Preparation

#### Formatting Options

**Prompt Type Selection**
- Auto-detect: Automatically determine based on data
- Training Only: Create prompts with expert ratings
- Inference Only: Create prompts for model prediction
- Mixed: Handle both training and inference data

**Formatting Parameters**
- Include Expert Rationale: Add expert reasoning to training prompts
- Batch Size: Number of rows to process at once (1-1000)
- Custom Template: Use custom prompt templates

#### Custom Templates
You can provide custom templates for:
- Training prompts: Include `{topic}`, `{context_block}`, `{rating}`, `{rationale}`
- Inference prompts: Include `{topic}`, `{context_block}`

#### Prompt Preview
- View sample formatted prompts
- Check prompt statistics (count, average length, rating distribution)
- Verify prompt quality and structure

### Step 4: Model Operations

#### Operation Selection
Choose from:
- **Run Inference**: Execute model predictions
- **Fine-tune Model**: Train a new model
- **Create Contextualizer**: Build contextual prompts
- **All Operations**: Execute all available operations

#### Model Configuration

**Model Settings**
- Model Name: Select from available Ollama models
- Server URL: Ollama server address (default: http://localhost:11434)
- Max Tokens: Maximum response length (1-4096)
- Temperature: Response randomness (0.0-2.0)

**Batch Settings**
- Batch Size: Number of prompts to process simultaneously
- Progress Tracking: Real-time progress updates

#### Execution
- Click "Start Operation" to begin processing
- Monitor progress in real-time
- View logs and status updates
- Handle any errors that occur

### Step 5: Results & Export

#### Results Overview
- View summary statistics
- Check processing metrics
- Review operation status

#### Inference Results
If inference was performed:
- View predictions in tabular format
- Check accuracy metrics
- Analyze rating distributions
- Review model rationales

#### Performance Analysis
- Accuracy calculation
- Confusion matrix
- Error analysis
- Performance metrics

#### Export Options

**Data Export**
- Download processed data as CSV
- Export formatted prompts as JSON
- Save inference results as CSV
- Generate session report as JSON

**Report Generation**
- Comprehensive session summary
- Processing statistics
- Error logs and analysis
- Configuration details

## Configuration

### Settings Panel

The sidebar provides access to various settings:

#### Debug Settings
- **Debug Mode**: Enable verbose logging
- **Interactive Debug**: Enable interactive error resolution

#### Model Settings
- **Model Name**: Default model for operations
- **Server URL**: Ollama server address
- **Batch Size**: Default batch processing size

#### System Information
- Session ID and timestamp
- Current configuration
- System status

### Session Management

#### Session State
The GUI maintains session state for:
- Current step and progress
- Uploaded data and processing results
- User configurations and settings
- Error logs and status

#### Session Persistence
- Settings persist during the session
- Data is maintained between steps
- Results are available for export

#### Session Reset
- Clear all session data
- Reset to initial state
- Start fresh workflow

## Error Handling

### Validation Errors
- **File Format**: Invalid file type or structure
- **Missing Columns**: Required columns not found
- **Data Quality**: Invalid or missing data values
- **Schema Issues**: Rating mapping problems

### Processing Errors
- **Model Errors**: Model not found or unavailable
- **Network Issues**: Connection problems with Ollama server
- **Memory Issues**: Large dataset processing problems
- **Formatting Errors**: Prompt creation failures

### User Guidance
- Clear error messages with explanations
- Suggested solutions and fixes
- Recovery options and alternatives
- Contact information for support

## Performance Considerations

### Large Datasets
- Use batch processing for datasets > 1000 rows
- Monitor memory usage during processing
- Consider splitting very large datasets

### Model Operations
- Fine-tuning requires significant time and resources
- Inference operations can be resource-intensive
- Monitor system resources during processing

### Optimization Tips
- Use appropriate batch sizes for your system
- Close unnecessary applications during processing
- Ensure adequate disk space for temporary files

## Integration with Other Modules

### Data Flow
The GUI integrates all EpiTuner modules:

1. **Data Loader**: File upload and validation
2. **Schema Mapper**: Rating standardization
3. **Formatter**: Prompt creation
4. **Fine Tuner**: Model training
5. **Inference Runner**: Model predictions
6. **Contextualizer**: Context-aware prompts
7. **Debugging Logger**: Error handling and logging

### Module Communication
- Seamless data flow between modules
- Consistent error handling across all operations
- Unified logging and status reporting
- Integrated configuration management

## Best Practices

### Data Preparation
- Ensure your CSV file is properly formatted
- Check for missing or invalid data before upload
- Use consistent rating scales in your dataset
- Provide clear, descriptive topic names

### Workflow Management
- Complete each step before proceeding to the next
- Review data and results at each stage
- Save intermediate results when possible
- Document your configuration and settings

### Error Prevention
- Validate data before processing
- Check model availability before operations
- Monitor system resources during processing
- Keep backup copies of important data

### Performance Optimization
- Use appropriate batch sizes for your system
- Close unnecessary applications during processing
- Ensure adequate disk space
- Monitor memory usage with large datasets

## Troubleshooting

### Common Issues

#### GUI Not Starting
**Problem**: Streamlit application fails to start
**Solution**:
```bash
# Check Streamlit installation
pip install streamlit

# Verify Python environment
python --version

# Check for port conflicts
streamlit run gui/epituner_gui.py --server.port 8502
```

#### File Upload Issues
**Problem**: Cannot upload CSV file
**Solution**:
- Check file format (must be CSV)
- Verify file size (max 200MB)
- Ensure file is not corrupted
- Check file permissions

#### Model Connection Issues
**Problem**: Cannot connect to Ollama server
**Solution**:
- Verify Ollama is running: `ollama list`
- Check server URL in settings
- Ensure network connectivity
- Restart Ollama service

#### Processing Errors
**Problem**: Operations fail during processing
**Solution**:
- Check data quality and format
- Verify model availability
- Monitor system resources
- Review error logs in the interface

### Getting Help

#### Error Messages
- Read error messages carefully
- Check suggested solutions
- Review logs for detailed information
- Use the debugging features

#### Documentation
- Review this documentation
- Check module-specific documentation
- Consult the EpiTuner main documentation
- Review example workflows

#### Support
- Check the logs directory for detailed error information
- Review the session report for configuration details
- Use the debugging logger for troubleshooting
- Contact the development team with specific issues

## Advanced Features

### Custom Templates
Create custom prompt templates for specific use cases:

```python
# Training template example
training_template = """MEDICAL ANALYSIS:
Topic: {topic}
Patient Data: {context_block}
Expert Rating: {rating}
Expert Reasoning: {rationale}"""

# Inference template example
inference_template = """MEDICAL ANALYSIS:
Topic: {topic}
Patient Data: {context_block}
Please provide: Rating and Rationale"""
```

### Batch Processing
Configure batch processing for large datasets:
- Adjust batch size based on system capabilities
- Monitor progress and resource usage
- Handle errors gracefully within batches

### Session Management
- Save and restore session state
- Export configuration for reuse
- Share workflows between users
- Version control for configurations

## Security Considerations

### Data Privacy
- Uploaded data is processed locally
- No data is transmitted to external servers
- Temporary files are cleaned up automatically
- Session data is stored locally

### Access Control
- No user authentication required for local use
- File system permissions control access
- Network access limited to Ollama server
- No external API calls or data transmission

### Best Practices
- Use secure file permissions
- Regularly clean temporary files
- Monitor system access logs
- Keep software updated

## Future Enhancements

### Planned Features
- User authentication and access control
- Cloud-based processing options
- Advanced visualization tools
- Workflow automation and scheduling
- Integration with external databases
- Real-time collaboration features

### Extensibility
- Plugin architecture for custom modules
- API endpoints for external integration
- Custom visualization components
- Advanced configuration options

## Related Documentation

- **Data Loader**: Data upload and validation
- **Schema Mapper**: Rating standardization
- **Formatter**: Prompt creation
- **Fine Tuner**: Model training
- **Inference Runner**: Model predictions
- **Contextualizer**: Context-aware prompts
- **Debugging Logger**: Error handling and logging

## Version History

- **v1.0.0**: Initial implementation with basic workflow
- **v1.1.0**: Added advanced validation and error handling
- **v1.2.0**: Enhanced visualization and export features
- **v1.3.0**: Improved performance and user experience 