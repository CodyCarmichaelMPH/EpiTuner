# Debugging Logger Module

## Overview

The Debugging Logger module provides centralized, structured logging for all modules in the EpiTuner suite. It supports clear categorization of errors and actionable guidance for resolution, making troubleshooting more efficient and user-friendly.

## Purpose

- Provide structured logging across all EpiTuner modules
- Automatically categorize errors for easier debugging
- Generate actionable fix suggestions for common issues
- Support interactive debugging workflows
- Enable advanced log searching and filtering
- Generate error summaries and statistical analysis

## Key Features

### Structured JSON Logging
- All logs stored in structured JSON format with timestamps
- Session-based logging with unique session IDs
- Context-aware logging with module and function information
- Support for multiple log levels (DEBUG, INFO, WARNING, ERROR)

### Error Categorization
- **Data Issue**: Missing columns, invalid data types, empty datasets
- **Schema Issue**: Validation failures, missing required fields
- **Model Issue**: Model not found, training failures, inference errors
- **Runtime Error**: General execution errors, system issues
- **Configuration Error**: Invalid settings, missing configuration
- **Network Error**: Connection timeouts, server issues

### Interactive Debugging
- Optional interactive mode for error resolution
- User prompts for retry, skip, abort, or continue actions
- Real-time error handling with user guidance

### Advanced Logging Features
- Log searching by module, level, error category, and time range
- Error summaries with statistical analysis
- Function decorator for automatic logging
- Global logger singleton pattern
- Log buffer for batch operations

## Installation

The debugging logger is included in the `scripts` directory and requires no additional installation.

```bash
# No additional installation required
# Module is available at: scripts/debugging_logger.py
```

## Quick Start

### Basic Usage

```python
from scripts.debugging_logger import get_logger, LogLevel

# Get logger instance
logger = get_logger(debug_mode=True)

# Basic logging
logger.info("Starting data processing", "data_loader", "process_data")
logger.warning("Missing optional field", "schema_mapper", "validate_schema")
logger.error("Failed to load model", "fine_tuner", "load_model", error=exception)
```

### Error Logging with Categorization

```python
# Errors are automatically categorized
logger.error("Missing required column 'TriageNotes'", "data_loader")  # Data Issue
logger.error("Model not found: llama3.2", "fine_tuner")  # Model Issue
logger.error("Connection timeout", "inference_runner")  # Network Error
```

### Interactive Debugging

```python
# Enable interactive debugging
logger = get_logger(interactive_debug=True)

# When an error occurs, user will be prompted for action
result = logger.error("File not found", "data_loader", error=exception)
# User can choose: retry, skip, abort, or continue
```

## API Reference

### DebuggingLogger Class

#### Constructor

```python
DebuggingLogger(
    log_file: Optional[str] = None,
    interactive_debug: bool = False,
    debug_mode: bool = False
)
```

**Parameters:**
- `log_file`: Path to log file (auto-generated if None)
- `interactive_debug`: Enable interactive debugging prompts
- `debug_mode`: Enable verbose debug logging

#### Methods

##### Basic Logging

```python
def log(level: Union[LogLevel, str], message: str, module: str, 
        function: str = "", context: Optional[Dict[str, Any]] = None, 
        error: Optional[Exception] = None) -> Optional[str]
```

```python
def info(message: str, module: str, function: str = "", 
         context: Optional[Dict[str, Any]] = None)
```

```python
def warning(message: str, module: str, function: str = "", 
           context: Optional[Dict[str, Any]] = None)
```

```python
def error(message: str, module: str, function: str = "", 
          context: Optional[Dict[str, Any]] = None, 
          error: Optional[Exception] = None)
```

```python
def debug(message: str, module: str, function: str = "", 
          context: Optional[Dict[str, Any]] = None)
```

##### Specialized Logging

```python
def log_function_entry(module: str, function: str, **kwargs)
def log_function_exit(module: str, function: str, result: Any = None)
def log_data_processing(module: str, function: str, data_info: Dict[str, Any])
```

##### Log Searching

```python
def search_logs(module: Optional[str] = None, level: Optional[LogLevel] = None,
               error_category: Optional[ErrorCategory] = None,
               start_time: Optional[datetime] = None,
               end_time: Optional[datetime] = None) -> List[Dict[str, Any]]
```

##### Error Analysis

```python
def get_error_summary(session_id: Optional[str] = None) -> Dict[str, Any]
```

##### Buffer Operations

```python
def clear_log_buffer()
def flush_log_buffer()
```

### Global Functions

#### get_logger()

```python
def get_logger(log_file: Optional[str] = None, interactive_debug: bool = False,
               debug_mode: bool = False) -> DebuggingLogger
```

Returns a global logger instance (singleton pattern).

#### log_function Decorator

```python
@log_function
def your_function(param1, param2):
    # Function body
    return result
```

Automatically logs function entry, exit, and any exceptions.

### Enums

#### LogLevel

```python
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
```

#### ErrorCategory

```python
class ErrorCategory(Enum):
    DATA_ISSUE = "Data Issue"
    SCHEMA_ISSUE = "Schema Issue"
    MODEL_ISSUE = "Model Issue"
    RUNTIME_ERROR = "Runtime Error"
    CONFIGURATION_ERROR = "Configuration Error"
    NETWORK_ERROR = "Network Error"
```

## Usage Examples

### Basic Logging Workflow

```python
from scripts.debugging_logger import get_logger, LogLevel

# Initialize logger
logger = get_logger(debug_mode=True)

# Log data processing steps
logger.log_data_processing("data_loader", "load_dataset", {
    "rows": 1000,
    "columns": 15,
    "file": "dataset.csv"
})

# Log function execution
logger.log_function_entry("data_loader", "process_data", 
                         file_path="data.csv", batch_size=100)

try:
    # Process data
    result = process_data("data.csv", batch_size=100)
    logger.log_function_exit("data_loader", "process_data", result)
except Exception as e:
    logger.error("Data processing failed", "data_loader", "process_data", error=e)
```

### Error Handling with Categorization

```python
def load_dataset(file_path):
    logger = get_logger()
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        logger.info(f"Successfully loaded {len(df)} rows", "data_loader", "load_dataset")
        return df
        
    except FileNotFoundError as e:
        # Will be categorized as DATA_ISSUE
        logger.error("Failed to load dataset", "data_loader", "load_dataset", error=e)
        raise
    except ValueError as e:
        # Will be categorized as DATA_ISSUE
        logger.error("Invalid dataset", "data_loader", "load_dataset", error=e)
        raise
```

### Interactive Debugging

```python
def process_with_interactive_debug():
    logger = get_logger(interactive_debug=True)
    
    try:
        # Some operation that might fail
        result = risky_operation()
        return result
    except Exception as e:
        # User will be prompted for action
        action = logger.error("Operation failed", "module", "function", error=e)
        
        if action == 'retry':
            return process_with_interactive_debug()
        elif action == 'skip':
            return None
        elif action == 'abort':
            sys.exit(1)
        else:  # continue
            raise
```

### Log Searching and Analysis

```python
def analyze_logs():
    logger = get_logger()
    
    # Search for all errors
    errors = logger.search_logs(level=LogLevel.ERROR)
    print(f"Found {len(errors)} errors")
    
    # Search for data-related errors
    data_errors = logger.search_logs(error_category=ErrorCategory.DATA_ISSUE)
    print(f"Found {len(data_errors)} data errors")
    
    # Search for errors from specific module
    module_errors = logger.search_logs(module="data_loader")
    print(f"Found {len(module_errors)} errors from data_loader")
    
    # Get error summary
    summary = logger.get_error_summary()
    print(f"Error summary: {summary}")
```

### Function Decorator Usage

```python
from scripts.debugging_logger import log_function

@log_function
def process_dataset(file_path, batch_size=100):
    """Process dataset with automatic logging."""
    logger = get_logger()
    
    logger.info(f"Processing {file_path} with batch size {batch_size}")
    
    # Function logic here
    result = do_processing(file_path, batch_size)
    
    return result

# Usage
result = process_dataset("data.csv", batch_size=50)
# Function entry, exit, and any exceptions are automatically logged
```

## Configuration

### Log File Management

```python
# Auto-generated log file (default)
logger = DebuggingLogger()
# Creates: logs/session_YYYYMMDD_HHMMSS.json

# Custom log file
logger = DebuggingLogger(log_file="custom_log.json")

# Multiple loggers with different files
logger1 = DebuggingLogger(log_file="data_processing.log")
logger2 = DebuggingLogger(log_file="model_training.log")
```

### Debug Mode

```python
# Enable debug mode for verbose logging
logger = DebuggingLogger(debug_mode=True)

# Debug mode enables:
# - More detailed log messages
# - Function entry/exit logging
# - Context information in logs
```

### Interactive Debugging

```python
# Enable interactive debugging
logger = DebuggingLogger(interactive_debug=True)

# When errors occur, user will be prompted:
# Choose action: [r]etry, [s]kip, [a]bort, [c]ontinue
```

## Log File Format

Log entries are stored in JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "session_id": "20240115_103000",
  "level": "ERROR",
  "module": "data_loader",
  "function": "load_dataset",
  "message": "Failed to load dataset",
  "context": {
    "file_path": "data.csv",
    "batch_size": 100
  },
  "error_category": "Data Issue",
  "suggested_fix": "Check dataset file format and encoding",
  "error_type": "FileNotFoundError",
  "error_details": "File not found: data.csv",
  "traceback": "..."
}
```

## Integration with Other Modules

### In Data Loader

```python
from scripts.debugging_logger import get_logger

class DataLoader:
    def __init__(self, debug_mode=False):
        self.logger = get_logger(debug_mode=debug_mode)
    
    def load_dataset(self, file_path):
        self.logger.log_function_entry("data_loader", "load_dataset", 
                                     file_path=file_path)
        try:
            # Loading logic
            result = self._load_file(file_path)
            self.logger.log_function_exit("data_loader", "load_dataset", 
                                        len(result))
            return result
        except Exception as e:
            self.logger.error("Failed to load dataset", "data_loader", 
                            "load_dataset", error=e)
            raise
```

### In Schema Mapper

```python
class SchemaMapper:
    def __init__(self, debug_mode=False):
        self.logger = get_logger(debug_mode=debug_mode)
    
    def apply_rating_mapping(self, df, rating_mapping):
        self.logger.log_data_processing("schema_mapper", "apply_rating_mapping", {
            "input_rows": len(df),
            "mapping_size": len(rating_mapping)
        })
        # Mapping logic
```

## Best Practices

### 1. Use Appropriate Log Levels

```python
# Use DEBUG for detailed troubleshooting
logger.debug("Processing row 123", "module", "function")

# Use INFO for normal operations
logger.info("Dataset loaded successfully", "module", "function")

# Use WARNING for potential issues
logger.warning("Missing optional field", "module", "function")

# Use ERROR for actual failures
logger.error("Failed to process data", "module", "function", error=exception)
```

### 2. Provide Context

```python
# Good: Include relevant context
logger.error("Failed to load dataset", "data_loader", "load_dataset", 
            context={"file_path": file_path, "file_size": file_size}, 
            error=exception)

# Bad: Minimal information
logger.error("Failed", "module", "function")
```

### 3. Use Function Decorator for Simple Functions

```python
@log_function
def simple_utility_function(param1, param2):
    return param1 + param2

# For complex functions, use manual logging
def complex_function(param1, param2):
    logger = get_logger()
    logger.log_function_entry("module", "complex_function", 
                             param1=param1, param2=param2)
    # Complex logic with multiple steps
    logger.log_function_exit("module", "complex_function", result)
    return result
```

### 4. Handle Exceptions Properly

```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error("Specific error occurred", "module", "function", error=e)
    # Handle specific exception
except Exception as e:
    logger.error("Unexpected error", "module", "function", error=e)
    # Handle general exception
    raise  # Re-raise if you can't handle it
```

## Troubleshooting

### Common Issues

#### Log Files Not Created

**Problem**: Log files are not being created
**Solution**: 
- Check that the `logs` directory exists
- Ensure write permissions for the directory
- Verify the logger is properly initialized

```python
# Create logs directory if it doesn't exist
from pathlib import Path
Path("logs").mkdir(exist_ok=True)
```

#### Interactive Debugging Not Working

**Problem**: Interactive debugging prompts don't appear
**Solution**:
- Ensure `interactive_debug=True` when creating the logger
- Check that you're calling the error method (not just logging)
- Verify you're in an interactive environment

```python
logger = get_logger(interactive_debug=True)
result = logger.error("Test error", "module", "function")  # Will prompt
```

#### Import Errors

**Problem**: Cannot import debugging_logger
**Solution**:
- Ensure the scripts directory is in your Python path
- Check that the file exists at `scripts/debugging_logger.py`

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'scripts'))
from debugging_logger import get_logger
```

### Performance Considerations

#### Large Log Files

**Problem**: Log files become very large
**Solution**:
- Use log rotation or archiving
- Implement log cleanup policies
- Consider using different log files for different operations

#### Frequent Logging

**Problem**: Too much logging impacts performance
**Solution**:
- Use appropriate log levels
- Disable debug logging in production
- Use batch logging for high-frequency operations

## Testing

### Running Tests

```bash
# Run debugging logger tests
pytest tests/test_debugging_logger.py -v

# Run with coverage
pytest tests/test_debugging_logger.py --cov=scripts.debugging_logger
```

### Test Coverage

The test suite covers:
- Basic logging functionality
- Error categorization
- Interactive debugging
- Log searching and filtering
- Error summaries
- File operations
- Edge cases and error conditions

### Demo Script

```bash
# Run the debugging logger demo
python demo_debugging_logger.py
```

## Related Modules

- **Data Loader**: Uses debugging logger for data loading operations
- **Schema Mapper**: Uses debugging logger for schema validation
- **Formatter**: Uses debugging logger for prompt creation
- **Fine Tuner**: Uses debugging logger for model training
- **Inference Runner**: Uses debugging logger for inference operations
- **GUI**: Uses debugging logger for all user operations

## Version History

- **v1.0.0**: Initial implementation with basic logging and error categorization
- **v1.1.0**: Added interactive debugging and log searching
- **v1.2.0**: Added function decorator and error summaries
- **v1.3.0**: Enhanced error categorization and fix suggestions 