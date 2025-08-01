# Debugging Logger Quick Reference

## Quick Start

```python
from scripts.debugging_logger import get_logger, LogLevel

# Get logger
logger = get_logger(debug_mode=True)

# Basic logging
logger.info("Message", "module", "function")
logger.error("Error", "module", "function", error=exception)
```

## Common Patterns

### Function Logging
```python
@log_function
def your_function(param1, param2):
    return result

# Or manual logging
logger.log_function_entry("module", "function", param1=param1)
try:
    result = do_work()
    logger.log_function_exit("module", "function", result)
    return result
except Exception as e:
    logger.error("Failed", "module", "function", error=e)
    raise
```

### Data Processing Logging
```python
logger.log_data_processing("module", "function", {
    "rows": 1000,
    "columns": 15,
    "file": "data.csv"
})
```

### Error Handling
```python
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed", "module", "function", error=e)
    # Error will be automatically categorized
```

## Log Levels

- `DEBUG`: Detailed troubleshooting information
- `INFO`: Normal operations and status updates
- `WARNING`: Potential issues that don't stop execution
- `ERROR`: Actual failures that need attention

## Error Categories

- `DATA_ISSUE`: Missing columns, invalid data, empty datasets
- `SCHEMA_ISSUE`: Validation failures, missing required fields
- `MODEL_ISSUE`: Model not found, training failures
- `RUNTIME_ERROR`: General execution errors
- `CONFIGURATION_ERROR`: Invalid settings, missing config
- `NETWORK_ERROR`: Connection timeouts, server issues

## Search and Analysis

```python
# Search logs
errors = logger.search_logs(level=LogLevel.ERROR)
data_errors = logger.search_logs(error_category=ErrorCategory.DATA_ISSUE)
module_logs = logger.search_logs(module="data_loader")

# Get error summary
summary = logger.get_error_summary()
```

## Configuration

```python
# Basic logger
logger = get_logger()

# Debug mode
logger = get_logger(debug_mode=True)

# Interactive debugging
logger = get_logger(interactive_debug=True)

# Custom log file
logger = get_logger(log_file="custom.log")
```

## Log File Format

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "session_id": "20240115_103000",
  "level": "ERROR",
  "module": "data_loader",
  "function": "load_dataset",
  "message": "Failed to load dataset",
  "error_category": "Data Issue",
  "suggested_fix": "Check file format and encoding",
  "error_type": "FileNotFoundError",
  "error_details": "File not found: data.csv"
}
``` 