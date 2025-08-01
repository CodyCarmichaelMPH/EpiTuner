"""
Debugging Logger Module for Ollama Fine-Tuning and Evaluation Suite

This module provides centralized, structured logging for all modules in the suite.
It supports clear categorization of errors and actionable guidance for resolution.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import traceback


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ErrorCategory(Enum):
    """Categories for error classification."""
    DATA_ISSUE = "Data Issue"
    SCHEMA_ISSUE = "Schema Issue"
    MODEL_ISSUE = "Model Issue"
    RUNTIME_ERROR = "Runtime Error"
    CONFIGURATION_ERROR = "Configuration Error"
    NETWORK_ERROR = "Network Error"


class DebuggingLogger:
    """
    Centralized structured logging utility for the EpiTuner suite.
    
    Provides:
    - Structured JSON logging
    - Error categorization
    - Actionable fix suggestions
    - Console and file output
    - Interactive debugging mode
    """
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 interactive_debug: bool = False,
                 debug_mode: bool = False):
        """
        Initialize the debugging logger.
        
        Args:
            log_file: Path to log file (auto-generated if None)
            interactive_debug: Enable interactive debugging prompts
            debug_mode: Enable verbose debug logging
        """
        self.interactive_debug = interactive_debug
        self.debug_mode = debug_mode
        self.log_file = log_file or self._generate_log_filename()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure logs directory exists
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup standard logging
        self._setup_standard_logging()
        
        # Initialize log buffer for batch operations
        self.log_buffer = []
        
    def _generate_log_filename(self) -> str:
        """Generate a timestamped log filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"logs/session_{timestamp}.json"
    
    def _setup_standard_logging(self):
        """Setup standard Python logging for compatibility."""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('debugging_logger.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _categorize_error(self, message: str, module: str, error: Optional[Exception] = None) -> ErrorCategory:
        """
        Automatically categorize errors based on message content and module.
        
        Args:
            message: Error message
            module: Source module name
            error: Exception object if available
            
        Returns:
            ErrorCategory enum value
        """
        message_lower = message.lower()
        module_lower = module.lower()
        
        # Data-related keywords
        data_keywords = ['missing column', 'invalid data', 'empty dataset', 'data type', 
                        'schema', 'field', 'column', 'csv', 'json', 'file format']
        
        # Schema-related keywords
        schema_keywords = ['schema', 'validation', 'required field', 'missing field',
                          'column mapping', 'field mapping']
        
        # Model-related keywords
        model_keywords = ['model', 'ollama', 'fine-tune', 'inference', 'prediction',
                         'gguf', 'model set', 'training']
        
        # Configuration-related keywords
        config_keywords = ['config', 'setting', 'parameter', 'url', 'endpoint',
                          'server', 'connection']
        
        # Network-related keywords
        network_keywords = ['connection', 'timeout', 'network', 'http', 'api',
                           'server', 'endpoint', 'url']
        
        # Check for data issues
        if any(keyword in message_lower for keyword in data_keywords):
            return ErrorCategory.DATA_ISSUE
        
        # Check for schema issues
        if any(keyword in message_lower for keyword in schema_keywords):
            return ErrorCategory.SCHEMA_ISSUE
        
        # Check for model issues
        if any(keyword in message_lower for keyword in model_keywords) or 'model' in module_lower:
            return ErrorCategory.MODEL_ISSUE
        
        # Check for configuration issues
        if any(keyword in message_lower for keyword in config_keywords):
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Check for network issues
        if any(keyword in message_lower for keyword in network_keywords):
            return ErrorCategory.NETWORK_ERROR
        
        # Default to runtime error
        return ErrorCategory.RUNTIME_ERROR
    
    def _generate_fix_suggestion(self, category: ErrorCategory, message: str, module: str) -> str:
        """
        Generate actionable fix suggestions based on error category.
        
        Args:
            category: Error category
            message: Error message
            module: Source module
            
        Returns:
            Suggested fix string
        """
        suggestions = {
            ErrorCategory.DATA_ISSUE: [
                "Check dataset file format and encoding",
                "Verify all required columns are present",
                "Clean and validate data before processing",
                "Check for missing or invalid values"
            ],
            ErrorCategory.SCHEMA_ISSUE: [
                "Review dataset schema requirements",
                "Map missing columns to required fields",
                "Validate data types for each column",
                "Check column naming conventions"
            ],
            ErrorCategory.MODEL_ISSUE: [
                "Verify Ollama server is running",
                "Check model availability and compatibility",
                "Validate model configuration settings",
                "Ensure sufficient system resources"
            ],
            ErrorCategory.CONFIGURATION_ERROR: [
                "Review configuration file settings",
                "Check environment variables",
                "Validate server URLs and endpoints",
                "Verify file paths and permissions"
            ],
            ErrorCategory.NETWORK_ERROR: [
                "Check network connectivity",
                "Verify server is accessible",
                "Check firewall and proxy settings",
                "Retry operation after network stabilization"
            ],
            ErrorCategory.RUNTIME_ERROR: [
                "Check system resources and memory",
                "Verify Python environment and dependencies",
                "Review code logic and error handling",
                "Check file permissions and disk space"
            ]
        }
        
        # Return first suggestion for the category
        return suggestions.get(category, ["Review error details and retry"])[0]
    
    def _create_log_entry(self, 
                         level: LogLevel,
                         message: str,
                         module: str,
                         function: str = "",
                         context: Optional[Dict[str, Any]] = None,
                         error: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Create a structured log entry.
        
        Args:
            level: Log level
            message: Log message
            module: Source module
            function: Function name
            context: Additional context data
            error: Exception object if applicable
            
        Returns:
            Structured log entry dictionary
        """
        timestamp = datetime.now().isoformat()
        
        # Determine error category and fix suggestion for errors
        category = None
        suggested_fix = None
        
        if level == LogLevel.ERROR:
            category = self._categorize_error(message, module, error)
            suggested_fix = self._generate_fix_suggestion(category, message, module)
        
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "level": level.value,
            "module": module,
            "function": function,
            "message": message,
            "context": context or {}
        }
        
        if category:
            log_entry["error_category"] = category.value
            log_entry["suggested_fix"] = suggested_fix
        
        if error:
            log_entry["error_type"] = type(error).__name__
            log_entry["error_details"] = str(error)
            log_entry["traceback"] = traceback.format_exc()
        
        return log_entry
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write log entry to file and console."""
        # Write to JSON log file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            # Fallback to standard logging if JSON write fails
            self.logger.error(f"Failed to write to JSON log: {e}")
        
        # Write to console with formatted output
        level_color = {
            LogLevel.DEBUG: "\033[36m",    # Cyan
            LogLevel.INFO: "\033[32m",     # Green
            LogLevel.WARNING: "\033[33m",  # Yellow
            LogLevel.ERROR: "\033[31m"     # Red
        }
        
        reset_color = "\033[0m"
        color = level_color.get(LogLevel(log_entry["level"]), "")
        
        # Format console output
        console_msg = f"{color}[{log_entry['level']}] {log_entry['module']}"
        if log_entry.get('function'):
            console_msg += f".{log_entry['function']}"
        console_msg += f": {log_entry['message']}{reset_color}"
        
        if log_entry.get('suggested_fix'):
            console_msg += f"\n{color}Suggested Fix: {log_entry['suggested_fix']}{reset_color}"
        
        print(console_msg)
        
        # Also log to standard logger for compatibility
        log_method = getattr(self.logger, log_entry['level'].lower())
        log_method(console_msg)
    
    def _handle_interactive_debug(self, log_entry: Dict[str, Any]) -> str:
        """
        Handle interactive debugging if enabled.
        
        Args:
            log_entry: Log entry with error details
            
        Returns:
            User's choice: 'retry', 'skip', or 'abort'
        """
        if not self.interactive_debug or log_entry['level'] != LogLevel.ERROR.value:
            return 'continue'
        
        print(f"\n{'='*60}")
        print(f"INTERACTIVE DEBUG - {log_entry['module']}")
        print(f"Error: {log_entry['message']}")
        if log_entry.get('suggested_fix'):
            print(f"Suggested Fix: {log_entry['suggested_fix']}")
        print(f"{'='*60}")
        
        while True:
            choice = input("Choose action: [r]etry, [s]kip, [a]bort, [c]ontinue: ").lower().strip()
            if choice in ['r', 'retry']:
                return 'retry'
            elif choice in ['s', 'skip']:
                return 'skip'
            elif choice in ['a', 'abort']:
                return 'abort'
            elif choice in ['c', 'continue']:
                return 'continue'
            else:
                print("Invalid choice. Please enter r, s, a, or c.")
    
    def log(self, 
            level: Union[LogLevel, str],
            message: str,
            module: str,
            function: str = "",
            context: Optional[Dict[str, Any]] = None,
            error: Optional[Exception] = None) -> Optional[str]:
        """
        Log a message with structured formatting.
        
        Args:
            level: Log level (LogLevel enum or string)
            message: Log message
            module: Source module name
            function: Function name (optional)
            context: Additional context data (optional)
            error: Exception object (optional)
            
        Returns:
            User choice for interactive debugging, or None
        """
        # Convert string level to enum if needed
        if isinstance(level, str):
            try:
                level = LogLevel(level.upper())
            except ValueError:
                level = LogLevel.INFO
        
        # Create log entry
        log_entry = self._create_log_entry(level, message, module, function, context, error)
        
        # Write to output
        self._write_log_entry(log_entry)
        
        # Handle interactive debugging
        if self.interactive_debug and level == LogLevel.ERROR:
            return self._handle_interactive_debug(log_entry)
        
        return None
    
    def debug(self, message: str, module: str, function: str = "", context: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self.log(LogLevel.DEBUG, message, module, function, context)
    
    def info(self, message: str, module: str, function: str = "", context: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self.log(LogLevel.INFO, message, module, function, context)
    
    def warning(self, message: str, module: str, function: str = "", context: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        self.log(LogLevel.WARNING, message, module, function, context)
    
    def error(self, message: str, module: str, function: str = "", context: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
        """Log an error message."""
        return self.log(LogLevel.ERROR, message, module, function, context, error)
    
    def log_function_entry(self, module: str, function: str, **kwargs):
        """Log function entry with parameters."""
        context = {"function_params": kwargs} if kwargs else None
        self.debug(f"Entering function {function}", module, function, context)
    
    def log_function_exit(self, module: str, function: str, result: Any = None):
        """Log function exit with result."""
        context = {"function_result": str(result)} if result is not None else None
        self.debug(f"Exiting function {function}", module, function, context)
    
    def log_data_processing(self, module: str, function: str, data_info: Dict[str, Any]):
        """Log data processing information."""
        self.info(f"Processing data: {data_info}", module, function, {"data_info": data_info})
    
    def search_logs(self, 
                   module: Optional[str] = None,
                   level: Optional[LogLevel] = None,
                   error_category: Optional[ErrorCategory] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Search logs by various criteria.
        
        Args:
            module: Filter by module name
            level: Filter by log level
            error_category: Filter by error category
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of matching log entries
        """
        if not os.path.exists(self.log_file):
            return []
        
        matching_entries = []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        
                        # Apply filters
                        if module and entry.get('module') != module:
                            continue
                        
                        if level and entry.get('level') != level.value:
                            continue
                        
                        if error_category and entry.get('error_category') != error_category.value:
                            continue
                        
                        if start_time:
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            if entry_time < start_time:
                                continue
                        
                        if end_time:
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            if entry_time > end_time:
                                continue
                        
                        matching_entries.append(entry)
        
        except Exception as e:
            self.logger.error(f"Error searching logs: {e}")
        
        return matching_entries
    
    def get_error_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of errors for a session.
        
        Args:
            session_id: Session ID to analyze (uses current if None)
            
        Returns:
            Error summary dictionary
        """
        target_session = session_id or self.session_id
        
        errors = self.search_logs(level=LogLevel.ERROR)
        session_errors = [e for e in errors if e.get('session_id') == target_session]
        
        summary = {
            "session_id": target_session,
            "total_errors": len(session_errors),
            "error_categories": {},
            "modules_with_errors": set(),
            "common_suggestions": []
        }
        
        for error in session_errors:
            # Count error categories
            category = error.get('error_category', 'Unknown')
            summary["error_categories"][category] = summary["error_categories"].get(category, 0) + 1
            
            # Track modules with errors
            summary["modules_with_errors"].add(error.get('module', 'Unknown'))
            
            # Collect suggestions
            if error.get('suggested_fix'):
                summary["common_suggestions"].append(error['suggested_fix'])
        
        # Convert set to list for JSON serialization
        summary["modules_with_errors"] = list(summary["modules_with_errors"])
        
        # Get most common suggestions
        from collections import Counter
        suggestion_counts = Counter(summary["common_suggestions"])
        summary["common_suggestions"] = [suggestion for suggestion, count in suggestion_counts.most_common(5)]
        
        return summary
    
    def clear_log_buffer(self):
        """Clear the log buffer."""
        self.log_buffer.clear()
    
    def flush_log_buffer(self):
        """Flush buffered logs to file."""
        for entry in self.log_buffer:
            self._write_log_entry(entry)
        self.clear_log_buffer()


# Global logger instance
_global_logger = None


def get_logger(log_file: Optional[str] = None,
               interactive_debug: bool = False,
               debug_mode: bool = False) -> DebuggingLogger:
    """
    Get or create a global logger instance.
    
    Args:
        log_file: Path to log file
        interactive_debug: Enable interactive debugging
        debug_mode: Enable debug mode
        
    Returns:
        DebuggingLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = DebuggingLogger(log_file, interactive_debug, debug_mode)
    
    return _global_logger


def log_function(func):
    """Decorator to automatically log function entry and exit."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        module = func.__module__ or "unknown"
        function = func.__name__
        
        logger.log_function_entry(module, function, args=args, kwargs=kwargs)
        
        try:
            result = func(*args, **kwargs)
            logger.log_function_exit(module, function, result)
            return result
        except Exception as e:
            logger.error(f"Function {function} failed", module, function, error=e)
            raise
    
    return wrapper


def main():
    """Demo function to test the debugging logger."""
    logger = get_logger(debug_mode=True, interactive_debug=False)
    
    # Test different log levels
    logger.info("Starting demo", "demo", "main")
    logger.debug("Debug information", "demo", "main", {"test": "data"})
    logger.warning("Warning message", "demo", "warning_test")
    
    # Test error logging
    try:
        raise ValueError("Test error for demonstration")
    except Exception as e:
        logger.error("Caught test error", "demo", "main", error=e)
    
    # Test data processing logging
    logger.log_data_processing("demo", "main", {
        "rows": 100,
        "columns": 10,
        "file": "test.csv"
    })
    
    # Test search functionality
    errors = logger.search_logs(level=LogLevel.ERROR)
    print(f"\nFound {len(errors)} error entries")
    
    # Test error summary
    summary = logger.get_error_summary()
    print(f"\nError summary: {summary}")


if __name__ == "__main__":
    main() 