"""
Unit tests for the DebuggingLogger module.

Tests cover:
- Structured logging functionality
- Error categorization
- Fix suggestion generation
- Interactive debugging
- Log searching and filtering
- Error summaries
- File operations
- Edge cases
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta
import time # Added missing import for time.sleep

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from debugging_logger import (
    DebuggingLogger, 
    LogLevel, 
    ErrorCategory, 
    get_logger, 
    log_function
)


class TestDebuggingLogger:
    """Test suite for DebuggingLogger class."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def logger(self, temp_log_file):
        """Create DebuggingLogger instance for testing."""
        return DebuggingLogger(log_file=temp_log_file, debug_mode=True)
    
    def test_init(self, logger):
        """Test DebuggingLogger initialization."""
        assert logger.debug_mode is True
        assert logger.interactive_debug is False
        assert logger.log_file is not None
        assert logger.session_id is not None
        assert hasattr(logger, 'logger')
    
    def test_generate_log_filename(self, logger):
        """Test log filename generation."""
        filename = logger._generate_log_filename()
        assert filename.startswith("logs/session_")
        assert filename.endswith(".json")
        assert len(filename) > 20  # Should have timestamp
    
    def test_categorize_error_data_issue(self, logger):
        """Test error categorization for data issues."""
        message = "Missing required column 'TriageNotes'"
        category = logger._categorize_error(message, "data_loader")
        assert category == ErrorCategory.DATA_ISSUE
    
    def test_categorize_error_schema_issue(self, logger):
        """Test error categorization for schema issues."""
        message = "Schema validation failed: missing field"
        category = logger._categorize_error(message, "schema_mapper")
        assert category == ErrorCategory.SCHEMA_ISSUE
    
    def test_categorize_error_model_issue(self, logger):
        """Test error categorization for model issues."""
        message = "Model not found: llama3.2"
        category = logger._categorize_error(message, "fine_tuner")
        assert category == ErrorCategory.MODEL_ISSUE
    
    def test_categorize_error_configuration_error(self, logger):
        """Test error categorization for configuration errors."""
        message = "Invalid configuration setting"
        category = logger._categorize_error(message, "config_manager")
        assert category == ErrorCategory.CONFIGURATION_ERROR
    
    def test_categorize_error_network_error(self, logger):
        """Test error categorization for network errors."""
        message = "Connection timeout to server"
        category = logger._categorize_error(message, "inference_runner")
        assert category == ErrorCategory.NETWORK_ERROR
    
    def test_categorize_error_runtime_error(self, logger):
        """Test error categorization for runtime errors."""
        message = "Unexpected error occurred"
        category = logger._categorize_error(message, "unknown_module")
        assert category == ErrorCategory.RUNTIME_ERROR
    
    def test_generate_fix_suggestion(self, logger):
        """Test fix suggestion generation."""
        suggestion = logger._generate_fix_suggestion(
            ErrorCategory.DATA_ISSUE, 
            "Missing column", 
            "data_loader"
        )
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0
    
    def test_create_log_entry_info(self, logger):
        """Test log entry creation for info level."""
        log_entry = logger._create_log_entry(
            LogLevel.INFO,
            "Test message",
            "test_module",
            "test_function"
        )
        
        assert log_entry['level'] == 'INFO'
        assert log_entry['message'] == 'Test message'
        assert log_entry['module'] == 'test_module'
        assert log_entry['function'] == 'test_function'
        assert 'timestamp' in log_entry
        assert 'session_id' in log_entry
        assert 'error_category' not in log_entry  # Should not be present for INFO
    
    def test_create_log_entry_error(self, logger):
        """Test log entry creation for error level."""
        error = ValueError("Test error")
        log_entry = logger._create_log_entry(
            LogLevel.ERROR,
            "Test error message",
            "test_module",
            "test_function",
            error=error
        )
        
        assert log_entry['level'] == 'ERROR'
        assert log_entry['message'] == 'Test error message'
        assert log_entry['error_category'] is not None
        assert log_entry['suggested_fix'] is not None
        assert log_entry['error_type'] == 'ValueError'
        assert log_entry['error_details'] == 'Test error'
        assert 'traceback' in log_entry
    
    def test_log_info(self, logger):
        """Test info logging."""
        logger.info("Test info message", "test_module", "test_function")
        
        # Check if log file was created and contains entry
        assert os.path.exists(logger.log_file)
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            # Parse last line
            last_entry = json.loads(lines[-1])
            assert last_entry['level'] == 'INFO'
            assert last_entry['message'] == 'Test info message'
    
    def test_log_error(self, logger):
        """Test error logging."""
        error = ValueError("Test error")
        result = logger.error("Test error message", "test_module", "test_function", error=error)
        
        # Should return None since interactive_debug is False
        assert result is None
        
        # Check log file
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            last_entry = json.loads(lines[-1])
            assert last_entry['level'] == 'ERROR'
            assert last_entry['error_category'] is not None
            assert last_entry['suggested_fix'] is not None
    
    def test_log_function_entry_exit(self, logger):
        """Test function entry and exit logging."""
        logger.log_function_entry("test_module", "test_function", param1="value1", param2=2)
        logger.log_function_exit("test_module", "test_function", result="success")
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 2
            
            # Check entry log
            entry_log = json.loads(lines[-2])
            assert entry_log['level'] == 'DEBUG'
            assert 'function_params' in entry_log['context']
            
            # Check exit log
            exit_log = json.loads(lines[-1])
            assert exit_log['level'] == 'DEBUG'
            assert 'function_result' in exit_log['context']
    
    def test_log_data_processing(self, logger):
        """Test data processing logging."""
        data_info = {"rows": 100, "columns": 10, "file": "test.csv"}
        logger.log_data_processing("test_module", "process_data", data_info)
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            last_entry = json.loads(lines[-1])
            assert last_entry['level'] == 'INFO'
            assert 'data_info' in last_entry['context']
    
    def test_search_logs_by_module(self, logger):
        """Test log searching by module."""
        # Create some test logs
        logger.info("Test message 1", "module1")
        logger.info("Test message 2", "module2")
        logger.error("Test error", "module1")
        
        # Search by module
        module1_logs = logger.search_logs(module="module1")
        assert len(module1_logs) == 2
        
        module2_logs = logger.search_logs(module="module2")
        assert len(module2_logs) == 1
    
    def test_search_logs_by_level(self, logger):
        """Test log searching by level."""
        # Create test logs
        logger.info("Info message", "test_module")
        logger.error("Error message", "test_module")
        logger.warning("Warning message", "test_module")
        
        # Search by level
        error_logs = logger.search_logs(level=LogLevel.ERROR)
        assert len(error_logs) == 1
        assert error_logs[0]['level'] == 'ERROR'
        
        info_logs = logger.search_logs(level=LogLevel.INFO)
        assert len(info_logs) == 1
        assert info_logs[0]['level'] == 'INFO'
    
    def test_search_logs_by_error_category(self, logger):
        """Test log searching by error category."""
        # Create error logs
        logger.error("Data issue", "data_loader")
        logger.error("Model issue", "fine_tuner")
        
        # Search by category
        data_errors = logger.search_logs(error_category=ErrorCategory.DATA_ISSUE)
        assert len(data_errors) == 1
        
        model_errors = logger.search_logs(error_category=ErrorCategory.MODEL_ISSUE)
        assert len(model_errors) == 1
    
    def test_search_logs_by_time_range(self, logger):
        """Test log searching by time range."""
        # Create test logs
        logger.info("Old message", "test_module")
        
        # Wait a moment
        time.sleep(0.1)
        start_time = datetime.now()
        
        logger.info("New message", "test_module")
        
        # Search by time range
        recent_logs = logger.search_logs(start_time=start_time)
        assert len(recent_logs) == 1
    
    def test_get_error_summary(self, logger):
        """Test error summary generation."""
        # Create some error logs
        logger.error("Error 1", "module1")
        logger.error("Error 2", "module1")
        logger.error("Error 3", "module2")
        
        summary = logger.get_error_summary()
        
        assert summary['total_errors'] == 3
        assert 'module1' in summary['modules_with_errors']
        assert 'module2' in summary['modules_with_errors']
        assert len(summary['error_categories']) > 0
        assert len(summary['common_suggestions']) > 0
    
    def test_log_buffer_operations(self, logger):
        """Test log buffer operations."""
        # Test buffer operations
        logger.clear_log_buffer()
        assert len(logger.log_buffer) == 0
        
        # Add to buffer
        logger.log_buffer.append({"test": "entry"})
        assert len(logger.log_buffer) == 1
        
        # Flush buffer
        logger.flush_log_buffer()
        assert len(logger.log_buffer) == 0
    
    def test_string_level_conversion(self, logger):
        """Test string to LogLevel conversion."""
        result = logger.log("info", "Test message", "test_module")
        assert result is None
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            last_entry = json.loads(lines[-1])
            assert last_entry['level'] == 'INFO'
    
    def test_invalid_string_level(self, logger):
        """Test handling of invalid string level."""
        result = logger.log("invalid_level", "Test message", "test_module")
        assert result is None
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            last_entry = json.loads(lines[-1])
            assert last_entry['level'] == 'INFO'  # Should default to INFO


class TestGlobalLogger:
    """Test suite for global logger functionality."""
    
    def test_get_logger_singleton(self):
        """Test that get_logger returns the same instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2
    
    def test_get_logger_with_params(self):
        """Test get_logger with parameters."""
        logger = get_logger(debug_mode=True, interactive_debug=True)
        assert logger.debug_mode is True
        assert logger.interactive_debug is True


class TestLogFunctionDecorator:
    """Test suite for log_function decorator."""
    
    def test_log_function_decorator(self):
        """Test the log_function decorator."""
        @log_function
        def test_function(param1, param2):
            return param1 + param2
        
        result = test_function(1, 2)
        assert result == 3
        
        # Check that logs were created
        # Note: This would require access to the global logger instance
        # which is complex to test in isolation
    
    def test_log_function_decorator_with_exception(self):
        """Test the log_function decorator with exceptions."""
        @log_function
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_log_file_permission_error(self, temp_log_file):
        """Test handling of log file permission errors."""
        # Make file read-only
        os.chmod(temp_log_file, 0o444)
        
        logger = DebuggingLogger(log_file=temp_log_file)
        
        # Should not raise exception, should fall back to standard logging
        logger.info("Test message", "test_module")
        
        # Restore permissions
        os.chmod(temp_log_file, 0o666)
    
    def test_empty_message(self, logger):
        """Test logging with empty message."""
        logger.info("", "test_module")
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            last_entry = json.loads(lines[-1])
            assert last_entry['message'] == ""
    
    def test_very_long_message(self, logger):
        """Test logging with very long message."""
        long_message = "x" * 10000
        logger.info(long_message, "test_module")
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            last_entry = json.loads(lines[-1])
            assert last_entry['message'] == long_message
    
    def test_special_characters_in_message(self, logger):
        """Test logging with special characters."""
        special_message = "Test message with special chars: éñüß@#$%^&*()"
        logger.info(special_message, "test_module")
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            last_entry = json.loads(lines[-1])
            assert last_entry['message'] == special_message
    
    def test_none_context(self, logger):
        """Test logging with None context."""
        logger.info("Test message", "test_module", context=None)
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            last_entry = json.loads(lines[-1])
            assert last_entry['context'] == {}


if __name__ == "__main__":
    pytest.main([__file__]) 