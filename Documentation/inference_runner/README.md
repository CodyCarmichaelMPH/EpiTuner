# Inference Runner Module Documentation

This folder contains comprehensive documentation for the Inference Runner module (`scripts/inference_runner.py`).

## Documentation Files

### [Complete Documentation](inference_runner_documentation.md)
**Function-by-function breakdown of every method and class**

- **Purpose**: Comprehensive documentation of every method, parameter, and behavior
- **Content**: 
  - Class structure and constants
  - All public and private methods
  - Custom exception classes
  - Usage examples and error handling
  - Performance characteristics
- **Audience**: Developers who need to understand or modify the inference runner
- **Length**: ~500 lines of detailed documentation

### ⚡ [Quick Reference](inference_runner_quick_reference.md)
**Fast reference guide for common operations**

- **Purpose**: Quick lookup for methods, parameters, and common patterns
- **Content**:
  - Key methods table
  - Model availability checking
  - Inference configuration options
  - Response parsing patterns
  - Common usage patterns
  - Error handling reference
- **Audience**: Developers who need quick access to syntax and patterns
- **Length**: ~100 lines of concise reference

### [Flow Diagrams](inference_runner_flow_diagram.md)
**Visual process flow and decision trees**

- **Purpose**: Understand the step-by-step process flow
- **Content**:
  - Overview flow diagram
  - Detailed process flows for each step
  - Error handling flows
  - Response parsing flows
  - Performance considerations
- **Audience**: Anyone wanting to understand the inference pipeline
- **Length**: ~300 lines of flow diagrams

### [Implementation Summary](inference_runner_summary.md)
**Executive overview and technical specifications**

- **Purpose**: High-level overview of the implementation
- **Content**:
  - Executive summary
  - Technical specifications
  - Quality assurance metrics
  - Integration points
  - Next steps
- **Audience**: Project managers and technical leads
- **Length**: ~170 lines of executive summary

## Quick Navigation

| Need | Go To | Description |
|------|-------|-------------|
| **Understand a specific function** | [inference_runner_documentation.md](inference_runner_documentation.md) | Detailed method-by-method breakdown |
| **Quick syntax reference** | [inference_runner_quick_reference.md](inference_runner_quick_reference.md) | Fast lookup for common operations |
| **Understand the process flow** | [inference_runner_flow_diagram.md](inference_runner_flow_diagram.md) | Visual representation of the pipeline |
| **Get executive overview** | [inference_runner_summary.md](inference_runner_summary.md) | High-level summary and technical specs |

## Module Status

- **Implementation**: `scripts/inference_runner.py` Complete
- **Tests**: `tests/test_inference_runner.py` 25+ tests, all passing
- **Documentation**: Complete (4 files)
- **Status**: Production-ready

## Related Files

- **Implementation**: `../../scripts/inference_runner.py`
- **Tests**: `../../tests/test_inference_runner.py`
- **Demo Script**: `../../demo_inference_runner.py`
- **Main Documentation**: `../README.md`

## Key Features

- **Ollama Integration**: Direct integration with Ollama CLI for model inference
- **Batch Processing**: Configurable batch sizes for efficient processing
- **Response Parsing**: Robust parsing of both JSON and unstructured responses
- **Error Handling**: Comprehensive error handling with retry logic
- **Model Validation**: Automatic model availability checking
- **Metadata Tracking**: Detailed metadata collection for each inference run
- **Multiple Output Formats**: Support for CSV and JSON output formats 