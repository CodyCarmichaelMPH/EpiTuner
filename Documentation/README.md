# EpiTuner Documentation

This folder contains comprehensive documentation for the EpiTuner project modules, organized by script for better navigation and scalability.

## Documentation Structure

```
Documentation/
├── README.md                    # This file - Documentation overview
├── data_loader/                 # Data Loader module documentation
│   ├── data_loader_documentation.md
│   ├── data_loader_quick_reference.md
│   ├── data_loader_flow_diagram.md
│   └── data_loader_summary.md
├── schema_mapper/               # Schema Mapper module documentation (future)
├── formatter/                   # Formatter module documentation (future)
├── inference_runner/            # Inference Runner module documentation (future)
├── fine_tuner/                  # Fine Tuner module documentation (future)
└── gui/                         # GUI module documentation (future)
```

## Module Documentation

### [Data Loader Module](data_loader/)
**Complete documentation for the data loading and preprocessing module**

- **Implementation**: `scripts/data_loader.py`
- **Tests**: `tests/test_data_loader.py` (25 tests, all passing)
- **Status**: Complete and production-ready

**Documentation Files:**
- **[Complete Documentation](data_loader/data_loader_documentation.md)** - Function-by-function breakdown
- **[Quick Reference](data_loader/data_loader_quick_reference.md)** - Fast syntax lookup
- **[Flow Diagrams](data_loader/data_loader_flow_diagram.md)** - Process flows and decision trees
- **[Implementation Summary](data_loader/data_loader_summary.md)** - Executive overview

### [Schema Mapper Module](schema_mapper/)
**Complete documentation for the rating standardization and mapping module**

- **Implementation**: `scripts/schema_mapper.py`
- **Tests**: `tests/test_schema_mapper.py` (14 tests, all passing)
- **Status**: Complete and production-ready

**Documentation Files:**
- **[Complete Documentation](schema_mapper/README.md)** - Comprehensive module documentation
- **Integration**: Seamlessly integrates with Data Loader module
- **Purpose**: Transforms non-standard expert ratings into standardized schema

### [Formatter Module](formatter/) - **ENHANCED**
**Complete documentation for the Context Summary and prompt formatting module**

- **Implementation**: `scripts/formatter.py`
- **Tests**: `tests/test_formatter.py` (comprehensive test coverage)
- **Status**: Complete and production-ready with Context Summary approach

**Documentation Files:**
- **[Complete Documentation](formatter/formatter_documentation.md)** - Function-by-function breakdown
- **[Quick Reference](formatter/formatter_quick_reference.md)** - Fast syntax lookup
- **[Flow Diagrams](formatter/formatter_flow_diagram.md)** - Context Summary process flows
- **[Implementation Summary](formatter/formatter_summary.md)** - Executive overview

**New Features:**
- **Context Summary Approach**: Extract key patterns from training data
- **Pattern-Based Evaluation**: "Look for respiratory symptoms: fever, cough, etc."
- **Transfer Learning**: Patterns transfer to new case evaluation
- **Smart Context Creation**: Automatic pattern extraction and summarization

### [GUI Interface](gui/) - **NEW**
**Complete documentation for the professional user interface**

- **Implementation**: `gui/epituner_gui.py`
- **Status**: Complete and production-ready

**Documentation Files:**
- **[Complete Documentation](gui/gui_documentation.md)** - Function-by-function breakdown
- **[Quick Reference](gui/gui_quick_reference.md)** - Fast usage guide
- **[Flow Diagrams](gui/gui_flow_diagram.md)** - User interface workflow
- **[Implementation Summary](gui/gui_summary.md)** - Executive overview

**Key Features:**
- **Dynamic Model Selection**: Dropdown with all available Ollama models
- **Clean Interface**: Modern, intuitive design
- **Context Summary Integration**: Default prompt formatting approach
- **Low Power Mode**: Optimized for tablet/limited hardware
- **Step-by-Step Workflow**: Intuitive navigation through all processes

### Other Modules (Available)
- **Inference Runner**: `Documentation/inference_runner/`
- **Fine Tuner**: `Documentation/fine_tuner/`
- **Contextualizer**: `Documentation/contextualizer/`

## Quick Navigation

| Need | Go To | Description |
|------|-------|-------------|
| **Data Loader - Understand functions** | [data_loader/data_loader_documentation.md](data_loader/data_loader_documentation.md) | Detailed method-by-method breakdown |
| **Data Loader - Quick syntax** | [data_loader/data_loader_quick_reference.md](data_loader/data_loader_quick_reference.md) | Fast lookup for common operations |
| **Data Loader - Process flow** | [data_loader/data_loader_flow_diagram.md](data_loader/data_loader_flow_diagram.md) | Visual representation of the pipeline |
| **Data Loader - Overview** | [data_loader/data_loader_summary.md](data_loader/data_loader_summary.md) | Executive summary and technical specs |
| **Schema Mapper - Documentation** | [schema_mapper/README.md](schema_mapper/README.md) | Complete module documentation and usage |
| **Integration Guide** | [integration_guide.md](integration_guide.md) | How Data Loader and Schema Mapper work together |

## Documentation Standards

Each module folder follows this structure:

```
module_name/
├── module_documentation.md      # Complete function-by-function documentation
├── module_quick_reference.md    # Quick syntax and usage reference
├── module_flow_diagram.md       # Process flows and decision trees
└── module_summary.md            # Executive overview and technical specs
```

### Documentation Requirements

1. **Function Documentation**: Every public method includes purpose, parameters, return values, and examples
2. **Error Handling**: All possible exceptions and their solutions are documented
3. **Usage Examples**: Practical code examples for common use cases
4. **Performance Notes**: Memory and speed characteristics where relevant
5. **Testing Coverage**: References to corresponding test files
6. **Integration Points**: How the module connects to other parts of the system

## Contributing to Documentation

When adding new modules or updating existing ones:

1. **Create module folder**: `Documentation/module_name/`
2. **Follow naming convention**: `module_name_documentation.md`, etc.
3. **Update this README** with new module entries
4. **Follow established format** for consistency
5. **Include practical examples** for all public methods
6. **Document error conditions** and their solutions
7. **Add flow diagrams** for complex processes

## Related Files

- **Main README**: `../README.md` - Project overview and setup
- **Cursor Rules**: `../Docs/.cursorrules` - Development standards
- **Functional Overview**: `../Docs/master_functional_overview.txt` - System architecture
- **Sample Dataset**: `../data/sample_dataset.csv` - Test data
- **Demo Script**: `../demo_data_loader.py` - Working examples

## Future Documentation Structure

As we implement more modules, the documentation will grow to include:

```
Documentation/
├── README.md
├── data_loader/                 Complete
├── schema_mapper/               Complete
├── formatter/                   Planned
├── prompt_builder/              Planned
├── inference_runner/            Planned
├── fine_tuner/                  Planned
├── model_packager/              Planned
├── utils/                       Planned
└── gui/                         Planned
```

This structure ensures:
- **Scalability**: Easy to add new modules
- **Organization**: Clear separation by functionality
- **Navigation**: Intuitive folder structure
- **Consistency**: Standardized documentation format 