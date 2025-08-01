# EpiTuner Flowchart Index

## Complete System Overview

### **[Master Flowchart](master_flowchart.md)**
*Complete pipeline from inputs to outputs with Context Summary approach and dynamic model selection*

## Individual Script Flowcharts

### **Core Processing Scripts**
- **[Data Loader](data_loader/data_loader_flow_diagram.md)**
  - Data validation and cleaning
  - Schema checking and error handling
  - Context block generation

- **[Schema Mapper](schema_mapper/schema_mapper_flow_diagram.md)**
  - Rating standardization
  - Mapping validation
  - Metadata generation

- **[Formatter](formatter/formatter_flow_diagram.md)** - **ENHANCED**
  - **Context Summary Approach**: Pattern extraction from training data
  - **Smart Context Creation**: "Look for symptoms: fever, cough, etc."
  - **Transfer Learning**: Patterns transfer to new case evaluation
  - **Traditional Prompts**: Individual case prompts for fine-tuning

- **[Inference Runner](inference_runner/inference_runner_flow_diagram.md)**
  - Model predictions
  - Batch processing
  - Result parsing

- **[Fine Tuner](fine_tuner/fine_tuner_flow_diagram.md)**
  - Model training
  - Dataset processing
  - Configuration generation

- **[Contextualizer](contextualizer/contextualizer_flow_diagram.md)**
  - Few-shot learning
  - Meta-prompt construction
  - Contextual inference

### **Support Scripts**
- **[GUI Interface](gui/gui_flow_diagram.md)** - **ENHANCED**
  - **Dynamic Model Selection**: Dropdown with all available Ollama models
  - **Clean Interface**: Modern, intuitive design
  - **Context Summary Integration**: Default prompt formatting approach
  - **Low Power Mode**: Optimized for tablet/limited hardware
  - **Step-by-Step Workflow**: Intuitive navigation through all processes

- **[Config Manager](config_manager.md)**
  - Configuration management
  - Settings validation
  - Environment handling

- **[Debugging Logger](debugging_logger/debugging_logger_flow_diagram.md)**
  - Logging system
  - Error tracking
  - Debug information

## Quick Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| **Data Loader** | Data validation & cleaning | Raw CSV/Excel | Cleaned DataFrame |
| **Schema Mapper** | Rating standardization | Cleaned data | Mapped data |
| **Formatter** | **Context Summary** + Prompt creation | Mapped data | Pattern-based prompts |
| **Inference Runner** | Model predictions | Prompts | Predictions |
| **Fine Tuner** | Model training | Training data | Fine-tuned model |
| **Contextualizer** | Few-shot learning | Examples + query | Contextual predictions |
| **GUI** | **Dynamic Model Selection** + User interface | User input | Complete workflow |
| **Config Manager** | Settings management | Config files | System settings |
| **Debugging Logger** | Logging system | System events | Log files |

## Navigation

- **Start Here**: [Master Flowchart](master_flowchart.md) for complete system overview
- **Individual Scripts**: Click any script link above for detailed flow
- **All Documentation**: [Documentation Index](README.md)

---

*All flowcharts use relative paths and will work on any PC when the repository is cloned.* 