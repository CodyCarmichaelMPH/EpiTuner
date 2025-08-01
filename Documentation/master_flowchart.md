# EpiTuner Master Flowchart

## Complete Pipeline: From Inputs to Outputs

```mermaid
graph TD
    %% Input Layer
    A[Raw Dataset<br/>CSV/Excel Files] --> B[Data Loader<br/>scripts/data_loader.py]
    C[Configuration<br/>config/settings.json] --> B
    
    %% Data Processing Layer
    B --> D[Schema Validation<br/>Required Columns<br/>Data Types<br/>Rating Values]
    D --> E[Schema Mapping<br/>scripts/schema_mapper.py<br/>[View Details](../Documentation/schema_mapper/schema_mapper_flow_diagram.md)]
    E --> F[Prompt Formatting<br/>scripts/formatter.py<br/>[View Details](../Documentation/formatter/formatter_flow_diagram.md)]
    
    %% New Context Summary Flow
    F --> F1{Formatting Method}
    F1 -->|Context Summary| F2[Extract Key Patterns<br/>From Training Data]
    F1 -->|Traditional| F3[Individual Case Prompts]
    F2 --> F4[Create Context Summary<br/>"Look for symptoms: fever, cough, etc."]
    F3 --> F5[Standard Prompts<br/>Per Case]
    F4 --> G
    F5 --> G
    
    %% Model Operations Layer
    G{Model Operation<br/>Selection}
    G --> H[Inference<br/>scripts/inference_runner.py<br/>[View Details](../Documentation/inference_runner/inference_runner_flow_diagram.md)]
    G --> I[Fine-tuning<br/>scripts/fine_tuner.py<br/>[View Details](../Documentation/fine_tuner/fine_tuner_flow_diagram.md)]
    G --> J[Contextualization<br/>scripts/contextualizer.py<br/>[View Details](../Documentation/contextualizer/contextualizer_flow_diagram.md)]
    
    %% GUI Layer
    K[GUI Interface<br/>gui/epituner_gui.py<br/>[View Details](../Documentation/gui/gui_flow_diagram.md)] --> G
    
    %% Output Layer
    H --> L[Inference Results<br/>outputs/*_results.csv]
    I --> M[Fine-tuned Model<br/>outputs/*_config.yaml]
    J --> N[Contextual Results<br/>outputs/contextual_*.csv]
    
    %% Logging & Debugging
    O[Debugging Logger<br/>scripts/debugging_logger.py<br/>[View Details](../Documentation/debugging_logger/debugging_logger_flow_diagram.md)] --> P[Log Files<br/>*.log]
    
    %% Configuration Management
    Q[Config Manager<br/>scripts/config_manager.py] --> R[System Settings<br/>Model Configs<br/>API Keys]
    
    %% Final Outputs
    L --> S[Evaluation Metrics<br/>Accuracy, Precision, Recall]
    M --> S
    N --> S
    S --> T[Session Report<br/>Complete Analysis]
    
    %% Styling
    classDef inputClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef guiClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef configClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef newFeatureClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
    
    class A,C inputClass
    class B,D,E,F,F1,F2,F3,F4,F5,H,I,J,O,Q processClass
    class L,M,N,P,R,S,T outputClass
    class K guiClass
    class F2,F4 newFeatureClass
```

## Detailed Process Flow

### **1. Input Processing**
- **Raw Dataset**: CSV/Excel files with expert ratings and text data
- **Configuration**: Settings, model parameters, API configurations
- **Data Loader**: Validates schema, cleans data, extracts unique ratings

### **2. Data Transformation**
- **Schema Mapping**: Converts non-standard ratings to standardized format
- **Prompt Formatting**: Creates structured prompts for LLM processing
- **Context Block Generation**: Combines relevant text fields automatically

### **3. Context Summary Approach (NEW)**
- **Pattern Extraction**: Analyzes expert-rated data to find key indicators
- **Context Creation**: Generates summary like "Look for respiratory symptoms: fever, cough, sore throat"
- **Smart Evaluation**: Uses extracted patterns to evaluate new cases

### **4. Model Operations**
- **Inference**: Direct model predictions using context summaries
- **Fine-tuning**: Model training with custom datasets
- **Contextualization**: Few-shot learning approach for base models

### **5. Output Generation**
- **Results Files**: CSV outputs with predictions and metrics
- **Model Configs**: Fine-tuned model configurations
- **Evaluation Metrics**: Performance analysis and comparisons

## Individual Script Flowcharts

### **Core Processing Scripts**
- **[Data Loader](../Documentation/data_loader/data_loader_flow_diagram.md)**: Data validation and cleaning
- **[Schema Mapper](../Documentation/schema_mapper/schema_mapper_flow_diagram.md)**: Rating standardization
- **[Formatter](../Documentation/formatter/formatter_flow_diagram.md)**: Prompt creation with Context Summary
- **[Inference Runner](../Documentation/inference_runner/inference_runner_flow_diagram.md)**: Model predictions
- **[Fine Tuner](../Documentation/fine_tuner/fine_tuner_flow_diagram.md)**: Model training
- **[Contextualizer](../Documentation/contextualizer/contextualizer_flow_diagram.md)**: Few-shot learning

### **Support Scripts**
- **[GUI Interface](../Documentation/gui/gui_flow_diagram.md)**: User interface with dynamic model selection
- **[Config Manager](../Documentation/config_manager.md)**: Settings and configuration management
- **[Debugging Logger](../Documentation/debugging_logger/debugging_logger_flow_diagram.md)**: Logging and error tracking

## Data Flow Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **Data Loading** | Raw CSV/Excel | Validation & Cleaning | Cleaned DataFrame |
| **Schema Mapping** | Cleaned Data | Rating Standardization | Mapped DataFrame |
| **Prompt Formatting** | Mapped Data | Context Creation | Formatted Prompts |
| **Context Summary** | Training Data | Pattern Extraction | Context Summary |
| **Model Operations** | Formatted Prompts | LLM Processing | Predictions & Results |
| **Evaluation** | Results | Metrics Calculation | Performance Analysis |

## Key Features

### **Automated Processing**
- Automatic schema validation
- Dynamic context block generation
- Intelligent rating mapping
- Progress tracking and logging

### **Context Summary Approach**
- Pattern extraction from expert-rated data
- Smart context creation for evaluation
- Transfer learning between models
- Modern, intuitive interface

### **Flexible Model Support**
- Dynamic model selection from Ollama
- Multiple inference modes
- Fine-tuning capabilities
- Contextualization fallback
- Low power mode optimization

### **Comprehensive Outputs**
- Detailed results files
- Performance metrics
- Model configurations
- Session reports

## Usage Workflow

1. **Upload Data** → Data Loader validates and processes
2. **Map Ratings** → Schema Mapper standardizes ratings
3. **Format Prompts** → Choose Context Summary (recommended) or traditional approach
4. **Run Models** → Choose inference, fine-tuning, or contextualization
5. **View Results** → Analyze performance and export findings

---

*This master flowchart provides a complete overview of the EpiTuner pipeline. Click on any script link to view detailed flow diagrams for individual components.* 