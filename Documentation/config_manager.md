# Config Manager Flowchart

## Configuration Management System

```mermaid
graph TD
    %% Input Sources
    A[config/settings.json] --> B[Config Manager<br/>scripts/config_manager.py]
    C[Environment Variables] --> B
    D[Command Line Args] --> B
    
    %% Configuration Loading
    B --> E[Load Configuration]
    E --> F{Validate Config}
    F -->|Invalid| G[Use Defaults]
    F -->|Valid| H[Config Loaded]
    
    %% Configuration Types
    H --> I[Model Configs<br/>Ollama Models<br/>API Endpoints]
    H --> J[Data Configs<br/>Schema Validation<br/>File Formats]
    H --> K[System Configs<br/>Timeouts<br/>Batch Sizes]
    H --> L[Logging Configs<br/>Debug Mode<br/>Log Levels]
    
    %% Configuration Access
    I --> M[Get Model Config]
    J --> N[Get Data Config]
    K --> O[Get System Config]
    L --> P[Get Logging Config]
    
    %% Configuration Updates
    Q[Update Request] --> R[Validate Changes]
    R -->|Valid| S[Save Config]
    R -->|Invalid| T[Reject Changes]
    
    %% Output
    M --> U[Model Settings]
    N --> V[Data Settings]
    O --> W[System Settings]
    P --> X[Logging Settings]
    S --> Y[Updated Config File]
    
    %% Styling
    classDef inputClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef errorClass fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class A,C,D inputClass
    class B,E,F,H,I,J,K,L,M,N,O,P,Q,R processClass
    class U,V,W,X,Y outputClass
    class G,T errorClass
```

## Configuration Management Process

### **1. Configuration Sources**
- **Settings File**: `config/settings.json` - Primary configuration
- **Environment Variables**: Override settings for deployment
- **Command Line**: Runtime configuration adjustments

### **2. Configuration Validation**
- **Schema Validation**: Ensures required fields are present
- **Type Checking**: Validates data types and formats
- **Default Fallback**: Uses sensible defaults for missing values

### **3. Configuration Types**
- **Model Configs**: Ollama model settings, API endpoints
- **Data Configs**: Schema validation rules, file format settings
- **System Configs**: Timeouts, batch sizes, performance settings
- **Logging Configs**: Debug mode, log levels, output formats

### **4. Configuration Access**
- **Get Methods**: Retrieve specific configuration sections
- **Update Methods**: Modify configurations with validation
- **Save Methods**: Persist changes to configuration files

## Configuration Structure

```json
{
  "models": {
    "ollama": {
      "base_url": "http://localhost:11434",
      "timeout": 180,
      "models": ["llama3.2:1b", "mistral:latest"]
    }
  },
  "data": {
    "schema_validation": true,
    "required_columns": ["Expert Rating", "Rationale of Rating"],
    "file_formats": ["csv", "xlsx"]
  },
  "system": {
    "batch_size": 1,
    "max_retries": 2,
    "debug_mode": true
  },
  "logging": {
    "level": "INFO",
    "file_output": true,
    "console_output": true
  }
}
```

## Key Features

### **Flexible Configuration**
- Multiple configuration sources
- Environment variable overrides
- Runtime configuration updates
- Validation and error handling

### **Type Safety**
- Schema validation
- Type checking
- Default value fallbacks
- Error reporting

### **Easy Access**
- Simple getter methods
- Configuration sections
- Nested configuration support
- Automatic reloading

---

*The Config Manager provides centralized configuration management for the entire EpiTuner system.* 