# GUI Flow Diagram

## User Interface and Workflow Orchestration

```mermaid
graph TD
    %% Initial Setup
    A[Launch GUI<br/>gui/epituner_gui.py] --> B[Initialize Session State<br/>Setup Page Configuration]
    B --> C[Render Header<br/>Progress Indicator]
    C --> D[Render Sidebar<br/>Navigation & Settings]
    
    %% Sidebar Components
    D --> E[Step Navigation<br/>Current Step Display]
    E --> F[Model Settings<br/>Dynamic Model Selection]
    F --> G[Get Available Models<br/>from Ollama Server]
    G --> H{Models Available?}
    H -->|Yes| I[Display Model Dropdown<br/>llama3.2:1b, mistral:latest, etc.]
    H -->|No| J[Manual Text Input<br/>with Warning]
    I --> K[System Settings<br/>Debug Mode, Batch Size]
    J --> K
    
    %% Step 1: Data Upload
    K --> L[Step 1: Data Upload]
    L --> M[File Upload Widget<br/>CSV Files Only]
    M --> N{File Uploaded?}
    N -->|Yes| O[Load & Validate Data<br/>DataLoader]
    N -->|No| P[Show Upload Instructions]
    O --> Q{Validation Passed?}
    Q -->|Yes| R[Display Data Preview<br/>Success Message]
    Q -->|No| S[Show Validation Errors]
    R --> T[Proceed to Step 2]
    S --> U[Return to Upload]
    
    %% Step 2: Schema Mapping
    T --> V[Step 2: Schema Mapping]
    V --> W[Display Expert Ratings<br/>Unique Values Found]
    W --> X[Mapping Method Selection<br/>Manual, Predefined, Auto-detect]
    X --> Y[Rating Standardization<br/>0=No Match, 1=Partial, 2=Clear]
    Y --> Z[Apply Rating Mapping<br/>SchemaMapper]
    Z --> AA[Display Mapping Results<br/>Success Message]
    AA --> BB[Target Topic Selection<br/>"respiratory diseases"]
    BB --> CC[Proceed to Step 3]
    
    %% Step 3: Prompt Formatting
    CC --> DD[Step 3: Prompt Formatting]
    DD --> EE[Formatting Method Selection<br/>Context Summary (Recommended)]
    EE --> FF{Method Selected}
    FF -->|Context Summary| GG[Extract Key Patterns<br/>from Training Data]
    FF -->|Traditional| HH[Create Individual Prompts<br/>per Case]
    GG --> II[Generate Context Summary<br/>"Look for symptoms: fever, cough"]
    HH --> JJ[Standard Prompt Creation<br/>with Templates]
    II --> KK[Display Context Summary<br/>Extracted Patterns]
    JJ --> LL[Display Prompt Preview<br/>Individual Cases]
    KK --> MM[Proceed to Step 4]
    LL --> MM
    
    %% Step 4: Model Operations
    MM --> NN[Step 4: Model Operations]
    NN --> OO[Operation Selection<br/>Inference, Fine-tuning, Contextualization]
    OO --> PP[Model Configuration<br/>Dynamic Dropdown from Step 2]
    PP --> QQ[Start Operation<br/>Progress Tracking]
    QQ --> RR{Operation Type}
    RR -->|Inference| SS[Run Inference<br/>InferenceRunner]
    RR -->|Fine-tuning| TT[Fine-tune Model<br/>FineTuner]
    RR -->|Contextualization| UU[Create Contextualizer<br/>Contextualizer]
    SS --> VV[Display Results<br/>Success/Failure Counts]
    TT --> VV
    UU --> VV
    VV --> WW[Proceed to Step 5]
    
    %% Step 5: Results & Export
    WW --> XX[Step 5: Results & Export]
    XX --> YY[Results Overview<br/>Statistics & Metrics]
    YY --> ZZ[Inference Results<br/>Predictions & Analysis]
    ZZ --> AAA[Export Options<br/>CSV, JSON, Reports]
    AAA --> BBB[Session Summary<br/>Complete Analysis]
    BBB --> CCC[Download Results<br/>All Formats]
    
    %% Styling
    classDef setupClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef stepClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef processClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef outputClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef newFeatureClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
    classDef errorClass fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class A,B,C,D setupClass
    class L,V,DD,NN,XX stepClass
    class E,F,G,H,I,J,K,M,O,P,Q,R,S,T,W,X,Y,Z,AA,BB,CC,EE,FF,GG,HH,II,JJ,KK,LL,MM,OO,PP,QQ,RR,SS,TT,UU,VV,WW,YY,ZZ,AAA,BBB,CCC processClass
    class U,AA,VV,CCC outputClass
    class G,I,GG,II newFeatureClass
    class S,U errorClass
```

## GUI Components and Features

### **Dynamic Model Selection**
- **Ollama Integration**: Fetches available models from server
- **Smart Dropdown**: Shows all available models (llama3.2:1b, mistral:latest, etc.)
- **Fallback Handling**: Manual input if server unavailable
- **Default Selection**: Prioritizes llama3.2:1b for low power mode

### **Context Summary Approach**
- **Pattern Extraction**: Analyzes training data for key indicators
- **Smart Context**: Creates evaluation context like "Look for respiratory symptoms"
- **Transfer Learning**: Patterns transfer to new case evaluation
- **Clean Interface**: Modern, intuitive design

### **Step-by-Step Workflow**
1. **Data Upload**: CSV validation and preview
2. **Schema Mapping**: Rating standardization with user-friendly interface
3. **Prompt Formatting**: Context Summary (recommended) or traditional approach
4. **Model Operations**: Dynamic model selection and operation execution
5. **Results & Export**: Comprehensive analysis and export options

## Key Features

### **Professional Interface**
- **Modern Design**: Clean, intuitive appearance
- **Consistent Styling**: Unified design throughout
- **Clear Navigation**: Step-by-step progress indicator
- **Error Handling**: Graceful error messages and recovery

### **Smart Defaults**
- **Low Power Mode**: Optimized for tablet/limited hardware
- **Model Selection**: Automatically selects appropriate models
- **Context Summary**: Default prompt formatting approach
- **Batch Processing**: Efficient handling of large datasets

### **User Experience**
- **Progress Tracking**: Visual progress through workflow steps
- **Data Preview**: Real-time data validation and preview
- **Context Display**: Shows extracted patterns and context summary
- **Result Analysis**: Comprehensive results with metrics and visualizations

## Integration Points

### **Data Processing**
- **DataLoader**: File upload and validation
- **SchemaMapper**: Rating standardization
- **Formatter**: Context Summary and prompt creation

### **Model Operations**
- **InferenceRunner**: Model predictions with progress tracking
- **FineTuner**: Model training and configuration
- **Contextualizer**: Few-shot learning approach

### **Output Generation**
- **Results Analysis**: Performance metrics and confusion matrices
- **Export Options**: Multiple format support (CSV, JSON, reports)
- **Session Reports**: Complete workflow documentation

## Error Handling

### **Validation Errors**
- **File Format**: CSV validation and error messages
- **Data Schema**: Missing column detection and guidance
- **Model Availability**: Ollama server connection issues
- **Operation Failures**: Graceful error recovery and user feedback

### **User Guidance**
- **Clear Instructions**: Step-by-step guidance for each operation
- **Error Recovery**: Suggestions for fixing common issues
- **Progress Feedback**: Real-time status updates during operations
- **Help Text**: Contextual help and explanations

---

*The GUI provides a professional, user-friendly interface for the complete EpiTuner workflow with dynamic model selection and Context Summary approach.* 