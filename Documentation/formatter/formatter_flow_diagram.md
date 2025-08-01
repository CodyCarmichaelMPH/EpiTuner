# Formatter Flow Diagram

## Prompt Creation and Context Summary Generation

```mermaid
graph TD
    %% Input
    A[Input DataFrame<br/>with Standardized_Rating] --> B{Validate Input Data}
    B -->|Valid| C[Create Context Block<br/>if missing]
    B -->|Invalid| D[Raise ValidationError]
    
    %% Context Block Creation
    C --> E[Combine Available Columns<br/>ChiefComplaintOrig, Discharge Diagnosis,<br/>TriageNotes, etc.]
    E --> F[Context_Block Created]
    
    %% Formatting Method Selection
    F --> G{Choose Formatting Method}
    
    %% Context Summary Path (NEW)
    G -->|Context Summary| H[Extract Key Findings<br/>from Training Data]
    H --> I[Separate Positive/Negative Cases]
    I --> J[Extract Common Patterns<br/>from Positive Cases]
    I --> K[Extract Common Patterns<br/>from Negative Cases]
    J --> L[Create Context Summary<br/>"Look for symptoms: fever, cough, etc."]
    K --> L
    L --> M[Generate Prompts with<br/>Context Summary]
    
    %% Traditional Path
    G -->|Traditional| N[Create Individual Prompts<br/>per Training Case]
    N --> O[Format with Template<br/>Include Rating & Rationale]
    
    %% Output
    M --> P[Context Summary Prompts<br/>with Pattern-Based Context]
    O --> Q[Traditional Training Prompts<br/>with Individual Cases]
    
    %% Prompt Statistics
    P --> R[Generate Prompt Statistics<br/>Count, Rating Distribution]
    Q --> R
    R --> S[Return Formatted Prompts]
    
    %% Styling
    classDef inputClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef errorClass fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef newFeatureClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
    
    class A inputClass
    class B,C,E,F,G,H,I,J,K,L,M,N,O,R processClass
    class P,Q,S outputClass
    class D errorClass
    class H,I,J,K,L,M newFeatureClass
```

## Context Summary Process Flow

### **1. Pattern Extraction**
- **Positive Cases**: Analyze cases rated > 0 to find common indicators
- **Negative Cases**: Analyze cases rated = 0 to find exclusion patterns
- **Term Frequency**: Extract most common medical terms and symptoms
- **Stop Word Filtering**: Remove common words to focus on meaningful terms

### **2. Context Summary Creation**
- **Topic Alignment**: "You are evaluating cases for respiratory diseases"
- **Positive Indicators**: "Look for: fever, cough, sore throat, chest pain"
- **Negative Indicators**: "Exclude: headache, abdominal pain, neurological symptoms"
- **Evaluation Instructions**: Clear rating criteria (0, 1, 2 scale)

### **3. Prompt Generation**
- **Context Application**: Use extracted patterns for all evaluation cases
- **Consistent Evaluation**: Same context applied to training and inference data
- **Transfer Learning**: Patterns learned from expert data applied to new cases

## Traditional vs Context Summary Approach

| Aspect | Traditional Approach | Context Summary Approach |
|--------|---------------------|-------------------------|
| **Training Data Usage** | Individual cases shown to model | Key patterns extracted and summarized |
| **Context** | Case-by-case evaluation | Pattern-based evaluation context |
| **Transfer Learning** | Limited | Strong - patterns transfer to new cases |
| **Prompt Length** | Long (full case details) | Short (pattern summary + case) |
| **Scalability** | Limited by context window | Efficient use of context window |
| **Interpretability** | Case-specific | Pattern-based reasoning |

## Key Methods

### **extract_key_findings()**
- Analyzes training data to extract common patterns
- Separates positive and negative case indicators
- Creates comprehensive evaluation context

### **create_context_summary_prompts()**
- Generates prompts using pattern-based context
- Applies consistent evaluation criteria
- Supports both training and inference data

### **create_training_prompts()** (Traditional)
- Creates individual prompts for each training case
- Includes expert ratings and rationales
- Suitable for fine-tuning approaches

### **create_inference_prompts()**
- Generates prompts for model prediction
- Uses context summary or individual case approach
- Optimized for batch processing

## Usage Examples

### **Context Summary Example**
```
CONTEXT:
You are evaluating whether cases align with the topic: respiratory diseases

Look for these indicators of respiratory diseases:
- fever
- cough
- sore throat
- chest pain
- shortness of breath

These patterns suggest the case does NOT align with respiratory diseases:
- headache
- abdominal pain
- neurological symptoms

Evaluation Instructions:
- Rate 0: Case does NOT align with respiratory diseases
- Rate 1: Case partially aligns with respiratory diseases
- Rate 2: Case clearly aligns with respiratory diseases

CASE TO EVALUATE:
Patient presents with severe cough, fever 101F, sore throat...

Please evaluate this case based on the context above.
Rating: <0, 1, or 2>
Rationale: <your explanation>
```

### **Traditional Example**
```
INPUT:
You are evaluating whether this case aligns with topic 'respiratory diseases'.

Patient Data:
Patient presents with severe cough, fever 101F, sore throat...

OUTPUT (Expert Provided):
Rating: 2
Rationale: Clear respiratory symptoms: cough, fever, sore throat
```

## Benefits of Context Summary Approach

### **Efficiency**
- **Reduced Context Usage**: Pattern summary vs full cases
- **Faster Processing**: Shorter prompts, faster inference
- **Better Scaling**: Handles larger datasets efficiently

### **Transfer Learning**
- **Pattern Transfer**: Learned patterns apply to new cases
- **Consistent Evaluation**: Same criteria across all cases
- **Knowledge Distillation**: Expert knowledge captured in patterns

### **Interpretability**
- **Clear Reasoning**: Model explains based on identified patterns
- **Auditable Decisions**: Pattern-based evaluation is transparent
- **Expert Alignment**: Uses same patterns experts identified

---

*The Formatter module now supports both traditional and Context Summary approaches for maximum flexibility.* 