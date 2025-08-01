# Contextualizer Flow Diagram

## Overview

The Contextualizer provides a fallback approach for scenarios where fine-tuning is not available or cost-effective. It uses few-shot learning with structured meta-prompts to guide base Ollama models.

## Main Flow

```mermaid
graph TD
    A[Input Dataset] --> B[Data Validation]
    B --> C[Few-shot Sampling]
    C --> D[Schema Mapping]
    D --> E[Meta-prompt Construction]
    E --> F[Ollama Inference]
    F --> G[Response Parsing]
    G --> H[Structured Output]
    
    B --> B1{Valid Data?}
    B1 -->|No| B2[Error Handling]
    B1 -->|Yes| C
    
    C --> C1{Enough Examples?}
    C1 -->|No| C2[Reduce Context Size]
    C2 --> C
    C1 -->|Yes| D
    
    F --> F1{Model Available?}
    F1 -->|No| F2[Model Error]
    F1 -->|Yes| F3[Send Prompt]
    F3 --> G
    
    G --> G1{Valid Response?}
    G1 -->|No| G2[Retry/Error]
    G2 --> F
    G1 -->|Yes| H
```

## Detailed Process Flow

### 1. Data Input and Validation

```mermaid
graph LR
    A[CSV Dataset] --> B[Data Loader]
    B --> C[Schema Validation]
    C --> D[Data Cleaning]
    D --> E[Context Block Creation]
    E --> F[Validated Dataset]
    
    C --> C1{Required Fields?}
    C1 -->|No| C2[Schema Error]
    C1 -->|Yes| D
```

### 2. Few-shot Example Sampling

```mermaid
graph TD
    A[Validated Dataset] --> B[Filter Valid Examples]
    B --> C[Apply Rating Mapping]
    C --> D[Balance Across Ratings]
    D --> E[Limit Context Size]
    E --> F[Selected Examples]
    
    B --> B1{Has Expert Rating?}
    B1 -->|No| B2[Skip Row]
    B1 -->|Yes| B3{Has Rationale?}
    B3 -->|No| B2
    B3 -->|Yes| C
    
    D --> D1{Balanced?}
    D1 -->|No| D2[Adjust Sampling]
    D2 --> D
    D1 -->|Yes| E
```

### 3. Meta-prompt Construction

```mermaid
graph LR
    A[Schema Description] --> D[Meta-prompt]
    B[Few-shot Examples] --> D
    C[Query Row] --> D
    
    D --> E[Structured Prompt]
    E --> F[Send to Ollama]
```

### 4. Inference and Response Processing

```mermaid
graph TD
    A[Constructed Prompt] --> B[Ollama CLI Call]
    B --> C[Model Response]
    C --> D[Response Parsing]
    D --> E[Extract Rating]
    E --> F[Extract Rationale]
    F --> G[Structured Result]
    
    B --> B1{Model Available?}
    B1 -->|No| B2[Model Error]
    B1 -->|Yes| C
    
    D --> D1{Valid Format?}
    D1 -->|No| D2[Retry/Error]
    D2 --> B
    D1 -->|Yes| E
```

## Integration with Existing Pipeline

```mermaid
graph TD
    A[Raw CSV Data] --> B[Data Loader]
    B --> C[Schema Mapper]
    C --> D{Use Fine-tuning?}
    
    D -->|Yes| E[Formatter PromptBuilder]
    E --> F[Fine-tuner]
    F --> G[Inference Runner]
    
    D -->|No| H[Contextualizer]
    H --> I[Direct Ollama Inference]
    
    G --> J[Results]
    I --> J
    
    J --> K[Output Files]
```

## Error Handling Flow

```mermaid
graph TD
    A[Process Start] --> B{Data Valid?}
    B -->|No| C[Data Error]
    B -->|Yes| D{Examples Found?}
    
    D -->|No| E[Sampling Error]
    D -->|Yes| F{Model Available?}
    
    F -->|No| G[Model Error]
    F -->|Yes| H{Response Valid?}
    
    H -->|No| I[Parsing Error]
    H -->|Yes| J[Success]
    
    C --> K[Log Error]
    E --> K
    G --> K
    I --> K
    
    K --> L[Return Error Result]
    J --> M[Return Success Result]
```

## Configuration Flow

```mermaid
graph LR
    A[Config File] --> B[Config Manager]
    B --> C[Contextualizer Settings]
    C --> D[Max Context Rows]
    C --> E[Timeout]
    C --> F[Max Retries]
    C --> G[Prompt Template]
    
    D --> H[Sampling Logic]
    E --> I[Inference Control]
    F --> I
    G --> J[Prompt Construction]
```

## Output Structure

```mermaid
graph TD
    A[Processing Complete] --> B[Results DataFrame]
    B --> C[CSV Export]
    B --> D[Metadata JSON]
    
    C --> E[contextual_evaluation_results.csv]
    D --> F[contextual_evaluation_metadata.json]
    
    B --> G[Result Fields]
    G --> H[C_BioSense_ID]
    G --> I[Prediction]
    G --> J[Confidence]
    G --> K[Rationale]
    
    D --> L[Metadata Fields]
    L --> M[Evaluation Type]
    L --> N[Model Name]
    L --> O[Success Rate]
    L --> P[Processing Time]
    L --> Q[Prediction Distribution]
```

## Performance Monitoring

```mermaid
graph LR
    A[Start Processing] --> B[Log Start Time]
    B --> C[Process Each Row]
    C --> D[Log Progress]
    D --> E{More Rows?}
    E -->|Yes| C
    E -->|No| F[Calculate Statistics]
    F --> G[Save Metadata]
    G --> H[Complete]
```

## Key Decision Points

1. **Data Validation**: Ensures required fields and data quality
2. **Example Sampling**: Balances representation across rating categories
3. **Model Availability**: Checks if specified Ollama model is available
4. **Response Validation**: Ensures model output can be parsed
5. **Error Recovery**: Implements retry logic and graceful degradation

## Success Metrics

- **Success Rate**: Percentage of successfully processed rows
- **Processing Time**: Total time for dataset evaluation
- **Prediction Distribution**: Spread of predictions across rating categories
- **Error Types**: Classification of different error conditions
- **Model Performance**: Response quality and consistency 