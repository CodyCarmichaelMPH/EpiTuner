# Inference Runner - Flow Diagrams

## Overview Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Data    │───▶│  Model Check     │───▶│  Inference      │
│                 │    │                  │    │                 │
│ • DataFrame     │    │ • Availability   │    │ • Batch Process │
│ • Model Name    │    │ • Metadata       │    │ • Parse Response│
│ • Config        │    │ • Validation     │    │ • Error Handle  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Error Exit     │    │   Results       │
                       │                  │    │                 │
                       │ • ModelNotFound  │    │ • DataFrame     │
                       │ • Validation     │    │ • Metadata      │
                       │ • Logging        │    │ • Files         │
                       └──────────────────┘    └─────────────────┘
```

## Detailed Process Flows

### 1. Initialization Flow

```
┌─────────────────┐
│   Start Init    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Set Parameters  │
│                 │
│ • debug_mode    │
│ • batch_size    │
│ • timeout       │
│ • max_retries   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Setup Logging   │
│                 │
│ • Level config  │
│ • Handlers      │
│ • File output   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Initialize      │
│ Metadata Store  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Ready State   │
└─────────────────┘
```

### 2. Model Availability Check Flow

```
┌─────────────────┐
│ Check Model     │
│ Availability    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Execute         │
│ 'ollama list'   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Check Return    │
│ Code            │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Success │ │ Failure │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Search  │ │ Log     │
│ Model   │ │ Error   │
│ Name    │ │         │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Found?  │ │ Return  │
└────┬────┘ │ False   │
     │      └─────────┘
     ▼
┌─────────┐
│ Return  │
│ True    │
└─────────┘
```

### 3. Model Metadata Retrieval Flow

```
┌─────────────────┐
│ Get Model       │
│ Metadata        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Execute         │
│ 'ollama show'   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Check Return    │
│ Code            │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Success │ │ Failure │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Parse   │ │ Raise   │
│ Output  │ │ Model   │
│         │ │ Not     │
└────┬────┘ │ Found   │
     │      └─────────┘
     ▼
┌─────────┐
│ Extract │
│ Fields  │
└────┬────┘
     │
     ▼
┌─────────┐
│ Store   │
│ Metadata│
└─────────┘
```

### 4. Single Inference Flow

```
┌─────────────────┐
│ Run Inference   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ For Each Retry  │
│ Attempt         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Execute         │
│ 'ollama run'    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Check Return    │
│ Code            │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Success │ │ Failure │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Parse   │ │ Retry   │
│ Response│ │ Logic   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Return  │ │ Max     │
│ Result  │ │ Retries │
└─────────┘ │ Reached?│
            └────┬────┘
                 │
                 ▼
            ┌─────────┐
            │ Raise   │
            │ Error   │
            └─────────┘
```

### 5. Response Parsing Flow

```
┌─────────────────┐
│ Parse Response  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Try JSON        │
│ Parsing         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ JSON Valid?     │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Yes     │ │ No      │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Extract │ │ Regex   │
│ Fields  │ │ Parsing │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Return  │ │ Extract │
│ Result  │ │ Fields  │
└─────────┘ └────┬────┘
                 │
                 ▼
┌─────────────────┐
│ Fields Found?   │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Yes     │ │ No      │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Return  │ │ Raise   │
│ Result  │ │ Parse   │
└─────────┘ │ Error   │
            └─────────┘
```

### 6. Batch Processing Flow

```
┌─────────────────┐
│ Batch Inference │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Split into      │
│ Batches         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ For Each Batch  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Process Batch   │
│ Prompts         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ For Each Prompt │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Run Inference   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Success?        │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Yes     │ │ No      │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Add     │ │ Add     │
│ Result  │ │ Error   │
└────┬────┘ │ Result  │
     │      └────┬────┘
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────┐
│ More Batches?   │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Yes     │ │ No      │
└─────────┘ └────┬────┘
                 │
                 ▼
┌─────────────────┐
│ Return All      │
│ Results         │
└─────────────────┘
```

### 7. Dataset Processing Flow

```
┌─────────────────┐
│ Process Dataset │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Validate Input  │
│                 │
│ • Check prompt  │
│   column        │
│ • Verify model  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Check Model     │
│ Availability    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Available?      │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Yes     │ │ No      │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Get     │ │ Raise   │
│ Metadata│ │ Model   │
└────┬────┘ │ Not     │
     │      │ Found   │
     ▼      └─────────┘
┌─────────────────┐
│ Extract Prompts │
│ and Row IDs     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Run Batch       │
│ Inference       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Merge Results   │
│ with Original   │
│ DataFrame       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Return          │
│ Processed       │
│ DataFrame       │
└─────────────────┘
```

### 8. Error Handling Flow

```
┌─────────────────┐
│ Error Occurs    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Classify Error  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Error Type      │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Model   │ │ Parse   │
│ Not     │ │ Error   │
│ Found   │ │         │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Log     │ │ Log     │
│ Error   │ │ Error   │
│         │ │         │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Raise   │ │ Raise   │
│ Model   │ │ Parse   │
│ Not     │ │ Error   │
│ Found   │ │         │
└─────────┘ └─────────┘
```

## Performance Considerations

### Batch Size Impact
```
Small Batch (5-10)     Medium Batch (10-20)    Large Batch (20-50)
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Low Memory    │    │ • Balanced      │    │ • High Memory   │
│ • High Overhead │    │ • Moderate      │    │ • Low Overhead  │
│ • Good for      │    │ • Good for      │    │ • Good for      │
│   Small Data    │    │   Medium Data   │    │   Large Data    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Timeout Strategy
```
Fast Models (< 30s)     Slow Models (> 60s)     Network Issues
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Timeout: 15s  │    │ • Timeout: 60s  │    │ • Timeout: 120s │
│ • Retries: 2    │    │ • Retries: 3    │    │ • Retries: 5    │
│ • Quick Fail    │    │ • Patient Wait  │    │ • Robust        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Memory Management
```
Processing Strategy
┌─────────────────┐
│ Load Batch      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Process Batch   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Store Results   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Clear Batch     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Next Batch      │
└─────────────────┘
``` 