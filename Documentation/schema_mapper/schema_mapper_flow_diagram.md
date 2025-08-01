# Schema Mapper - Flow Diagrams

## Overview Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Data    │───▶│  Extract Unique  │───▶│  Suggest        │
│                 │    │  Ratings         │    │  Mapping        │
│ • DataFrame     │    │                  │    │                 │
│ • Expert Rating │    │ • Find unique    │    │ • Auto-suggest  │
│   column        │    │   values         │    │ • Pattern match │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Validate       │    │   Apply         │
                       │   Mapping        │    │   Mapping       │
                       │                  │    │                 │
                       │ • Check coverage │    │ • Transform     │
                       │ • Report missing │    │ • Add column    │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Error Exit     │    │   Results       │
                       │                  │    │                 │
                       │ • Missing        │    │ • Mapped        │
                       │   mappings       │    │   DataFrame     │
                       │ • Validation     │    │ • Metadata      │
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
│   Ready State   │
└─────────────────┘
```

### 2. Extract Unique Ratings Flow

```
┌─────────────────┐
│ Extract Unique  │
│ Ratings         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Validate        │
│ DataFrame       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Check Expert    │
│ Rating Column   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Column Exists?  │
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
│ Extract │ │ Raise   │
│ Unique  │ │ Mapping │
│ Values  │ │ Error   │
└────┬────┘ └─────────┘
     │
     ▼
┌─────────┐
│ Return  │
│ List    │
└─────────┘
```

### 3. Suggest Mapping Flow

```
┌─────────────────┐
│ Suggest Mapping │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ For Each Rating │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Check Type      │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Numeric │ │ Text    │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Direct  │ │ Pattern │
│ Mapping │ │ Match   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Known   │ │ Known   │
│ Value?  │ │ Pattern?│
└────┬────┘ └────┬────┘
     │           │
    ┌┴─┐        ┌┴─┐
    │Y │        │Y │
    └─┬┘        └─┬┘
      │           │
      ▼           ▼
┌─────────┐ ┌─────────┐
│ Map to  │ │ Map to  │
│ Value   │ │ Pattern │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────┐
│ Default to -1   │
│ (Unknown)       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Return Mapping  │
│ Dictionary      │
└─────────────────┘
```

### 4. Validate Mapping Flow

```
┌─────────────────┐
│ Validate Mapping│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Extract Unique  │
│ Ratings         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ For Each Rating │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ In Mapping?     │
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
│ Continue│ │ Add to  │
│         │ │ Missing │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────┐
│ All Mapped?     │
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
│ Return  │ │ Return  │
│ True    │ │ False   │
└─────────┘ └─────────┘
```

### 5. Apply Mapping Flow

```
┌─────────────────┐
│ Apply Mapping   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Validate        │
│ Mapping         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Valid?          │
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
│ Create  │ │ Raise   │
│ New     │ │ Error   │
│ Column  │ │         │
└────┬────┘ └─────────┘
     │
     ▼
┌─────────────────┐
│ For Each Row    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Get Original    │
│ Rating          │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Look Up in      │
│ Mapping         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Set Standardized│
│ Rating          │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ More Rows?      │
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
│ Return Mapped   │
│ DataFrame       │
└─────────────────┘
```

### 6. Process Mapping Flow

```
┌─────────────────┐
│ Process Mapping │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Validate Input  │
│                 │
│ • Check DataFrame│
│ • Check mapping │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Validate        │
│ Mapping         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Valid?          │
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
│ Apply   │ │ Log     │
│ Mapping │ │ Error   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Create  │ │ Return  │
│ Metadata│ │ Error   │
└────┬────┘ └─────────┘
     │
     ▼
┌─────────────────┐
│ Save Metadata   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Return Results  │
└─────────────────┘
```

### 7. Metadata Creation Flow

```
┌─────────────────┐
│ Create Metadata │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Extract Info    │
│                 │
│ • Original      │
│   values        │
│ • Mapped values │
│ • Schema        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Create Summary  │
│                 │
│ • Total mappings│
│ • Coverage      │
│ • Distribution  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Build Metadata  │
│ Structure       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Return Metadata │
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
│ Mapping │ │ Data    │
│ Error   │ │ Type    │
│         │ │ Error   │
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
│ Mapping │ │ Data    │
│ Error   │ │ Type    │
│         │ │ Error   │
└─────────┘ └─────────┘
```

## Performance Considerations

### Data Size Impact
```
Small Dataset (< 1K rows)     Medium Dataset (1K-10K rows)    Large Dataset (> 10K rows)
┌─────────────────┐          ┌─────────────────┐             ┌─────────────────┐
│ • Fast          │          │ • Moderate      │             │ • Slower        │
│ • Low Memory    │          │ • Balanced      │             │ • Higher Memory │
│ • Quick         │          │ • Efficient     │             │ • Optimized     │
│   Validation    │          │ • Good          │             │ • Batch         │
└─────────────────┘          └─────────────────┘             └─────────────────┘
```

### Mapping Complexity
```
Simple Mapping (Numeric)      Mixed Mapping (Text+Numeric)    Complex Mapping (Custom)
┌─────────────────┐          ┌─────────────────┐             ┌─────────────────┐
│ • Direct        │          │ • Pattern       │             │ • Custom        │
│ • Fast          │          │ • Matching      │             │ • Logic         │
│ • Efficient     │          │ • Moderate      │             │ • Slower        │
└─────────────────┘          └─────────────────┘             └─────────────────┘
```

### Memory Management
```
Processing Strategy
┌─────────────────┐
│ Load DataFrame  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Extract Unique  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Create Mapping  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Apply Mapping   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Create Metadata │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Save Results    │
└─────────────────┘
``` 