# Debugging Logger Flow Diagram

## Overview

The debugging logger provides a structured approach to logging, error handling, and debugging throughout the EpiTuner suite.

## Main Flow

```
┌─────────────────┐
│   Application   │
│     Start       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Initialize     │
│   Logger        │
│                 │
│ • Set log file  │
│ • Set debug mode│
│ • Set session ID│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Log Message   │
│                 │
│ • Determine     │
│   log level     │
│ • Add timestamp │
│ • Add context   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Is Error Log?  │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│   Yes   │ │    No   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│Categorize│ │ Write   │
│  Error   │ │  Log    │
└────┬────┘ └────┬────┘
     │           │
     ▼           │
┌─────────┐      │
│Generate │      │
│  Fix    │      │
│Suggestion│     │
└────┬────┘      │
     │           │
     ▼           ▼
┌─────────────────┐
│  Write Log to   │
│   File & Console│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Interactive     │
│  Debug Mode?    │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│   Yes   │ │    No   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Prompt  │ │ Continue│
│  User   │ │  Flow   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────────────┐
│   Continue      │
│   Application   │
└─────────────────┘
```

## Error Categorization Flow

```
┌─────────────────┐
│   Error Message │
│   & Module      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Check Message  │
│   Keywords      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Data Keywords? │
│ • missing column│
│ • invalid data  │
│ • empty dataset │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│   Yes   │ │    No   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│  DATA   │ │ Schema  │
│ ISSUE   │ │Keywords?│
└─────────┘ └────┬────┘
                 │
           ┌─────┴─────┐
           │           │
           ▼           ▼
       ┌─────────┐ ┌─────────┐
       │   Yes   │ │    No   │
       └────┬────┘ └────┬────┘
            │           │
            ▼           ▼
        ┌─────────┐ ┌─────────┐
        │ SCHEMA  │ │ Model   │
        │ ISSUE   │ │Keywords?│
        └─────────┘ └────┬────┘
                         │
                   ┌─────┴─────┐
                   │           │
                   ▼           ▼
               ┌─────────┐ ┌─────────┐
               │   Yes   │ │    No   │
               └────┬────┘ └────┬────┘
                    │           │
                    ▼           ▼
                ┌─────────┐ ┌─────────┐
                │ MODEL   │ │ Config  │
                │ ISSUE   │ │Keywords?│
                └─────────┘ └────┬────┘
                                 │
                           ┌─────┴─────┐
                           │           │
                           ▼           ▼
                       ┌─────────┐ ┌─────────┐
                       │   Yes   │ │    No   │
                       └────┬────┘ └────┬────┘
                            │           │
                            ▼           ▼
                        ┌─────────┐ ┌─────────┐
                        │CONFIG   │ │Network  │
                        │ERROR    │ │Keywords?│
                        └─────────┘ └────┬────┘
                                         │
                                   ┌─────┴─────┐
                                   │           │
                                   ▼           ▼
                               ┌─────────┐ ┌─────────┐
                               │   Yes   │ │    No   │
                               └────┬────┘ └────┬────┘
                                    │           │
                                    ▼           ▼
                                ┌─────────┐ ┌─────────┐
                                │NETWORK  │ │RUNTIME  │
                                │ERROR    │ │ERROR    │
                                └─────────┘ └─────────┘
```

## Log Searching Flow

```
┌─────────────────┐
│   Search Query  │
│                 │
│ • Module        │
│ • Level         │
│ • Category      │
│ • Time Range    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Load Log File  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Parse Each Log │
│     Entry       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Apply Filters  │
│                 │
│ • Module match? │
│ • Level match?  │
│ • Category match│
│ • Time in range │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Add to Results │
│    if Match     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  More Entries?  │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│   Yes   │ │    No   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Continue│ │ Return  │
│ Parsing │ │ Results │
└─────────┘ └─────────┘
```

## Interactive Debugging Flow

```
┌─────────────────┐
│   Error Occurs  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Interactive     │
│ Debug Enabled?  │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│   Yes   │ │    No   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Display │ │ Continue│
│  Error  │ │  Flow   │
│ Details │ │         │
└────┬────┘ └─────────┘
     │
     ▼
┌─────────────────┐
│  Prompt User    │
│                 │
│ [r]etry         │
│ [s]kip          │
│ [a]bort         │
│ [c]ontinue      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Process User   │
│    Choice       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Execute Action │
│                 │
│ • retry: call   │
│   function again│
│ • skip: return  │
│   None/null     │
│ • abort: exit   │
│   application   │
│ • continue:     │
│   raise error   │
└─────────────────┘
```

## Function Decorator Flow

```
┌─────────────────┐
│  Function Call  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Log Function   │
│    Entry        │
│                 │
│ • Function name │
│ • Parameters    │
│ • Timestamp     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Execute        │
│  Function       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Exception      │
│  Occurs?        │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│   Yes   │ │    No   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Log     │ │ Log     │
│ Error   │ │ Function│
│         │ │  Exit   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Re-raise│ │ Return  │
│ Error   │ │ Result  │
└─────────┘ └─────────┘
```

## Data Processing Logging Flow

```
┌─────────────────┐
│  Data Processing│
│    Operation    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Extract Data   │
│   Information   │
│                 │
│ • Row count     │
│ • Column count  │
│ • File size     │
│ • Processing    │
│   parameters    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Log Data       │
│  Processing     │
│  Information    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Continue with  │
│  Processing     │
└─────────────────┘
```

## Error Summary Generation Flow

```
┌─────────────────┐
│  Generate Error │
│    Summary      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Load All Logs  │
│  for Session    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Filter Error   │
│    Logs Only    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Analyze Errors │
│                 │
│ • Count total   │
│ • Group by      │
│   category      │
│ • Group by      │
│   module        │
│ • Extract       │
│   suggestions   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Generate       │
│  Summary        │
│  Statistics     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Return Summary │
│  Dictionary     │
└─────────────────┘
``` 