# Streamlit GUI Quick Reference

## Quick Start

```bash
# Launch the GUI
python run_gui.py

# Or directly with Streamlit
streamlit run gui/epituner_gui.py
```

## Workflow Steps

### Step 1: Data Upload
- Upload CSV file with required columns
- System validates data automatically
- View data preview and statistics

### Step 2: Schema Mapping
- Map expert ratings to standardized values
- Choose mapping method (Manual/Predefined/Auto-detect)
- Specify analysis topic

### Step 3: Prompt Formatting
- Select prompt type (Auto-detect/Training/Inference/Mixed)
- Configure formatting options
- Review formatted prompts

### Step 4: Model Operations
- Choose operation (Inference/Fine-tuning/Contextualizer/All)
- Configure model settings
- Execute and monitor progress

### Step 5: Results & Export
- View results and performance metrics
- Export data in various formats
- Generate session reports

## Required Data Format

```csv
C_BioSense_ID,ChiefComplaintOrig,Discharge Diagnosis,Sex,Age,Admit_Reason_Combo,Chief_Complaint_Combo,Diagnosis_Combo,CCDD,CCDDCategory,TriageNotes,Expert Rating,Rationale of Rating
P001,Fever,Viral pneumonia,M,25,Fever,Fever,Infection,Fever,Viral,High fever,1,Clear respiratory infection
P002,Chest pain,Angina,F,45,Chest pain,Chest pain,Cardiac,Chest pain,Cardiac,Severe chest pain,2,Cardiac symptoms present
```

## Settings Configuration

### Debug Settings
- **Debug Mode**: Enable verbose logging
- **Interactive Debug**: Enable interactive error resolution

### Model Settings
- **Model Name**: Default Ollama model (e.g., llama3.2:3b)
- **Server URL**: Ollama server address (default: http://localhost:11434)
- **Max Tokens**: Response length (1-4096, default: 512)
- **Temperature**: Response randomness (0.0-2.0, default: 0.7)

### Batch Settings
- **Batch Size**: Processing batch size (1-1000, default: 100)

## Common Operations

### Data Upload
1. Click "Browse files" in Step 1
2. Select your CSV file
3. Wait for validation
4. Review data preview

### Rating Mapping
1. In Step 2, choose mapping method
2. Map each unique rating to standardized value
3. Enter analysis topic
4. Click "Apply Rating Mapping"

### Prompt Creation
1. In Step 3, select prompt type
2. Configure formatting options
3. Click "Format Prompts"
4. Review prompt preview

### Model Execution
1. In Step 4, select operation
2. Configure model settings
3. Click "Start Operation"
4. Monitor progress

### Results Export
1. In Step 5, review results
2. Click export buttons for desired format
3. Download files to your system

## Error Handling

### Common Errors
- **File Upload**: Check CSV format and required columns
- **Model Connection**: Verify Ollama server is running
- **Processing**: Check data quality and system resources

### Recovery Actions
- Use "Go Back" buttons to return to previous steps
- Check error messages for specific guidance
- Review logs in the interface for details
- Restart the application if needed

## Navigation

### Sidebar Controls
- **Quick Navigation**: Jump to any step
- **Settings**: Configure debug mode, model settings
- **System Info**: View session details
- **Clear Session**: Reset all data

### Progress Indicator
- Shows current step in header
- Visual progress through workflow
- Step completion status

## Export Options

### Data Export
- **Processed Data**: CSV with standardized ratings
- **Formatted Prompts**: JSON with all prompts
- **Inference Results**: CSV with predictions
- **Session Report**: JSON with complete session summary

### Report Contents
- Session information and timestamp
- Data processing statistics
- Model operation results
- Error logs and performance metrics

## Keyboard Shortcuts

### Navigation
- Use Tab to navigate between fields
- Enter to submit forms
- Escape to cancel operations

### Browser Controls
- Ctrl+R: Refresh page
- Ctrl+F: Find in page
- Ctrl+S: Save page (if supported)

## Performance Tips

### Large Datasets
- Use smaller batch sizes for memory-constrained systems
- Monitor system resources during processing
- Consider splitting very large datasets

### Model Operations
- Fine-tuning requires significant time and resources
- Use appropriate model sizes for your system
- Monitor GPU/CPU usage during processing

### Optimization
- Close unnecessary browser tabs
- Use appropriate batch sizes
- Ensure adequate disk space

## Troubleshooting

### GUI Issues
```bash
# Check Streamlit installation
pip install streamlit

# Verify dependencies
pip install -r requirements.txt

# Check port availability
streamlit run gui/epituner_gui.py --server.port 8502
```

### Data Issues
- Verify CSV format and encoding
- Check required columns are present
- Ensure data quality and consistency

### Model Issues
```bash
# Check Ollama installation
ollama list

# Verify server is running
curl http://localhost:11434/api/tags
```

### System Issues
- Monitor memory and disk usage
- Check system resources
- Restart application if needed 