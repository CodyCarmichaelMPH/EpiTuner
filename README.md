# EpiTuner - Simple Medical LoRA Training

Simple, reliable LoRA fine-tuning for medical syndromic surveillance data.

## ✨ Features

- **Always Works™** - Simplified architecture with minimal dependencies
- **Local-only processing** - Your medical data never leaves your machine
- **Consumer GPU friendly** - Optimized for RTX 3060/4060 class hardware
- **Ollama integration** - Use your locally downloaded models
- **Expert review workflow** - Human-in-the-loop validation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone repository
git clone [your-repo-url]
cd EpiTuner

# Install requirements
pip install -r requirements.txt

# Install Ollama models
ollama pull llama3.2:1b
ollama pull phi3
```

### 2. Run the GUI

```bash
streamlit run app.py
```

### 3. Follow the 5-Step Process

1. **📁 Upload** - CSV with medical data
2. **🤖 Model** - Select Ollama model + classification task  
3. **🎯 Train** - LoRA training (10-60 minutes)
4. **📊 Review** - Check model predictions
5. **💾 Download** - Get your trained model

## 📋 Data Format

Your CSV must have these columns:

- `C_Biosense_ID` - Unique identifier
- `ChiefComplaintOrig` - Chief complaint text
- `DischargeDiagnosis` - Discharge diagnosis
- `Expert Rating` - Match/Not a Match/Unknown/Not able to determine  
- `Rationale_of_Rating` - Expert's reasoning

## 🖥️ System Requirements

**Minimum:**
- Windows 10/11
- 8GB RAM
- Python 3.8+
- Ollama installed

**Recommended:**
- NVIDIA GPU (RTX 3060 or better)
- 16GB RAM
- 50GB free disk space

## 🛠️ CLI Usage

For advanced users or batch processing:

```bash
# Train a model
python scripts/train.py \
  --data data.csv \
  --model microsoft/Phi-3-mini-4k-instruct \
  --topic "Motor vehicle collisions" \
  --output outputs/my_model \
  --epochs 2

# Make predictions  
python scripts/predict.py \
  --model outputs/my_model \
  --data new_data.csv \
  --topic "Motor vehicle collisions" \
  --output predictions.json
```

## 🐛 Troubleshooting

### "No Ollama models found"
```bash
ollama pull llama3.2:1b
ollama pull phi3
```

### "CUDA out of memory"
- Use a smaller model (llama3.2:1b)
- Reduce batch size in config
- Close other GPU applications

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Training fails
- Check model name is correct
- Ensure data has required columns
- Try CPU-only mode by setting `device="cpu"`

## 📖 Architecture

```
EpiTuner/
├── epituner/
│   ├── core/           # Core training & inference 
│   └── utils/          # Hardware detection
├── scripts/            # CLI tools
├── configs/            # Single adaptive config
├── sample_data/        # Example data
└── app.py             # Streamlit GUI (~400 lines)
```

**Key Simplifications:**
- 5 required dependencies (vs 15+ before)
- 1 config file (vs 5 before)
- Linear workflow (vs complex state management)
- Template-free text processing
- Graceful fallbacks for optional components

## 🤝 Contributing

This is designed for **simplicity and reliability**. Before adding features:

1. Does it align with "Always Works™" philosophy?
2. Is it essential for core functionality?  
3. Does it add complexity that could break?

## 📄 License

[Your License Here]

---

**Always Works™ Philosophy**: Simple things should be simple. Complex things should be possible. Broken things should be impossible.