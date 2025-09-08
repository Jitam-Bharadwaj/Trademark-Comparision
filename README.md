# 🏷️ Trademark Analysis System

A comprehensive AI-powered system that combines both **text-based data extraction** and **visual logo comparison** capabilities for trademark analysis.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key
export GROQ_API_KEY="your_groq_api_key_here"

# Run the main system
python main.py
```

## 📁 System Structure

```
Trademark/
├── text_extraction/           # Text-based analysis system
│   ├── controllers/           # FastAPI controllers
│   ├── DAL_files/            # Data access layer
│   └── ui/                   # Streamlit interface
├── visual_comparison/        # Visual comparison system
│   ├── DAL_files/            # AI models and processing
│   └── ui/                   # Streamlit interface
├── main.py                   # Main command-line interface
├── main_app.py               # Main Streamlit interface
└── requirements.txt          # All dependencies
```

## 🎯 Two Analysis Systems

### 📄 Text-Based Extraction
- **Purpose**: Extract structured data from trademark documents
- **AI Models**: LLaMA 3.3 70B, LLaMA 4 Scout (Groq)
- **Input**: PDF, JPG, PNG, BMP, TIFF files
- **Output**: Structured JSON data with 16+ fields

### 🖼️ Visual Logo Comparison
- **Purpose**: Find visually similar trademarks and logos
- **AI Models**: Vision Transformer, EfficientNet, SentenceTransformer
- **Input**: PDF, PNG, JPG, JPEG files
- **Output**: Similarity scores and ranked results

## 🚀 Usage Options

### Main Interface (Default)
```bash
python main.py
# or
python main.py --main-ui
```

### Individual Systems
```bash
python main.py --text-api      # Text extraction API server
python main.py --text-ui       # Text extraction UI only
python main.py --visual-ui     # Visual comparison UI only
python main.py --all           # Show all options
```

### Direct Streamlit
```bash
streamlit run main_app.py      # Main interface
streamlit run text_extraction/ui/streamlit_app.py  # Text only
streamlit run visual_comparison/ui/app.py          # Visual only
```

## 📊 Features

### Text Extraction
- Extract company information, trademark details, contact info
- Fuzzy string matching for similarity analysis
- CSV database integration and comparison
- FastAPI backend with automatic documentation

### Visual Comparison
- Computer vision logo matching
- OCR text extraction from logos
- High-resolution PDF processing
- Interactive similarity search with FAISS

## 🛠️ Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster visual processing)
- At least 4GB RAM (8GB+ recommended)
- Groq API key for text extraction

## 📋 Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

3. **Run the system**:
   ```bash
   python main.py
   ```

## 🎯 Best Practices

- Use high-quality, clear documents for best results
- For visual comparison, ensure logos are prominently visible
- Upload CSV databases for text-based comparison
- Use different similarity thresholds for different use cases

## 📞 Support

For issues related to:
- **Text Extraction**: Check the text extraction documentation
- **Visual Comparison**: Check the visual comparison documentation
- **Groq API**: Visit [Groq Documentation](https://console.groq.com/docs)

---

**Note**: This system is designed for trademark comparison and analysis. Always consult with legal professionals for trademark-related decisions.