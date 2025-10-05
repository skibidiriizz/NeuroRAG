# 🚀 RAG Agent System - Setup Instructions

This guide will help you set up and run the RAG Agent System on your machine.

## 📋 Prerequisites

- **Python 3.8 or higher**
- **Git** (for cloning the repository)
- **Docker** (optional, for Qdrant vector database)
- **At least 4GB RAM** (for embedding models)

## 🛠️ Installation

### Step 1: Clone and Navigate

```bash
git clone <repository-url>
cd rag-agent-system
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will install all necessary packages including:
- LangChain framework
- Vector databases (Qdrant, Chroma)
- Document processors (pypdf, python-docx, etc.)
- Embedding models (sentence-transformers)
- Web frameworks (Streamlit, FastAPI)

## 🔧 Configuration

### Step 1: Environment Variables

Copy the environment template and configure your settings:

```bash
cp .env.template .env
```

Edit `.env` with your API keys:

```env
# OpenAI Configuration (if using OpenAI embeddings/LLM)
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration (if using Qdrant Cloud)
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Other configurations...
```

### Step 2: Configuration File

The system uses `config/config.yaml` for main configuration. Default settings should work for most use cases, but you can customize:

- **Chunk size and overlap**
- **Embedding models**
- **Vector database settings**
- **LLM parameters**

## 🗄️ Vector Database Setup

### Option 1: Qdrant (Recommended)

**Using Docker:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Using Docker Compose:**
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

### Option 2: Chroma (Local)

Chroma will automatically create a local database in `./data/chroma/` - no additional setup required.

### Option 3: Qdrant Cloud

1. Sign up at [qdrant.tech](https://qdrant.tech/)
2. Create a cluster
3. Get your URL and API key
4. Update `.env` file with your credentials

## 🧪 Testing the Installation

### Step 1: Basic Test

```bash
python example_usage.py
```

This will:
- Initialize the RAG system
- Create sample documents
- Process and embed documents
- Store in vector database
- Show system status

### Step 2: Health Check

```python
from src.core.rag_system import RAGSystem

rag = RAGSystem()
health = rag.health_check()
print(health)
```

### Step 3: Add Your Documents

```python
from src.core.rag_system import RAGSystem

# Initialize system
rag = RAGSystem()

# Add single document
doc = rag.add_document("path/to/your/document.pdf")

# Add all documents from directory
docs = rag.add_documents("path/to/your/documents/")

# Check status
status = rag.get_system_status()
print(f"Processed: {status['metrics']['total_documents']} documents")
print(f"Created: {status['metrics']['total_chunks']} chunks")
```

## 📊 Running the Dashboard

```bash
streamlit run dashboards/main_dashboard.py
```

Then open http://localhost:8501 in your browser.

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the project directory and virtual environment is activated
cd rag-agent-system
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

**2. Vector Database Connection**
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Or check the health endpoint
curl http://localhost:6333/health
```

**3. Memory Issues**
If you encounter memory issues with embedding models:
- Use smaller embedding models (e.g., "all-MiniLM-L6-v2")
- Reduce batch size in configuration
- Process documents one at a time

**4. CUDA/GPU Issues**
If you have GPU but models aren't using it:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configuration Validation

```python
from src.core.config_manager import ConfigManager

config = ConfigManager()
errors = config.validate_config()
if errors:
    print("Configuration errors:", errors)
else:
    print("Configuration is valid!")
```

## 🔧 Development Setup

### Additional Development Dependencies

```bash
pip install pytest black flake8 mypy pre-commit
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Pre-commit Hooks

```bash
pre-commit install
```

## 🌟 Quick Start Examples

### Example 1: Process PDFs

```python
from src import RAGSystem

rag = RAGSystem()
documents = rag.add_documents("./research_papers/", ["*.pdf"])
print(f"Processed {len(documents)} PDFs")
```

### Example 2: Query System (when retrieval/generation implemented)

```python
response = rag.query("What are the main findings about machine learning?")
print(response)
```

### Example 3: Monitor System

```python
status = rag.get_system_status()
health = rag.health_check()

print("System Status:", status['overall_status'])
print("Documents:", status['metrics']['total_documents'])
print("Health:", health['overall_status'])
```

## 📚 Next Steps

1. **Add Documents**: Place your documents in `data/raw/`
2. **Configure Settings**: Adjust `config/config.yaml` for your needs
3. **Set API Keys**: Configure `.env` with your service credentials
4. **Run Dashboard**: Use Streamlit interface for easy management
5. **Implement Retrieval**: Complete the remaining agents (in progress)

## 🆘 Getting Help

- **Check Health**: Use `rag.health_check()` for system diagnostics
- **View Logs**: Check `logs/rag_system.log` for detailed information
- **Configuration**: Use `rag.get_configuration()` to see current settings
- **Status**: Use `rag.get_system_status()` for metrics and info

## 🎯 Performance Tips

1. **Embedding Models**: Start with smaller models for faster processing
2. **Chunk Size**: Experiment with different chunk sizes (500-2000 characters)
3. **Batch Processing**: Process multiple documents together for efficiency
4. **Vector Database**: Use Qdrant for better performance with large datasets
5. **Hardware**: GPU acceleration significantly speeds up embedding generation

---

**Need more help?** Check the documentation in the `docs/` directory or open an issue on GitHub.