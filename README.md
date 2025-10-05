# 🤖 RAG Agent System

A comprehensive **Retrieval-Augmented Generation (RAG) system with intelligent agents** built using Python, LangChain/LangGraph, and vector databases. This production-ready system provides modular components for document processing, embedding generation, semantic retrieval, and grounded response generation.

## 🌟 Features

- **Multi-Agent Architecture**: Orchestrated with LangGraph for complex workflows
- **Document Processing**: Support for PDF, DOCX, TXT, and HTML files
- **Advanced Chunking**: Multiple text splitting strategies with overlap optimization
- **Vector Storage**: Integration with Qdrant and Chroma vector databases
- **Semantic Retrieval**: High-performance similarity search with reranking
- **Grounded Generation**: Context-aware response generation with source attribution
- **Evaluation Framework**: Comprehensive metrics for faithfulness, relevance, and fluency
- **Interactive Dashboard**: Real-time monitoring and visualization with Streamlit
- **Production Ready**: Logging, monitoring, and configuration management

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Document Parser │────│   Chunker &     │────│  Vector Store   │
│     Agent       │    │ Embedder Agent  │    │     (Qdrant)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Visualization  │────│   Evaluation    │────│   Retrieval     │
│     Agent       │    │     Agent       │    │     Agent       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │    LangGraph    │────│   Generation    │
                       │  Orchestrator   │    │     Agent       │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rag-agent-system
```

2. **Create virtual environment**:
```bash
python -m venv venv
# On Windows
venv\\Scripts\\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.template .env
# Edit .env with your API keys
```

5. **Start vector database** (Qdrant):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Basic Usage

```python
from src.core.rag_system import RAGSystem

# Initialize the system
rag = RAGSystem()

# Process documents
rag.add_documents("data/raw/")

# Ask questions
response = rag.query("What is the main topic discussed in the documents?")
print(response)
```

### Run the Dashboard

```bash
streamlit run dashboards/main_dashboard.py
```

## 📁 Project Structure

```
rag-agent-system/
├── src/
│   ├── agents/              # Individual agent implementations
│   │   ├── document_parser.py
│   │   ├── chunking_embedding.py
│   │   ├── retrieval.py
│   │   ├── generation.py
│   │   ├── evaluation.py
│   │   └── visualization.py
│   ├── core/                # Core system components
│   │   ├── rag_system.py
│   │   ├── config_manager.py
│   │   └── orchestrator.py
│   ├── utils/               # Utility functions
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── helpers.py
│   └── models/              # Data models
│       ├── document.py
│       ├── chunk.py
│       └── response.py
├── config/                  # Configuration files
│   └── config.yaml
├── data/                    # Data storage
│   ├── raw/                 # Raw documents
│   ├── processed/           # Processed documents
│   └── embeddings/          # Vector embeddings
├── tests/                   # Test files
├── dashboards/              # Streamlit dashboards
├── notebooks/               # Jupyter notebooks
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── .env.template           # Environment variables template
└── README.md               # This file
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- **LLM Settings**: Model selection, temperature, tokens
- **Embeddings**: Provider and model configuration
- **Vector DB**: Database connection and parameters
- **Chunking**: Strategy and size parameters
- **Retrieval**: Top-k, scoring thresholds
- **Evaluation**: Metrics and batch settings

## 🤖 Agents Overview

### 1. Document Parser Agent
- **Purpose**: Extract and clean text from various file formats
- **Formats**: PDF, DOCX, TXT, HTML
- **Features**: Metadata extraction, text cleaning, format detection

### 2. Chunking & Embedding Agent
- **Purpose**: Split text and generate vector embeddings
- **Strategies**: Recursive, fixed-size, semantic chunking
- **Models**: Sentence Transformers, OpenAI embeddings

### 3. Retrieval Agent
- **Purpose**: Semantic search over vector database
- **Features**: Top-k retrieval, score filtering, reranking
- **Databases**: Qdrant, Chroma support

### 4. Generation Agent
- **Purpose**: Generate grounded responses using retrieved context
- **Features**: Context injection, source attribution, prompt templates
- **Models**: OpenAI GPT, local LLaMA support

### 5. Evaluation Agent
- **Purpose**: Assess response quality and system performance
- **Metrics**: Faithfulness, relevance, fluency, BLEU, ROUGE, BERTScore
- **Output**: Detailed evaluation reports and metrics

### 6. Visualization Agent
- **Purpose**: Create interactive dashboards and reports
- **Features**: Real-time metrics, retrieval analytics, performance monitoring
- **Framework**: Streamlit with Plotly visualizations

## 📊 Evaluation Metrics

- **Faithfulness**: How well the answer is supported by retrieved context
- **Relevance**: How well the answer addresses the original question
- **Fluency**: Linguistic quality and coherence of the response
- **Retrieval Precision**: Accuracy of document retrieval
- **Response Time**: End-to-end latency analysis

## 🔍 Advanced Features

### Multi-Modal Support
- Document metadata extraction
- Image and table processing (roadmap)
- Structured data handling

### Performance Optimization
- Batch processing capabilities
- Caching mechanisms
- Async processing support

### Monitoring & Logging
- Comprehensive logging with Loguru
- Experiment tracking with Weights & Biases
- LangChain tracing integration

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific agent tests:
```bash
pytest tests/test_document_parser.py -v
```

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- [Agent Architecture](docs/architecture.md)
- [Configuration Guide](docs/configuration.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the excellent framework
- [Qdrant](https://qdrant.tech/) for high-performance vector search
- [Streamlit](https://streamlit.io/) for the dashboard framework

## 📬 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ using Python, LangChain, and modern AI technologies.**