# RAG System API Reference

## RAGSystem Class

The main entry point for the RAG Agent System, providing a unified interface for document processing, querying, and system management.

### Constructor

```python
RAGSystem(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path` (Optional[str]): Path to configuration file. If None, uses default configuration.

**Example:**
```python
# Use default configuration
rag = RAGSystem()

# Use custom configuration
rag = RAGSystem("config/production.yaml")
```

### Document Management

#### add_document

```python
add_document(file_path: str) -> Document
```

Add a single document to the system.

**Parameters:**
- `file_path` (str): Path to the document file

**Returns:**
- `Document`: Processed document object with metadata and status

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format not supported

**Example:**
```python
doc = rag.add_document("documents/research_paper.pdf")
print(f"Status: {doc.processing_status}")
print(f"Chunks: {len(doc.chunks)}")
```

#### add_documents

```python
add_documents(
    directory_path: str, 
    file_patterns: List[str] = None
) -> List[Document]
```

Add multiple documents from a directory.

**Parameters:**
- `directory_path` (str): Path to directory containing documents
- `file_patterns` (List[str], optional): List of file patterns (e.g., ['*.pdf', '*.docx'])

**Returns:**
- `List[Document]`: List of processed document objects

**Example:**
```python
documents = rag.add_documents(
    "documents/", 
    file_patterns=["*.pdf", "*.docx"]
)
print(f"Processed {len(documents)} documents")
```

#### validate_document

```python
validate_document(file_path: str) -> Dict[str, Any]
```

Validate if a document can be processed.

**Parameters:**
- `file_path` (str): Path to document file

**Returns:**
- `Dict[str, Any]`: Validation result with 'valid', 'file_size', 'file_type', and optional 'error'

**Example:**
```python
validation = rag.validate_document("large_file.pdf")
if validation['valid']:
    rag.add_document("large_file.pdf")
else:
    print(f"Validation failed: {validation['error']}")
```

### Query Processing

#### query

```python
query(
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    prompt_template: str = "default"
) -> GenerationResponse
```

Process a query and generate a response.

**Parameters:**
- `question` (str): The question to ask
- `top_k` (int): Number of top results to retrieve
- `score_threshold` (float): Minimum similarity score threshold
- `prompt_template` (str): Template to use for generation

**Returns:**
- `GenerationResponse`: Response object with answer, sources, and metadata

**Example:**
```python
response = rag.query(
    "What is machine learning?",
    top_k=10,
    score_threshold=0.7
)
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)}")
```

#### query_with_sources

```python
query_with_sources(question: str, top_k: int = 5) -> GenerationResponse
```

Query with detailed source information.

**Parameters:**
- `question` (str): The question to ask
- `top_k` (int): Number of sources to retrieve

**Returns:**
- `GenerationResponse`: Response with detailed source attribution

#### detailed_query

```python
detailed_query(question: str, top_k: int = 10) -> GenerationResponse
```

Query using detailed prompt template for comprehensive answers.

**Parameters:**
- `question` (str): The question to ask
- `top_k` (int): Number of sources to retrieve

**Returns:**
- `GenerationResponse`: Detailed response object

#### summarize_query

```python
summarize_query(question: str, top_k: int = 5) -> GenerationResponse
```

Query using summary template for concise answers.

**Parameters:**
- `question` (str): The question to ask
- `top_k` (int): Number of sources to retrieve

**Returns:**
- `GenerationResponse`: Summarized response object

#### factual_query

```python
factual_query(
    question: str, 
    top_k: int = 3, 
    score_threshold: float = 0.8
) -> GenerationResponse
```

Query for factual information with high confidence.

**Parameters:**
- `question` (str): The question to ask
- `top_k` (int): Number of sources to retrieve
- `score_threshold` (float): Minimum similarity score

**Returns:**
- `GenerationResponse`: Factual response object

### System Management

#### get_system_status

```python
get_system_status() -> Dict[str, Any]
```

Get current system status and metrics.

**Returns:**
- `Dict[str, Any]`: System status including agent status, metrics, and configuration

**Example:**
```python
status = rag.get_system_status()
print(f"Documents: {status['metrics']['total_documents']}")
print(f"Chunks: {status['metrics']['total_chunks']}")
print(f"Agent Status: {status['agents_status']}")
```

#### health_check

```python
health_check() -> Dict[str, Any]
```

Perform comprehensive health check on all components.

**Returns:**
- `Dict[str, Any]`: Health status with overall status and component details

**Example:**
```python
health = rag.health_check()
print(f"Overall Status: {health['overall_status']}")
for component, status in health['components'].items():
    print(f"  {component}: {status}")
```

#### get_supported_formats

```python
get_supported_formats() -> List[str]
```

Get list of supported document formats.

**Returns:**
- `List[str]`: List of supported file extensions

**Example:**
```python
formats = rag.get_supported_formats()
print(f"Supported formats: {formats}")
```

### Configuration Management

#### update_configuration

```python
update_configuration(**kwargs) -> None
```

Update system configuration at runtime.

**Parameters:**
- `**kwargs`: Configuration parameters to update

**Example:**
```python
rag.update_configuration(
    temperature=0.8,
    max_tokens=2000,
    chunk_size=1500
)
```

#### get_configuration

```python
get_configuration() -> Dict[str, Any]
```

Get current configuration (excluding sensitive data).

**Returns:**
- `Dict[str, Any]`: Current configuration dictionary

**Example:**
```python
config = rag.get_configuration()
print(f"Embedding model: {config['embeddings']['model_name']}")
print(f"LLM temperature: {config['llm']['temperature']}")
```

#### reload_configuration

```python
reload_configuration() -> None
```

Reload configuration from files and environment.

**Example:**
```python
rag.reload_configuration()
```

### Workflow Management (Advanced)

#### execute_workflow

```python
async execute_workflow(workflow_type: str, **kwargs) -> Dict[str, Any]
```

Execute an orchestrated workflow.

**Parameters:**
- `workflow_type` (str): Type of workflow to execute
- `**kwargs`: Workflow-specific parameters

**Returns:**
- `Dict[str, Any]`: Workflow execution results

**Example:**
```python
import asyncio

result = await rag.execute_workflow(
    workflow_type="standard",
    query="What is artificial intelligence?",
    enable_evaluation=True
)
print(f"Processing times: {result['processing_times']}")
print(f"Evaluation results: {result['evaluation_results']}")
```

#### get_available_workflows

```python
get_available_workflows() -> List[str]
```

Get list of available workflow types.

**Returns:**
- `List[str]`: List of workflow type names

#### get_workflow_schema

```python
get_workflow_schema(workflow_type: str) -> Dict[str, Any]
```

Get schema information for a workflow type.

**Parameters:**
- `workflow_type` (str): Type of workflow

**Returns:**
- `Dict[str, Any]`: Workflow schema dictionary

### Context Manager Support

The RAGSystem class supports context manager protocol:

```python
with RAGSystem() as rag:
    rag.add_document("document.pdf")
    response = rag.query("What is this about?")
    print(response.answer)
```

### String Representations

```python
# Basic string representation
str(rag)  # Returns: "RAGSystem(config=MyRAGSystem v1.0.0)"

# Detailed representation
repr(rag)  # Returns: "RAGSystem(documents=10, chunks=150, embedding_provider=sentence_transformers)"
```

## Response Objects

### GenerationResponse

Response object returned by query methods.

**Attributes:**
- `answer` (str): Generated answer text
- `sources` (List[Dict]): List of source documents with scores
- `query` (str): Original query text
- `model_used` (str): LLM model used for generation
- `processing_time` (float): Time taken to process query
- `metadata` (Dict): Additional metadata about the response

**Example:**
```python
response = rag.query("What is AI?")
print(f"Answer: {response.answer}")
print(f"Model: {response.model_used}")
print(f"Processing time: {response.processing_time:.2f}s")

for i, source in enumerate(response.sources):
    print(f"Source {i+1}: {source['chunk_id']} (score: {source['score']:.3f})")
```

### Document

Document object representing processed documents.

**Attributes:**
- `id` (str): Unique document identifier
- `filename` (str): Document filename
- `file_path` (str): Path to source file
- `content` (str): Extracted text content
- `document_type` (str): Document type (pdf, docx, etc.)
- `file_size` (int): File size in bytes
- `processing_status` (ProcessingStatus): Current processing status
- `metadata` (Dict): Document metadata
- `created_at` (datetime): Creation timestamp
- `updated_at` (datetime): Last update timestamp

**Example:**
```python
doc = rag.add_document("research.pdf")
print(f"ID: {doc.id}")
print(f"Type: {doc.document_type}")
print(f"Status: {doc.processing_status}")
print(f"Size: {doc.file_size} bytes")
print(f"Metadata: {doc.metadata}")
```

## Convenience Functions

### create_rag_system

```python
create_rag_system(config_path: Optional[str] = None) -> RAGSystem
```

Convenience function to create a RAG system instance.

**Parameters:**
- `config_path` (Optional[str]): Path to configuration file

**Returns:**
- `RAGSystem`: Initialized RAG system instance

### quick_setup

```python
quick_setup(documents_path: str, config_path: Optional[str] = None) -> RAGSystem
```

Quick setup function that creates a RAG system and processes documents.

**Parameters:**
- `documents_path` (str): Path to documents directory
- `config_path` (Optional[str]): Path to configuration file

**Returns:**
- `RAGSystem`: RAG system with processed documents

**Example:**
```python
from src.core.rag_system import quick_setup

# Quick setup with documents
rag = quick_setup("documents/", "config/production.yaml")

# Start querying immediately
response = rag.query("What topics are covered?")
print(response.answer)
```

## Error Handling

The RAG system uses custom exceptions for error handling:

```python
from src.core.rag_system import RAGSystem

try:
    rag = RAGSystem()
    doc = rag.add_document("nonexistent.pdf")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid file format: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Batch Processing
For processing multiple documents, use `add_documents()` instead of multiple `add_document()` calls:

```python
# Efficient batch processing
documents = rag.add_documents("documents/")

# Less efficient individual processing
# for file in files:
#     rag.add_document(file)  # Don't do this for many files
```

### Query Optimization
- Use appropriate `top_k` values (5-10 for most cases)
- Set `score_threshold` to filter low-relevance results
- Choose specific query types for better performance

### Memory Management
- The system automatically manages memory for embeddings
- For large document collections, consider chunking strategies
- Monitor system status for resource usage

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI
from src.core.rag_system import RAGSystem

app = FastAPI()
rag = RAGSystem()

@app.post("/query")
async def query_endpoint(question: str):
    response = rag.query(question)
    return {
        "answer": response.answer,
        "sources": response.sources,
        "processing_time": response.processing_time
    }
```

### With Flask

```python
from flask import Flask, request, jsonify
from src.core.rag_system import RAGSystem

app = Flask(__name__)
rag = RAGSystem()

@app.route("/query", methods=["POST"])
def query_endpoint():
    question = request.json.get("question")
    response = rag.query(question)
    return jsonify({
        "answer": response.answer,
        "sources": response.sources
    })
```

## Best Practices

1. **Initialization**: Initialize the RAG system once and reuse it
2. **Configuration**: Use environment-specific configuration files
3. **Error Handling**: Always handle potential exceptions
4. **Monitoring**: Regularly check system health and metrics
5. **Performance**: Monitor query response times and optimize as needed
6. **Security**: Keep API keys and sensitive configuration secure