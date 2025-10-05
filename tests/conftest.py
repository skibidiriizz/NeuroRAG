"""
Global test configuration and fixtures for RAG Agent System tests.

This module provides shared fixtures, test utilities, and configuration
for all test modules in the project.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
from unittest.mock import Mock, MagicMock
import json

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_system import RAGSystem
from src.core.config_manager import ConfigManager
from src.models.document import Document, TextChunk, ProcessingStatus, GenerationResponse


# Test Configuration
TEST_CONFIG = {
    'app': {
        'name': 'TestRAGSystem',
        'version': '1.0.0',
        'environment': 'test',
        'debug': True
    },
    'embeddings': {
        'provider': 'sentence_transformers',
        'model_name': 'all-MiniLM-L6-v2',  # Small model for testing
        'api_key': 'test_key'
    },
    'llm': {
        'provider': 'openai',
        'model_name': 'gpt-3.5-turbo',
        'api_key': 'test_key',
        'temperature': 0.7,
        'max_tokens': 1000
    },
    'vector_db': {
        'provider': 'chroma',
        'host': 'localhost',
        'port': 8000,
        'collection_name': 'test_collection'
    },
    'chunking': {
        'strategy': 'recursive',
        'chunk_size': 500,
        'chunk_overlap': 50
    },
    'document_processing': {
        'max_file_size_mb': 10,
        'supported_formats': ['txt', 'pdf', 'docx', 'html']
    },
    'retrieval': {
        'top_k': 5,
        'score_threshold': 0.5
    },
    'generation': {
        'prompt_template': 'default',
        'max_context_length': 4000
    }
}


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_directory = tempfile.mkdtemp()
    yield temp_directory
    shutil.rmtree(temp_directory)


@pytest.fixture(scope="session")
def sample_documents(temp_dir):
    """Create sample documents for testing."""
    documents = {}
    
    # Sample text document
    txt_content = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that work and react like humans. Some of the activities 
    computers with artificial intelligence are designed for include speech recognition, 
    learning, planning, and problem solving.
    """
    txt_path = Path(temp_dir) / "sample_ai.txt"
    txt_path.write_text(txt_content.strip())
    documents['txt'] = str(txt_path)
    
    # Sample HTML document
    html_content = """
    <html>
    <head><title>Machine Learning</title></head>
    <body>
    <h1>Machine Learning</h1>
    <p>Machine learning is a method of data analysis that automates analytical 
    model building. It is a branch of artificial intelligence based on the idea 
    that systems can learn from data, identify patterns and make decisions with 
    minimal human intervention.</p>
    </body>
    </html>
    """
    html_path = Path(temp_dir) / "sample_ml.html"
    html_path.write_text(html_content.strip())
    documents['html'] = str(html_path)
    
    # Sample JSON document (as text)
    json_content = {
        "topic": "Deep Learning",
        "description": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        "applications": ["image recognition", "natural language processing", "speech recognition"]
    }
    json_path = Path(temp_dir) / "sample_dl.json"
    json_path.write_text(json.dumps(json_content, indent=2))
    documents['json'] = str(json_path)
    
    return documents


@pytest.fixture
def mock_config():
    """Mock configuration manager."""
    config = Mock(spec=ConfigManager)
    config.app = Mock()
    config.app.name = TEST_CONFIG['app']['name']
    config.app.version = TEST_CONFIG['app']['version']
    config.app.environment = TEST_CONFIG['app']['environment']
    config.app.debug = TEST_CONFIG['app']['debug']
    config.app.dict.return_value = TEST_CONFIG['app']
    
    config.embeddings = Mock()
    config.embeddings.provider = TEST_CONFIG['embeddings']['provider']
    config.embeddings.model_name = TEST_CONFIG['embeddings']['model_name']
    config.embeddings.dict.return_value = TEST_CONFIG['embeddings']
    
    config.database = Mock()
    config.database.provider = TEST_CONFIG['vector_db']['provider']
    config.database.dict.return_value = TEST_CONFIG['vector_db']
    
    config.get_section = Mock(side_effect=lambda section: TEST_CONFIG.get(section, {}))
    config.get_model_config = Mock(side_effect=lambda model: TEST_CONFIG.get(model, {}))
    config.get = Mock(side_effect=lambda key, default=None: TEST_CONFIG.get(key, default))
    config.validate_config.return_value = {}
    
    return config


@pytest.fixture
def sample_document():
    """Create a sample Document object."""
    return Document(
        filename="sample.txt",
        file_path="/path/to/sample.txt",
        content="This is sample document content for testing purposes.",
        document_type="text",
        file_size=len("This is sample document content for testing purposes."),
        processing_status=ProcessingStatus.PROCESSED,
        metadata={"author": "test", "created_date": "2024-01-01"}
    )


@pytest.fixture
def sample_text_chunks():
    """Create sample TextChunk objects."""
    return [
        TextChunk(
            content="First chunk of text content.",
            chunk_id="chunk_001",
            document_id="doc_001",
            start_char=0,
            end_char=30,
            metadata={"chunk_index": 0}
        ),
        TextChunk(
            content="Second chunk of text content.",
            chunk_id="chunk_002", 
            document_id="doc_001",
            start_char=31,
            end_char=61,
            metadata={"chunk_index": 1}
        ),
        TextChunk(
            content="Third chunk with different content.",
            chunk_id="chunk_003",
            document_id="doc_001", 
            start_char=62,
            end_char=97,
            metadata={"chunk_index": 2}
        )
    ]


@pytest.fixture
def sample_generation_response():
    """Create a sample GenerationResponse object."""
    return GenerationResponse(
        answer="This is a generated response to the user query.",
        sources=[
            {"chunk_id": "chunk_001", "score": 0.95, "content": "First chunk content"},
            {"chunk_id": "chunk_002", "score": 0.88, "content": "Second chunk content"}
        ],
        query="What is the sample query?",
        model_used="gpt-3.5-turbo",
        processing_time=1.25,
        metadata={"temperature": 0.7, "tokens_used": 150}
    )


@pytest.fixture
def mock_vector_store():
    """Mock vector database store."""
    mock_store = Mock()
    mock_store.add_texts.return_value = ["id1", "id2", "id3"]
    mock_store.similarity_search_with_score.return_value = [
        (Mock(page_content="Sample content 1", metadata={"source": "doc1"}), 0.95),
        (Mock(page_content="Sample content 2", metadata={"source": "doc2"}), 0.88)
    ]
    mock_store.get_collection_info.return_value = {
        "name": "test_collection",
        "count": 10,
        "dimension": 384
    }
    return mock_store


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model."""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3] * 128]  # Mock 384-dim embedding
    return mock_model


@pytest.fixture
def mock_llm():
    """Mock language model."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="This is a mocked LLM response.")
    mock_llm.generate.return_value = Mock(generations=[[Mock(text="Mocked response")]])
    return mock_llm


@pytest.fixture
def isolated_rag_system(mock_config):
    """Create an isolated RAG system for testing."""
    with pytest.MonkeyPatch.context() as m:
        # Mock the initialization of external services
        m.setattr('src.agents.document_parser.DocumentParserAgent', Mock)
        m.setattr('src.agents.chunking_embedding.ChunkingEmbeddingAgent', Mock)
        m.setattr('src.agents.retrieval.RetrievalAgent', Mock)
        m.setattr('src.agents.generation.GenerationAgent', Mock)
        
        rag = RAGSystem()
        rag.config = mock_config
        return rag


# Test Utilities
class TestUtils:
    """Utility class for test helpers."""
    
    @staticmethod
    def create_test_file(content: str, file_path: str) -> str:
        """Create a test file with given content."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    @staticmethod
    def assert_document_valid(document: Document):
        """Assert that a document object is valid."""
        assert document.filename is not None
        assert document.file_path is not None
        assert document.content is not None
        assert document.processing_status in [s.value for s in ProcessingStatus]
    
    @staticmethod
    def assert_chunks_valid(chunks: List[TextChunk]):
        """Assert that text chunks are valid."""
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.chunk_id is not None
            assert chunk.document_id is not None
    
    @staticmethod
    def mock_api_response(status_code: int = 200, json_data: Dict[Any, Any] = None):
        """Create a mock API response."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or {}
        return mock_response


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Performance Testing Fixtures
@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Async Testing Support
@pytest.fixture
def event_loop():
    """Create an event loop for async testing."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Skip markers for conditional testing
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "skip_if_no_api: skip test if API credentials are not available"
    )
    config.addinivalue_line(
        "markers", "skip_if_no_vector_db: skip test if vector database is not available"
    )


def pytest_runtest_setup(item):
    """Setup for test runs."""
    # Skip tests requiring API if no credentials
    if item.get_closest_marker("skip_if_no_api"):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("No API credentials available")
    
    # Skip tests requiring vector DB if not available
    if item.get_closest_marker("skip_if_no_vector_db"):
        # Could add vector DB connectivity check here
        pass


# Test Data Constants
TEST_QUERIES = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "Explain deep learning concepts",
    "What are the applications of AI?",
    "Compare different AI techniques"
]

TEST_DOCUMENTS_CONTENT = {
    "ai_basics": "Artificial Intelligence is the simulation of human intelligence in machines...",
    "ml_fundamentals": "Machine Learning is a subset of AI that focuses on algorithms...", 
    "dl_concepts": "Deep Learning uses neural networks with multiple layers...",
    "ai_applications": "AI is used in healthcare, finance, autonomous vehicles...",
    "ai_ethics": "AI ethics involves considerations about fairness, transparency..."
}