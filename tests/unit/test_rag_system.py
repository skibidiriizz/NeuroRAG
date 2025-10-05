"""
Unit tests for RAG System core functionality.

This module tests the main RAG system orchestration, configuration,
and integration between different agents.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import asyncio

from src.core.rag_system import RAGSystem, create_rag_system, quick_setup
from src.models.document import Document, ProcessingStatus, GenerationResponse
from src.core.config_manager import ConfigManager


class TestRAGSystem:
    """Test cases for RAG System core functionality."""
    
    def test_init_with_default_config(self):
        """Test RAG system initialization with default configuration."""
        with patch('src.core.rag_system.ConfigManager') as mock_config_manager:
            mock_config = Mock()
            mock_config.validate_config.return_value = {}
            mock_config_manager.return_value = mock_config
            
            with patch.object(RAGSystem, '_initialize_agents'):
                rag = RAGSystem()
                
                assert rag.config == mock_config
                assert rag.document_parser is None  # Not initialized in test
                assert rag.metrics is not None
    
    def test_init_with_custom_config(self):
        """Test RAG system initialization with custom configuration."""
        config_path = "/path/to/config.yaml"
        
        with patch('src.core.rag_system.ConfigManager') as mock_config_manager:
            mock_config = Mock()
            mock_config.validate_config.return_value = {}
            mock_config_manager.return_value = mock_config
            
            with patch.object(RAGSystem, '_initialize_agents'):
                rag = RAGSystem(config_path)
                
                mock_config_manager.assert_called_once_with(config_path)
                mock_config.update_from_env.assert_called_once()
    
    @patch('src.core.rag_system.DocumentParserAgent')
    @patch('src.core.rag_system.ChunkingEmbeddingAgent')
    @patch('src.core.rag_system.RetrievalAgent')
    @patch('src.core.rag_system.GenerationAgent')
    def test_initialize_agents(self, mock_gen, mock_ret, mock_chunk, mock_parser, mock_config):
        """Test agent initialization."""
        # Setup mock config
        mock_config.get_section.return_value = {}
        mock_config.get_model_config.return_value = {}
        mock_config.get.return_value = 3
        
        rag = RAGSystem()
        rag.config = mock_config
        rag._initialize_agents()
        
        # Verify agents were created
        mock_parser.assert_called_once()
        mock_chunk.assert_called_once()
        mock_ret.assert_called_once()
        mock_gen.assert_called_once()
    
    def test_add_document_success(self, mock_config, sample_document):
        """Test successful document addition."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock agents
        rag.document_parser = Mock()
        rag.document_parser.parse_document.return_value = sample_document
        
        rag.chunking_embedding = Mock()
        rag.chunking_embedding.process_document.return_value = ["chunk1", "chunk2"]
        
        result = rag.add_document("/path/to/test.txt")
        
        assert result == sample_document
        assert rag.metrics.total_documents > 0
        rag.document_parser.parse_document.assert_called_once_with("/path/to/test.txt")
        rag.chunking_embedding.process_document.assert_called_once_with(sample_document)
    
    def test_add_document_parser_failure(self, mock_config):
        """Test document addition with parser failure."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Create failed document
        failed_doc = Document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            content="",
            document_type="text",
            processing_status=ProcessingStatus.FAILED,
            error_message="Parsing failed"
        )
        
        rag.document_parser = Mock()
        rag.document_parser.parse_document.return_value = failed_doc
        rag.chunking_embedding = Mock()
        
        result = rag.add_document("/path/to/test.txt")
        
        assert result.processing_status == ProcessingStatus.FAILED
        assert "Parsing failed" in result.error_message
        rag.chunking_embedding.process_document.assert_not_called()
    
    def test_add_documents_from_directory(self, mock_config, temp_dir):
        """Test adding documents from a directory."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Create test files
        (Path(temp_dir) / "doc1.txt").write_text("Content 1")
        (Path(temp_dir) / "doc2.txt").write_text("Content 2")
        (Path(temp_dir) / "doc3.pdf").write_text("Content 3")  # Mock PDF
        
        # Mock agents
        rag.document_parser = Mock()
        rag.document_parser.get_supported_formats.return_value = ['txt', 'pdf']
        rag.document_parser.parse_document.return_value = Document(
            filename="test.txt",
            file_path="test.txt",
            content="test content",
            document_type="text",
            processing_status=ProcessingStatus.PROCESSED
        )
        
        rag.chunking_embedding = Mock()
        rag.chunking_embedding.process_document.return_value = ["chunk1"]
        
        documents = rag.add_documents(temp_dir)
        
        assert len(documents) == 3  # All files should be processed
        assert rag.document_parser.parse_document.call_count == 3
    
    def test_add_documents_directory_not_found(self, mock_config):
        """Test adding documents from non-existent directory."""
        rag = RAGSystem()
        rag.config = mock_config
        
        with pytest.raises(FileNotFoundError):
            rag.add_documents("/non/existent/directory")
    
    def test_query_success(self, mock_config, sample_generation_response):
        """Test successful query processing."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock agents
        rag.retrieval = Mock()
        rag.retrieval.retrieve.return_value = ["chunk1", "chunk2"]
        
        rag.generation = Mock()
        rag.generation.generate_response.return_value = sample_generation_response
        
        result = rag.query("What is AI?")
        
        assert result == sample_generation_response
        rag.retrieval.retrieve.assert_called_once_with("What is AI?", top_k=5, score_threshold=0.0)
        rag.generation.generate_response.assert_called_once()
    
    def test_query_with_parameters(self, mock_config, sample_generation_response):
        """Test query with custom parameters."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.retrieval = Mock()
        rag.retrieval.retrieve.return_value = ["chunk1"]
        
        rag.generation = Mock()
        rag.generation.generate_response.return_value = sample_generation_response
        
        result = rag.query("What is AI?", top_k=3, score_threshold=0.8, prompt_template="detailed")
        
        rag.retrieval.retrieve.assert_called_once_with("What is AI?", top_k=3, score_threshold=0.8)
        rag.generation.generate_response.assert_called_once_with(
            query="What is AI?",
            retrieval_results=["chunk1"],
            prompt_template="detailed"
        )
    
    def test_query_agents_not_initialized(self, mock_config):
        """Test query when agents are not initialized."""
        rag = RAGSystem()
        rag.config = mock_config
        rag.retrieval = None
        rag.generation = None
        
        with pytest.raises(RuntimeError, match="Agents not properly initialized"):
            rag.query("What is AI?")
    
    def test_query_with_sources(self, mock_config, sample_generation_response):
        """Test query_with_sources method."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.retrieval = Mock()
        rag.retrieval.retrieve.return_value = ["chunk1", "chunk2"]
        
        rag.generation = Mock()
        rag.generation.generate_response.return_value = sample_generation_response
        
        result = rag.query_with_sources("What is AI?")
        
        assert result == sample_generation_response
        assert len(result.sources) > 0
    
    def test_detailed_query(self, mock_config, sample_generation_response):
        """Test detailed query method."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.retrieval = Mock()
        rag.retrieval.retrieve.return_value = ["chunk1", "chunk2", "chunk3"]
        
        rag.generation = Mock()
        rag.generation.generate_response.return_value = sample_generation_response
        
        result = rag.detailed_query("Explain AI in detail", top_k=10)
        
        rag.retrieval.retrieve.assert_called_once_with("Explain AI in detail", top_k=10, score_threshold=0.0)
        rag.generation.generate_response.assert_called_once_with(
            query="Explain AI in detail",
            retrieval_results=["chunk1", "chunk2", "chunk3"],
            prompt_template="detailed"
        )
    
    def test_summarize_query(self, mock_config, sample_generation_response):
        """Test summary query method."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.retrieval = Mock()
        rag.retrieval.retrieve.return_value = ["chunk1", "chunk2"]
        
        rag.generation = Mock()
        rag.generation.generate_response.return_value = sample_generation_response
        
        result = rag.summarize_query("Summarize AI concepts")
        
        rag.generation.generate_response.assert_called_once_with(
            query="Summarize AI concepts",
            retrieval_results=["chunk1", "chunk2"],
            prompt_template="summary"
        )
    
    def test_factual_query(self, mock_config, sample_generation_response):
        """Test factual query method."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.retrieval = Mock()
        rag.retrieval.retrieve.return_value = ["chunk1"]
        
        rag.generation = Mock()
        rag.generation.generate_response.return_value = sample_generation_response
        
        result = rag.factual_query("What year was AI founded?", score_threshold=0.9)
        
        rag.retrieval.retrieve.assert_called_once_with(
            "What year was AI founded?", 
            top_k=3, 
            score_threshold=0.9
        )
        rag.generation.generate_response.assert_called_once_with(
            query="What year was AI founded?",
            retrieval_results=["chunk1"],
            prompt_template="factual"
        )
    
    def test_get_system_status(self, mock_config):
        """Test system status retrieval."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock agents
        rag.document_parser = Mock()
        rag.chunking_embedding = Mock()
        rag.retrieval = Mock()
        rag.generation = Mock()
        
        # Mock vector store info
        rag.chunking_embedding.vector_store = Mock()
        rag.chunking_embedding.get_collection_info.return_value = {
            "name": "test_collection",
            "count": 100
        }
        
        status = rag.get_system_status()
        
        assert status['system_info']['name'] == 'TestRAGSystem'
        assert status['agents_status']['document_parser'] == 'initialized'
        assert status['agents_status']['chunking_embedding'] == 'initialized'
        assert status['agents_status']['retrieval'] == 'initialized'
        assert status['agents_status']['generation'] == 'initialized'
        assert 'metrics' in status
        assert 'configuration' in status
        assert 'vector_database' in status
    
    def test_health_check(self, mock_config):
        """Test system health check."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock healthy agents
        rag.document_parser = Mock()
        
        rag.chunking_embedding = Mock()
        rag.chunking_embedding.get_collection_info.return_value = {"name": "test"}
        
        rag.retrieval = Mock()
        rag.retrieval.test_connection.return_value = {"status": "connected"}
        
        rag.generation = Mock()
        rag.generation.test_generation.return_value = {"status": "success"}
        
        health = rag.health_check()
        
        assert health['overall_status'] == 'healthy'
        assert health['components']['document_parser'] == 'healthy'
        assert health['components']['chunking_embedding'] == 'healthy'
        assert health['components']['retrieval'] == 'healthy'
        assert health['components']['generation'] == 'healthy'
    
    def test_health_check_with_errors(self, mock_config):
        """Test health check with component errors."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock failing agents
        rag.document_parser = None  # Not initialized
        
        rag.chunking_embedding = Mock()
        rag.chunking_embedding.get_collection_info.side_effect = Exception("DB error")
        
        rag.retrieval = Mock()
        rag.retrieval.test_connection.return_value = {"status": "failed", "error": "connection failed"}
        
        rag.generation = Mock()
        rag.generation.test_generation.return_value = {"status": "error", "error": "model error"}
        
        health = rag.health_check()
        
        assert health['overall_status'] == 'degraded'
        assert health['components']['document_parser'] == 'not_initialized'
        assert 'error' in health['components']['chunking_embedding']
        assert 'error' in health['components']['retrieval']
        assert 'error' in health['components']['generation']
    
    def test_validate_document(self, mock_config):
        """Test document validation."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.document_parser = Mock()
        rag.document_parser.validate_file.return_value = {
            'valid': True,
            'file_size': 1024,
            'file_type': 'text'
        }
        
        result = rag.validate_document("/path/to/test.txt")
        
        assert result['valid'] is True
        assert result['file_size'] == 1024
        rag.document_parser.validate_file.assert_called_once_with("/path/to/test.txt")
    
    def test_get_supported_formats(self, mock_config):
        """Test getting supported file formats."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.document_parser = Mock()
        rag.document_parser.get_supported_formats.return_value = ['txt', 'pdf', 'docx']
        
        formats = rag.get_supported_formats()
        
        assert formats == ['txt', 'pdf', 'docx']
    
    def test_update_configuration(self, mock_config):
        """Test configuration updates."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.update_configuration(temperature=0.8, max_tokens=2000)
        
        rag.config.set.assert_any_call('temperature', 0.8)
        rag.config.set.assert_any_call('max_tokens', 2000)
    
    def test_get_configuration(self, mock_config):
        """Test configuration retrieval."""
        rag = RAGSystem()
        rag.config = mock_config
        
        config = rag.get_configuration()
        
        assert 'app' in config
        assert 'chunking' in config
        assert 'embeddings' in config
        assert 'vector_db' in config
        # Sensitive data should be excluded
        assert 'api_key' not in str(config)
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, mock_config):
        """Test workflow execution."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock orchestrator
        rag.orchestrator = Mock()
        rag.orchestrator.execute_workflow = Mock()
        rag.orchestrator.execute_workflow.return_value = {"status": "completed"}
        
        # Make the mock async
        async def async_execute_workflow(*args, **kwargs):
            return {"status": "completed"}
        
        rag.orchestrator.execute_workflow = async_execute_workflow
        
        result = await rag.execute_workflow("standard", query="What is AI?")
        
        assert result["status"] == "completed"
    
    def test_execute_workflow_no_orchestrator(self, mock_config):
        """Test workflow execution without orchestrator."""
        rag = RAGSystem()
        rag.config = mock_config
        rag.orchestrator = None
        
        with pytest.raises(ValueError, match="Orchestrator not available"):
            asyncio.run(rag.execute_workflow("standard", query="What is AI?"))
    
    def test_get_available_workflows(self, mock_config):
        """Test getting available workflows."""
        rag = RAGSystem()
        rag.config = mock_config
        
        rag.orchestrator = Mock()
        rag.orchestrator.list_available_workflows.return_value = ["standard", "batch"]
        
        workflows = rag.get_available_workflows()
        
        assert workflows == ["standard", "batch"]
    
    def test_get_available_workflows_no_orchestrator(self, mock_config):
        """Test getting workflows without orchestrator."""
        rag = RAGSystem()
        rag.config = mock_config
        rag.orchestrator = None
        
        workflows = rag.get_available_workflows()
        
        assert workflows == []
    
    def test_context_manager(self, mock_config):
        """Test RAG system as context manager."""
        rag = RAGSystem()
        rag.config = mock_config
        
        with rag as system:
            assert system == rag
        # Context should exit without error
    
    def test_string_representations(self, mock_config):
        """Test string representations of RAG system."""
        rag = RAGSystem()
        rag.config = mock_config
        
        str_repr = str(rag)
        detailed_repr = repr(rag)
        
        assert 'RAGSystem' in str_repr
        assert 'TestRAGSystem' in str_repr
        assert 'RAGSystem' in detailed_repr
        assert 'documents=' in detailed_repr


class TestRAGSystemConvenienceFunctions:
    """Test convenience functions for RAG system creation."""
    
    def test_create_rag_system(self):
        """Test create_rag_system convenience function."""
        with patch('src.core.rag_system.RAGSystem') as mock_rag:
            result = create_rag_system("/path/to/config.yaml")
            
            mock_rag.assert_called_once_with("/path/to/config.yaml")
            assert result == mock_rag.return_value
    
    def test_quick_setup_with_existing_directory(self, temp_dir):
        """Test quick_setup with existing documents directory."""
        # Create test documents
        (Path(temp_dir) / "doc1.txt").write_text("Content 1")
        (Path(temp_dir) / "doc2.txt").write_text("Content 2")
        
        with patch('src.core.rag_system.RAGSystem') as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            
            result = quick_setup(temp_dir, "/path/to/config.yaml")
            
            mock_rag.assert_called_once_with("/path/to/config.yaml")
            mock_rag_instance.add_documents.assert_called_once_with(temp_dir)
            assert result == mock_rag_instance
    
    def test_quick_setup_with_nonexistent_directory(self):
        """Test quick_setup with non-existent documents directory."""
        with patch('src.core.rag_system.RAGSystem') as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            
            result = quick_setup("/non/existent/directory")
            
            mock_rag.assert_called_once_with(None)
            mock_rag_instance.add_documents.assert_not_called()
            assert result == mock_rag_instance


@pytest.mark.integration
class TestRAGSystemIntegration:
    """Integration tests for RAG System."""
    
    def test_end_to_end_document_processing(self, isolated_rag_system, sample_documents):
        """Test end-to-end document processing."""
        rag = isolated_rag_system
        
        # Mock successful processing chain
        mock_doc = Mock()
        mock_doc.processing_status = ProcessingStatus.PROCESSED
        mock_doc.filename = "test.txt"
        
        rag.document_parser.parse_document.return_value = mock_doc
        rag.chunking_embedding.process_document.return_value = ["chunk1", "chunk2"]
        
        # Process a document
        result = rag.add_document(list(sample_documents.values())[0])
        
        assert result.processing_status == ProcessingStatus.PROCESSED
        assert rag.metrics.total_documents > 0
    
    def test_end_to_end_query_processing(self, isolated_rag_system):
        """Test end-to-end query processing."""
        rag = isolated_rag_system
        
        # Mock successful query chain
        rag.retrieval.retrieve.return_value = [
            {"content": "AI is artificial intelligence", "score": 0.95}
        ]
        
        mock_response = GenerationResponse(
            answer="Artificial Intelligence (AI) refers to intelligent machines.",
            sources=[{"content": "AI is artificial intelligence", "score": 0.95}],
            query="What is AI?",
            model_used="test_model",
            processing_time=0.5
        )
        rag.generation.generate_response.return_value = mock_response
        
        # Process a query
        result = rag.query("What is AI?")
        
        assert result.answer == "Artificial Intelligence (AI) refers to intelligent machines."
        assert len(result.sources) > 0
        assert result.query == "What is AI?"