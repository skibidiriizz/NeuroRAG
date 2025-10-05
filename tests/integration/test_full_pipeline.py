"""
Integration tests for complete RAG system workflows.

This module tests end-to-end functionality of the RAG system,
including document processing, retrieval, and generation workflows.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import time

from src.core.rag_system import RAGSystem
from src.models.document import Document, ProcessingStatus, GenerationResponse


@pytest.mark.integration
class TestFullRAGPipeline:
    """Integration tests for complete RAG pipeline."""
    
    @pytest.fixture
    def mock_rag_system(self, mock_config):
        """Create a mock RAG system with all agents."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock all agents
        rag.document_parser = Mock()
        rag.chunking_embedding = Mock()
        rag.retrieval = Mock()
        rag.generation = Mock()
        rag.evaluation = Mock()
        rag.orchestrator = Mock()
        
        return rag
    
    def test_document_to_query_pipeline(self, mock_rag_system, temp_dir):
        """Test complete pipeline from document ingestion to query response."""
        rag = mock_rag_system
        
        # Create test document
        test_content = "Machine learning is a subset of artificial intelligence."
        test_file = Path(temp_dir) / "ml_doc.txt"
        test_file.write_text(test_content)
        
        # Mock document processing
        mock_document = Document(
            filename="ml_doc.txt",
            file_path=str(test_file),
            content=test_content,
            document_type="text",
            processing_status=ProcessingStatus.PROCESSED
        )
        rag.document_parser.parse_document.return_value = mock_document
        
        # Mock chunking and embedding
        mock_chunks = ["Machine learning is a subset", "of artificial intelligence"]
        rag.chunking_embedding.process_document.return_value = mock_chunks
        
        # Step 1: Add document to system
        processed_doc = rag.add_document(str(test_file))
        assert processed_doc.processing_status == ProcessingStatus.PROCESSED
        assert processed_doc.content == test_content
        
        # Mock retrieval results
        retrieval_results = [
            {"content": "Machine learning is a subset", "score": 0.95, "chunk_id": "chunk_1"},
            {"content": "of artificial intelligence", "score": 0.88, "chunk_id": "chunk_2"}
        ]
        rag.retrieval.retrieve.return_value = retrieval_results
        
        # Mock generation response
        mock_response = GenerationResponse(
            answer="Machine learning is indeed a subset of artificial intelligence that focuses on enabling computers to learn from data.",
            sources=[
                {"content": "Machine learning is a subset", "score": 0.95, "chunk_id": "chunk_1"},
                {"content": "of artificial intelligence", "score": 0.88, "chunk_id": "chunk_2"}
            ],
            query="What is machine learning?",
            model_used="gpt-3.5-turbo",
            processing_time=1.2
        )
        rag.generation.generate_response.return_value = mock_response
        
        # Step 2: Query the system
        query = "What is machine learning?"
        response = rag.query(query)
        
        # Verify the complete pipeline
        assert response.answer is not None
        assert len(response.sources) > 0
        assert response.query == query
        assert "machine learning" in response.answer.lower()
        
        # Verify method calls
        rag.document_parser.parse_document.assert_called_once_with(str(test_file))
        rag.chunking_embedding.process_document.assert_called_once_with(mock_document)
        rag.retrieval.retrieve.assert_called_once_with(query, top_k=5, score_threshold=0.0)
        rag.generation.generate_response.assert_called_once()
    
    def test_multiple_documents_pipeline(self, mock_rag_system, temp_dir):
        """Test pipeline with multiple documents."""
        rag = mock_rag_system
        
        # Create multiple test documents
        documents_data = [
            ("ai_doc.txt", "Artificial Intelligence is the simulation of human intelligence."),
            ("ml_doc.txt", "Machine Learning is a method of data analysis."),
            ("dl_doc.txt", "Deep Learning uses neural networks with multiple layers.")
        ]
        
        created_files = []
        for filename, content in documents_data:
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)
            created_files.append(str(file_path))
        
        # Mock document processing for each document
        processed_documents = []
        for i, (filename, content) in enumerate(documents_data):
            mock_doc = Document(
                filename=filename,
                file_path=created_files[i],
                content=content,
                document_type="text",
                processing_status=ProcessingStatus.PROCESSED
            )
            processed_documents.append(mock_doc)
        
        rag.document_parser.parse_document.side_effect = processed_documents
        rag.chunking_embedding.process_document.return_value = ["chunk1", "chunk2"]
        
        # Process all documents
        results = []
        for file_path in created_files:
            result = rag.add_document(file_path)
            results.append(result)
        
        assert len(results) == 3
        assert all(doc.processing_status == ProcessingStatus.PROCESSED for doc in results)
        assert rag.document_parser.parse_document.call_count == 3
        assert rag.chunking_embedding.process_document.call_count == 3
        
        # Test querying across all documents
        mixed_retrieval_results = [
            {"content": "Artificial Intelligence is the simulation", "score": 0.92, "source": "ai_doc.txt"},
            {"content": "Machine Learning is a method", "score": 0.89, "source": "ml_doc.txt"},
            {"content": "Deep Learning uses neural networks", "score": 0.86, "source": "dl_doc.txt"}
        ]
        rag.retrieval.retrieve.return_value = mixed_retrieval_results
        
        comprehensive_response = GenerationResponse(
            answer="AI encompasses various subfields including Machine Learning and Deep Learning.",
            sources=mixed_retrieval_results,
            query="What are the different types of AI?",
            model_used="gpt-3.5-turbo",
            processing_time=1.5
        )
        rag.generation.generate_response.return_value = comprehensive_response
        
        response = rag.query("What are the different types of AI?")
        
        assert response.answer is not None
        assert len(response.sources) == 3  # Sources from all documents
        assert any("ai" in source.get("source", "").lower() for source in response.sources)
    
    def test_query_without_documents(self, mock_rag_system):
        """Test querying system with no documents."""
        rag = mock_rag_system
        
        # Mock empty retrieval results
        rag.retrieval.retrieve.return_value = []
        
        # Mock generation response for no sources
        no_source_response = GenerationResponse(
            answer="I don't have enough information to answer your question.",
            sources=[],
            query="What is quantum computing?",
            model_used="gpt-3.5-turbo",
            processing_time=0.8
        )
        rag.generation.generate_response.return_value = no_source_response
        
        response = rag.query("What is quantum computing?")
        
        assert response.answer is not None
        assert len(response.sources) == 0
        assert "don't have enough information" in response.answer
    
    def test_error_handling_in_pipeline(self, mock_rag_system, temp_dir):
        """Test error handling throughout the pipeline."""
        rag = mock_rag_system
        
        # Test document parsing error
        test_file = Path(temp_dir) / "corrupt.txt"
        test_file.write_text("Test content")
        
        # Mock parsing failure
        failed_doc = Document(
            filename="corrupt.txt",
            file_path=str(test_file),
            content="",
            document_type="text",
            processing_status=ProcessingStatus.FAILED,
            error_message="File corrupted"
        )
        rag.document_parser.parse_document.return_value = failed_doc
        
        result = rag.add_document(str(test_file))
        
        assert result.processing_status == ProcessingStatus.FAILED
        assert "corrupted" in result.error_message
        rag.chunking_embedding.process_document.assert_not_called()
        
        # Test retrieval error
        rag.retrieval.retrieve.side_effect = Exception("Vector database connection failed")
        
        with pytest.raises(Exception, match="Vector database connection failed"):
            rag.query("What is AI?")
        
        # Reset retrieval mock and test generation error
        rag.retrieval.retrieve.side_effect = None
        rag.retrieval.retrieve.return_value = [{"content": "test", "score": 0.9}]
        rag.generation.generate_response.side_effect = Exception("LLM API limit exceeded")
        
        with pytest.raises(Exception, match="LLM API limit exceeded"):
            rag.query("What is AI?")
    
    def test_different_query_types(self, mock_rag_system):
        """Test different types of queries."""
        rag = mock_rag_system
        
        # Mock retrieval results
        sample_chunks = [
            {"content": "AI has applications in healthcare", "score": 0.95},
            {"content": "Machine learning improves medical diagnosis", "score": 0.90}
        ]
        rag.retrieval.retrieve.return_value = sample_chunks
        
        # Test detailed query
        detailed_response = GenerationResponse(
            answer="AI has extensive applications in healthcare, particularly in medical diagnosis where machine learning algorithms can analyze medical images and patient data to assist doctors.",
            sources=sample_chunks,
            query="Tell me about AI in healthcare",
            model_used="gpt-3.5-turbo",
            processing_time=1.8
        )
        rag.generation.generate_response.return_value = detailed_response
        
        result = rag.detailed_query("Tell me about AI in healthcare")
        assert len(result.answer) > 50  # Detailed answer should be longer
        
        # Test summary query
        summary_response = GenerationResponse(
            answer="AI helps healthcare through medical diagnosis improvements.",
            sources=sample_chunks,
            query="Summarize AI in healthcare",
            model_used="gpt-3.5-turbo",
            processing_time=1.0
        )
        rag.generation.generate_response.return_value = summary_response
        
        result = rag.summarize_query("Summarize AI in healthcare")
        assert len(result.answer) < 100  # Summary should be shorter
        
        # Test factual query
        factual_response = GenerationResponse(
            answer="Yes, AI is used in healthcare for medical diagnosis.",
            sources=sample_chunks,
            query="Is AI used in healthcare?",
            model_used="gpt-3.5-turbo",
            processing_time=0.9
        )
        rag.generation.generate_response.return_value = factual_response
        
        result = rag.factual_query("Is AI used in healthcare?")
        assert result.answer.startswith("Yes") or result.answer.startswith("No")
    
    def test_system_configuration_impact(self, mock_rag_system):
        """Test how configuration changes impact pipeline behavior."""
        rag = mock_rag_system
        
        # Test with different top_k values
        sample_chunks = [
            {"content": f"Content chunk {i}", "score": 0.9 - i*0.1} 
            for i in range(10)
        ]
        
        # Test top_k = 3
        rag.retrieval.retrieve.return_value = sample_chunks[:3]
        rag.generation.generate_response.return_value = GenerationResponse(
            answer="Answer with 3 sources",
            sources=sample_chunks[:3],
            query="Test query",
            model_used="gpt-3.5-turbo",
            processing_time=1.0
        )
        
        result = rag.query("Test query", top_k=3)
        assert len(result.sources) == 3
        rag.retrieval.retrieve.assert_called_with("Test query", top_k=3, score_threshold=0.0)
        
        # Test top_k = 7
        rag.retrieval.retrieve.return_value = sample_chunks[:7]
        rag.generation.generate_response.return_value = GenerationResponse(
            answer="Answer with 7 sources",
            sources=sample_chunks[:7],
            query="Test query",
            model_used="gpt-3.5-turbo",
            processing_time=1.2
        )
        
        result = rag.query("Test query", top_k=7)
        assert len(result.sources) == 7
        rag.retrieval.retrieve.assert_called_with("Test query", top_k=7, score_threshold=0.0)
    
    @pytest.mark.asyncio
    async def test_orchestrated_workflow(self, mock_rag_system):
        """Test orchestrated workflow execution."""
        rag = mock_rag_system
        
        # Mock successful orchestrated workflow
        workflow_result = {
            "workflow_type": "standard",
            "generated_response": GenerationResponse(
                answer="This is an orchestrated response",
                sources=[{"content": "orchestrated source", "score": 0.95}],
                query="What is orchestration?",
                model_used="gpt-3.5-turbo",
                processing_time=2.1
            ),
            "processing_times": {
                "retrieval": 0.5,
                "generation": 1.2,
                "evaluation": 0.4
            },
            "evaluation_results": {
                "faithfulness": 0.92,
                "relevance": 0.89
            }
        }
        
        async def mock_execute_workflow(*args, **kwargs):
            return workflow_result
            
        rag.orchestrator.execute_workflow = mock_execute_workflow
        
        result = await rag.execute_workflow("standard", query="What is orchestration?")
        
        assert result["workflow_type"] == "standard"
        assert result["generated_response"].answer == "This is an orchestrated response"
        assert "processing_times" in result
        assert "evaluation_results" in result
    
    def test_performance_metrics_tracking(self, mock_rag_system):
        """Test that performance metrics are properly tracked."""
        rag = mock_rag_system
        
        initial_doc_count = rag.metrics.total_documents
        initial_chunk_count = rag.metrics.total_chunks
        
        # Mock document processing
        mock_doc = Document(
            filename="test.txt",
            file_path="test.txt",
            content="test content",
            document_type="text",
            processing_status=ProcessingStatus.PROCESSED
        )
        rag.document_parser.parse_document.return_value = mock_doc
        rag.chunking_embedding.process_document.return_value = ["chunk1", "chunk2", "chunk3"]
        
        # Add document and check metrics update
        rag.add_document("test.txt")
        
        assert rag.metrics.total_documents == initial_doc_count + 1
        assert rag.metrics.total_chunks == initial_chunk_count + 3
        assert rag.metrics.avg_response_time > 0  # Should be updated with processing time


@pytest.mark.integration
class TestRAGSystemHealth:
    """Integration tests for RAG system health monitoring."""
    
    def test_comprehensive_health_check(self, mock_rag_system):
        """Test comprehensive system health check."""
        rag = mock_rag_system
        
        # Mock healthy components
        rag.chunking_embedding.get_collection_info.return_value = {
            "name": "test_collection",
            "count": 100,
            "dimension": 384
        }
        rag.retrieval.test_connection.return_value = {"status": "connected"}
        rag.generation.test_generation.return_value = {"status": "success"}
        
        health = rag.health_check()
        
        assert health["overall_status"] == "healthy"
        assert all(status in ["healthy", "not_available"] 
                  for status in health["components"].values())
    
    def test_degraded_system_health(self, mock_rag_system):
        """Test system health with degraded components."""
        rag = mock_rag_system
        
        # Mock failing components
        rag.chunking_embedding.get_collection_info.side_effect = Exception("DB connection failed")
        rag.retrieval.test_connection.return_value = {
            "status": "failed", 
            "error": "Connection timeout"
        }
        rag.generation.test_generation.return_value = {
            "status": "error", 
            "error": "API rate limit exceeded"
        }
        
        health = rag.health_check()
        
        assert health["overall_status"] == "degraded"
        assert "error" in health["components"]["chunking_embedding"]
        assert "error" in health["components"]["retrieval"]
        assert "error" in health["components"]["generation"]


@pytest.mark.slow
@pytest.mark.integration
class TestRAGSystemPerformance:
    """Performance integration tests."""
    
    def test_bulk_document_processing_performance(self, mock_rag_system, temp_dir, performance_timer):
        """Test performance of bulk document processing."""
        rag = mock_rag_system
        
        # Create multiple test documents
        num_docs = 10
        test_files = []
        for i in range(num_docs):
            content = f"This is test document {i} with content about topic {i}. " * 20
            file_path = Path(temp_dir) / f"doc_{i}.txt"
            file_path.write_text(content)
            test_files.append(str(file_path))
        
        # Mock processing for each document
        def mock_parse_document(file_path):
            return Document(
                filename=Path(file_path).name,
                file_path=file_path,
                content=f"Content for {Path(file_path).name}",
                document_type="text",
                processing_status=ProcessingStatus.PROCESSED
            )
        
        rag.document_parser.parse_document.side_effect = mock_parse_document
        rag.chunking_embedding.process_document.return_value = ["chunk1", "chunk2"]
        
        # Time bulk processing
        performance_timer.start()
        for file_path in test_files:
            rag.add_document(file_path)
        performance_timer.stop()
        
        # Performance assertions
        assert performance_timer.elapsed < 5.0  # Should process 10 docs in under 5 seconds
        assert rag.metrics.total_documents == num_docs
        assert rag.document_parser.parse_document.call_count == num_docs
    
    def test_query_response_time(self, mock_rag_system, performance_timer):
        """Test query response time performance."""
        rag = mock_rag_system
        
        # Mock retrieval and generation
        large_chunks = [
            {"content": f"Large content chunk {i} with detailed information", "score": 0.9 - i*0.05}
            for i in range(20)
        ]
        rag.retrieval.retrieve.return_value = large_chunks[:5]
        
        rag.generation.generate_response.return_value = GenerationResponse(
            answer="This is a comprehensive answer based on multiple sources.",
            sources=large_chunks[:5],
            query="Complex query requiring analysis",
            model_used="gpt-3.5-turbo",
            processing_time=0.8
        )
        
        # Time query processing
        performance_timer.start()
        response = rag.query("Complex query requiring analysis", top_k=5)
        performance_timer.stop()
        
        # Performance assertions
        assert performance_timer.elapsed < 2.0  # Should respond in under 2 seconds
        assert response.answer is not None
        assert len(response.sources) == 5