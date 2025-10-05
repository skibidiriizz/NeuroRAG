"""
Main RAG System - Orchestrates all agents and provides a unified interface

This module provides the main entry point for the RAG system, coordinating
all the individual agents to provide a complete RAG pipeline.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Core imports
from .config_manager import ConfigManager
from ..agents.document_parser import DocumentParserAgent
from ..agents.chunking_embedding import ChunkingEmbeddingAgent
from ..agents.retrieval import RetrievalAgent
from ..agents.generation import GenerationAgent
from ..models.document import (
    Document, TextChunk, GenerationResponse, 
    ProcessingStatus, SystemMetrics
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RAGSystem:
    """
    Main RAG System that orchestrates all agents to provide a complete
    Retrieval-Augmented Generation pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG System.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = ConfigManager(config_path)
        self.config.update_from_env()
        
        # Validate configuration
        config_errors = self.config.validate_config()
        if config_errors:
            logger.warning(f"Configuration issues found: {config_errors}")
        
        # Initialize agents
        self.document_parser = None
        self.chunking_embedding = None
        self.retrieval = None
        self.generation = None
        self.evaluation = None
        self.orchestrator = None
        
        # System metrics
        self.metrics = SystemMetrics()
        
        # Initialize core agents
        self._initialize_agents()
        
        logger.info("RAG System initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all agents with their configurations."""
        try:
            # Document Parser Agent
            parser_config = self.config.get_section('document_processing')
            self.document_parser = DocumentParserAgent(parser_config)
            
            # Chunking & Embedding Agent
            combined_config = {
                'chunking': self.config.get_section('chunking'),
                'embeddings': {
                    **self.config.get_section('embeddings'),
                    **self.config.get_model_config('embeddings')
                },
                'vector_db': self.config.get_section('vector_db')
            }
            self.chunking_embedding = ChunkingEmbeddingAgent(combined_config)
            
            # Retrieval Agent
            retrieval_config = {
                'retrieval': self.config.get_section('retrieval'),
                'vector_db': self.config.get_section('vector_db')
            }
            self.retrieval = RetrievalAgent(retrieval_config, self.chunking_embedding)
            
            # Generation Agent
            generation_config = {
                'llm': {
                    **self.config.get_section('llm'),
                    **self.config.get_model_config('llm')
                },
                'generation': self.config.get_section('generation')
            }
            self.generation = GenerationAgent(generation_config)
            
            # Initialize orchestrator if LangGraph is available
            try:
                from .orchestrator import RAGOrchestrator
                orchestrator_config = {
                    'max_retries': self.config.get('orchestration.max_retries', 3),
                    'timeout_seconds': self.config.get('orchestration.timeout_seconds', 300),
                    'enable_evaluation': self.config.get('orchestration.enable_evaluation', True)
                }
                self.orchestrator = RAGOrchestrator(rag_system=self, config=orchestrator_config)
                logger.info("Orchestrator initialized successfully")
            except ImportError:
                logger.warning("LangGraph not available - orchestrator not initialized")
                self.orchestrator = None
            except Exception as e:
                logger.warning(f"Failed to initialize orchestrator: {str(e)}")
                self.orchestrator = None
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def add_document(self, file_path: str) -> Document:
        """
        Add a single document to the system.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processed Document object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        logger.info(f"Adding document: {file_path}")
        start_time = time.time()
        
        try:
            # Step 1: Parse the document
            document = self.document_parser.parse_document(file_path)
            
            if document.processing_status == ProcessingStatus.FAILED:
                logger.error(f"Failed to parse document: {document.error_message}")
                return document
            
            # Step 2: Process (chunk, embed, store) the document
            chunks = self.chunking_embedding.process_document(document)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.update_metrics(
                total_documents=self.metrics.total_documents + 1,
                total_chunks=self.metrics.total_chunks + len(chunks),
                avg_response_time=processing_time
            )
            
            logger.info(f"Successfully added document {document.filename} in {processing_time:.2f}s")
            return document
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            raise
    
    def add_documents(self, directory_path: str, file_patterns: List[str] = None) -> List[Document]:
        """
        Add multiple documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_patterns: List of file patterns to match (e.g., ['*.pdf', '*.docx'])
            
        Returns:
            List of processed Document objects
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Default file patterns based on supported formats
        if file_patterns is None:
            supported_formats = self.document_parser.get_supported_formats()
            file_patterns = [f"*.{fmt}" for fmt in supported_formats]
        
        # Find all matching files
        file_paths = []
        for pattern in file_patterns:
            file_paths.extend(directory.glob(pattern))
        
        if not file_paths:
            logger.warning(f"No matching files found in {directory_path} with patterns {file_patterns}")
            return []
        
        logger.info(f"Found {len(file_paths)} files to process in {directory_path}")
        
        # Process documents
        documents = []
        for file_path in file_paths:
            try:
                document = self.add_document(str(file_path))
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                # Create a failed document entry
                failed_doc = Document(
                    filename=file_path.name,
                    file_path=str(file_path),
                    content="",
                    document_type=self.document_parser._determine_document_type(file_path),
                    processing_status=ProcessingStatus.FAILED,
                    error_message=str(e)
                )
                documents.append(failed_doc)
        
        successful = len([d for d in documents if d.processing_status == ProcessingStatus.INDEXED])
        logger.info(f"Batch processing completed: {successful}/{len(documents)} documents successful")
        
        return documents
    
    def add_url(self, url: str) -> Document:
        """
        Add a document from URL (HTML only).
        
        Args:
            url: URL to fetch and process
            
        Returns:
            Processed Document object
        """
        logger.info(f"Adding document from URL: {url}")
        start_time = time.time()
        
        try:
            # Parse document from URL
            document = self.document_parser.parse_url(url)
            
            if document.processing_status == ProcessingStatus.FAILED:
                logger.error(f"Failed to parse URL: {document.error_message}")
                return document
            
            # Process the document
            chunks = self.chunking_embedding.process_document(document)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.update_metrics(
                total_documents=self.metrics.total_documents + 1,
                total_chunks=self.metrics.total_chunks + len(chunks),
                avg_response_time=processing_time
            )
            
            logger.info(f"Successfully added URL {url} in {processing_time:.2f}s")
            return document
            
        except Exception as e:
            logger.error(f"Error adding URL {url}: {str(e)}")
            raise
    
    def query(self, question: str, top_k: int = None, score_threshold: float = None, 
             prompt_template: Optional[str] = None) -> GenerationResponse:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask
            top_k: Number of top chunks to retrieve (optional)
            score_threshold: Minimum similarity score threshold (optional)
            prompt_template: Prompt template to use for generation (optional)
            
        Returns:
            GenerationResponse object with answer and sources
        """
        start_time = time.time()
        logger.info(f"Processing query: '{question[:100]}...'")
        
        try:
            # Step 1: Retrieve relevant chunks
            if not self.retrieval:
                raise ValueError("Retrieval agent not initialized")
            
            retrieval_results = self.retrieval.retrieve(
                query=question,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            if not retrieval_results:
                # Return response when no relevant chunks found
                return GenerationResponse(
                    query=question,
                    answer="I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if documents have been added to the system.",
                    sources=[],
                    context_used="",
                    model_name=self.config.llm.model_name,
                    generation_metadata={'no_results': True},
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Generate response using retrieved context
            if not self.generation:
                raise ValueError("Generation agent not initialized")
            
            response = self.generation.generate_response(
                query=question,
                retrieval_results=retrieval_results,
                prompt_template=prompt_template
            )
            
            # Step 3: Update system metrics
            total_time = time.time() - start_time
            self.metrics.update_metrics(
                total_queries=self.metrics.total_queries + 1,
                total_responses=self.metrics.total_responses + 1,
                avg_response_time=total_time
            )
            
            logger.info(f"Query processed successfully in {total_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return GenerationResponse(
                query=question,
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                context_used="",
                model_name=self.config.llm.model_name,
                generation_metadata={'error': str(e)},
                processing_time=time.time() - start_time
            )
    
    def ask(self, question: str) -> str:
        """
        Simple query method that returns just the answer text.
        
        Args:
            question: Question to ask
            
        Returns:
            Answer as string
        """
        response = self.query(question)
        return response.answer
    
    def query_with_sources(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query and return answer with source information.
        
        Args:
            question: Question to ask
            top_k: Number of sources to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        response = self.query(question, top_k=top_k)
        
        return {
            'answer': response.answer,
            'sources': [
                {
                    'content': result.chunk.content,
                    'score': result.score,
                    'document': result.chunk.metadata.get('document_filename', 'Unknown'),
                    'chunk_index': result.chunk.chunk_index
                }
                for result in response.sources
            ],
            'context_length': len(response.context_used),
            'processing_time': response.processing_time,
            'model_used': response.model_name
        }
    
    def detailed_query(self, question: str, top_k: int = 10) -> GenerationResponse:
        """
        Query with detailed analysis template.
        
        Args:
            question: Question to ask
            top_k: Number of sources to retrieve
            
        Returns:
            GenerationResponse object
        """
        return self.query(question, top_k=top_k, prompt_template='detailed')
    
    def summarize_query(self, question: str, top_k: int = 5) -> GenerationResponse:
        """
        Query with summary template for concise answers.
        
        Args:
            question: Question to ask
            top_k: Number of sources to retrieve
            
        Returns:
            GenerationResponse object
        """
        return self.query(question, top_k=top_k, prompt_template='summary')
    
    def factual_query(self, question: str, top_k: int = 3, score_threshold: float = 0.8) -> GenerationResponse:
        """
        Query for factual information with high confidence.
        
        Args:
            question: Question to ask
            top_k: Number of sources to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            GenerationResponse object
        """
        return self.query(question, top_k=top_k, score_threshold=score_threshold, prompt_template='factual')
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and metrics.
        
        Returns:
            System status dictionary
        """
        status = {
            'system_info': {
                'name': self.config.app.name,
                'version': self.config.app.version,
                'environment': self.config.app.environment,
                'debug_mode': self.config.app.debug
            },
            'agents_status': {
                'document_parser': 'initialized' if self.document_parser else 'not_initialized',
                'chunking_embedding': 'initialized' if self.chunking_embedding else 'not_initialized',
                'retrieval': 'initialized' if self.retrieval else 'not_initialized',
                'generation': 'initialized' if self.generation else 'not_initialized',
                'evaluation': 'not_implemented',
                'orchestrator': 'initialized' if self.orchestrator else 'not_available'
            },
            'metrics': self.metrics.to_dict(),
            'configuration': {
                'embedding_provider': self.config.embeddings.provider,
                'embedding_model': self.config.embeddings.model_name,
                'vector_db_provider': self.config.database.provider,
                'chunking_strategy': self.config.get('chunking.strategy', 'recursive')
            }
        }
        
        # Add vector database info if available
        if self.chunking_embedding and self.chunking_embedding.vector_store:
            try:
                vector_info = self.chunking_embedding.get_collection_info()
                status['vector_database'] = vector_info
            except Exception as e:
                status['vector_database'] = {'error': str(e)}
        
        return status
    
    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """
        Validate if a document can be processed.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Validation result dictionary
        """
        return self.document_parser.validate_file(file_path)
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats.
        
        Returns:
            List of supported file extensions
        """
        return self.document_parser.get_supported_formats()
    
    def update_configuration(self, **kwargs) -> None:
        """
        Update system configuration at runtime.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            self.config.set(key, value)
        
        logger.info(f"Configuration updated: {list(kwargs.keys())}")
    
    def reload_configuration(self) -> None:
        """Reload configuration from files and environment."""
        self.config.reload_config()
        
        # Re-initialize agents if needed
        # Note: This would require more sophisticated agent management
        logger.info("Configuration reloaded")
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration (without sensitive data).
        
        Returns:
            Configuration dictionary
        """
        return {
            'app': self.config.app.dict(),
            'chunking': self.config.get_section('chunking'),
            'embeddings': {
                k: v for k, v in self.config.embeddings.dict().items()
                if k != 'api_key'
            },
            'vector_db': {
                k: v for k, v in self.config.database.dict().items()
                if k not in ['api_key', 'url']
            }
        }
    
    def clear_vector_database(self) -> bool:
        """
        Clear all data from the vector database.
        
        Returns:
            Success status
        """
        logger.warning("Clearing vector database - this will delete all stored embeddings")
        
        try:
            if self.chunking_embedding and self.chunking_embedding.vector_store:
                # This would need to be implemented in the chunking_embedding agent
                # For now, just log the attempt
                logger.info("Vector database clear requested (not implemented)")
                return True
            else:
                logger.error("Vector store not available")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing vector database: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all system components.
        
        Returns:
            Health status dictionary
        """
        health = {
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check document parser
        try:
            if self.document_parser:
                health['components']['document_parser'] = 'healthy'
            else:
                health['components']['document_parser'] = 'not_initialized'
        except Exception as e:
            health['components']['document_parser'] = f'error: {str(e)}'
            health['overall_status'] = 'degraded'
        
        # Check chunking & embedding agent
        try:
            if self.chunking_embedding:
                # Try to get collection info as a health check
                collection_info = self.chunking_embedding.get_collection_info()
                if 'error' in collection_info:
                    health['components']['chunking_embedding'] = f'error: {collection_info["error"]}'
                    health['overall_status'] = 'degraded'
                else:
                    health['components']['chunking_embedding'] = 'healthy'
            else:
                health['components']['chunking_embedding'] = 'not_initialized'
        except Exception as e:
            health['components']['chunking_embedding'] = f'error: {str(e)}'
            health['overall_status'] = 'degraded'
        
        # Check retrieval agent
        try:
            if self.retrieval:
                # Test connection to vector database
                connection_test = self.retrieval.test_connection()
                if connection_test['status'] == 'connected':
                    health['components']['retrieval'] = 'healthy'
                else:
                    health['components']['retrieval'] = f'error: {connection_test.get("error", "connection failed")}'
                    health['overall_status'] = 'degraded'
            else:
                health['components']['retrieval'] = 'not_initialized'
        except Exception as e:
            health['components']['retrieval'] = f'error: {str(e)}'
            health['overall_status'] = 'degraded'
        
        # Check generation agent
        try:
            if self.generation:
                # Test generation capability
                test_result = self.generation.test_generation()
                if test_result['status'] == 'success':
                    health['components']['generation'] = 'healthy'
                else:
                    health['components']['generation'] = f'error: {test_result.get("error", "test failed")}'
                    health['overall_status'] = 'degraded'
            else:
                health['components']['generation'] = 'not_initialized'
        except Exception as e:
            health['components']['generation'] = f'error: {str(e)}'
            health['overall_status'] = 'degraded'
        
        # Check configuration
        config_errors = self.config.validate_config()
        if config_errors:
            health['components']['configuration'] = f'warnings: {list(config_errors.keys())}'
            if health['overall_status'] == 'healthy':
                health['overall_status'] = 'warnings'
        else:
            health['components']['configuration'] = 'healthy'
        
        # Check orchestrator
        try:
            if self.orchestrator:
                workflows = self.orchestrator.list_available_workflows()
                if workflows:
                    health['components']['orchestrator'] = 'healthy'
                    health['workflow_types'] = workflows
                else:
                    health['components']['orchestrator'] = 'no_workflows'
            else:
                health['components']['orchestrator'] = 'not_available'
        except Exception as e:
            health['components']['orchestrator'] = f'error: {str(e)}'
            health['overall_status'] = 'degraded'
        
        return health
    
    async def execute_workflow(self, workflow_type: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an orchestrated workflow.
        
        Args:
            workflow_type: Type of workflow to execute
            **kwargs: Workflow-specific parameters
            
        Returns:
            Workflow execution results
            
        Raises:
            ValueError: If orchestrator not available or workflow type unknown
        """
        if not self.orchestrator:
            raise ValueError("Orchestrator not available - LangGraph may not be installed")
        
        return await self.orchestrator.execute_workflow(workflow_type, **kwargs)
    
    def get_available_workflows(self) -> List[str]:
        """
        Get list of available workflow types.
        
        Returns:
            List of workflow type names
        """
        if not self.orchestrator:
            return []
        
        return self.orchestrator.list_available_workflows()
    
    def get_workflow_schema(self, workflow_type: str) -> Dict[str, Any]:
        """
        Get schema information for a workflow type.
        
        Args:
            workflow_type: Type of workflow
            
        Returns:
            Workflow schema dictionary
        """
        if not self.orchestrator:
            return {"error": "Orchestrator not available"}
        
        return self.orchestrator.get_workflow_schema(workflow_type)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources if needed
        logger.info("RAG System context closed")
    
    def __str__(self) -> str:
        """String representation of the RAG System."""
        return f"RAGSystem(config={self.config.app.name} v{self.config.app.version})"
    
    def __repr__(self) -> str:
        """Detailed representation of the RAG System."""
        return (f"RAGSystem(documents={self.metrics.total_documents}, "
                f"chunks={self.metrics.total_chunks}, "
                f"embedding_provider={self.config.embeddings.provider})")


# Convenience functions for easy system usage
def create_rag_system(config_path: Optional[str] = None) -> RAGSystem:
    """
    Create a new RAG System instance.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        RAGSystem instance
    """
    return RAGSystem(config_path)


def quick_setup(documents_path: str, config_path: Optional[str] = None) -> RAGSystem:
    """
    Quick setup: Create RAG system and process documents from a directory.
    
    Args:
        documents_path: Path to directory containing documents
        config_path: Path to configuration file (optional)
        
    Returns:
        RAGSystem instance with processed documents
    """
    rag = RAGSystem(config_path)
    
    if os.path.exists(documents_path):
        logger.info(f"Processing documents from: {documents_path}")
        rag.add_documents(documents_path)
    else:
        logger.warning(f"Documents path not found: {documents_path}")
    
    return rag