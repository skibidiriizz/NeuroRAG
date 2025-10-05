"""
Chunking & Embedding Agent for RAG System

This agent handles text chunking using various strategies and generates embeddings
using different models. It also manages vector database storage.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

# LangChain imports
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SpacyTextSplitter,
    TokenTextSplitter
)
from langchain_community.vectorstores import Qdrant, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Vector database imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb

# Embedding models
from sentence_transformers import SentenceTransformer
import torch

# Local imports
from ..models.document import Document, TextChunk, ProcessingStatus
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChunkingEmbeddingAgent:
    """
    Agent responsible for text chunking and embedding generation with vector database storage.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Chunking & Embedding Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Chunking configuration
        self.chunking_config = self.config.get('chunking', {})
        self.chunk_size = self.chunking_config.get('chunk_size', 1000)
        self.chunk_overlap = self.chunking_config.get('chunk_overlap', 200)
        self.chunking_strategy = self.chunking_config.get('strategy', 'recursive')
        
        # Embedding configuration
        self.embedding_config = self.config.get('embeddings', {})
        self.embedding_provider = self.embedding_config.get('provider', 'sentence_transformers')
        self.embedding_model_name = self.embedding_config.get('model_name', 'all-MiniLM-L6-v2')
        self.embedding_dimension = self.embedding_config.get('dimension', 384)
        
        # Vector database configuration
        self.vector_db_config = self.config.get('vector_db', {})
        self.vector_db_provider = self.vector_db_config.get('provider', 'qdrant')
        self.collection_name = self.vector_db_config.get('collection_name', 'rag_documents')
        
        # Initialize components
        self.text_splitter = None
        self.embedding_model = None
        self.vector_store = None
        
        self._initialize_components()
        
        logger.info(f"Chunking & Embedding Agent initialized with {self.chunking_strategy} chunking and {self.embedding_provider} embeddings")
    
    def _initialize_components(self):
        """Initialize text splitter, embedding model, and vector store."""
        # Initialize text splitter
        self.text_splitter = self._create_text_splitter()
        
        # Initialize embedding model
        self.embedding_model = self._create_embedding_model()
        
        # Initialize vector store
        self.vector_store = self._create_vector_store()
    
    def _create_text_splitter(self):
        """Create text splitter based on configuration."""
        separators = self.chunking_config.get('separators', ["\\n\\n", "\\n", " ", ""])
        
        if self.chunking_strategy == 'recursive':
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == 'character':
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\\n\\n",
                length_function=len,
            )
        elif self.chunking_strategy == 'token':
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base",  # GPT-4 tokenizer
            )
        elif self.chunking_strategy == 'spacy':
            try:
                return SpacyTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    pipeline="en_core_web_sm"
                )
            except OSError:
                logger.warning("SpaCy model not found, falling back to recursive splitter")
                return RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=separators,
                )
        else:
            logger.warning(f"Unknown chunking strategy: {self.chunking_strategy}, using recursive")
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
            )
    
    def _create_embedding_model(self):
        """Create embedding model based on configuration."""
        if self.embedding_provider == 'openai':
            api_key = self.embedding_config.get('api_key')
            return OpenAIEmbeddings(
                openai_api_key=api_key,
                model=self.embedding_model_name,
                show_progress_bar=True
            )
        elif self.embedding_provider == 'huggingface':
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': self.embedding_config.get('normalize', True)}
            )
        elif self.embedding_provider == 'sentence_transformers':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer(self.embedding_model_name, device=device)
            return model
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
    
    def _create_vector_store(self):
        """Create vector store based on configuration."""
        if self.vector_db_provider == 'qdrant':
            return self._create_qdrant_client()
        elif self.vector_db_provider == 'chroma':
            return self._create_chroma_client()
        else:
            raise ValueError(f"Unsupported vector database provider: {self.vector_db_provider}")
    
    def _create_qdrant_client(self):
        """Create Qdrant client and collection."""
        host = self.vector_db_config.get('host', 'localhost')
        port = self.vector_db_config.get('port', 6333)
        url = self.vector_db_config.get('url')
        api_key = self.vector_db_config.get('api_key')
        
        if url:
            client = QdrantClient(url=url, api_key=api_key)
        else:
            client = QdrantClient(host=host, port=port)
        
        # Create collection if it doesn't exist
        try:
            collections = client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                distance_metric = self.vector_db_config.get('distance_metric', 'cosine')
                distance_map = {
                    'cosine': Distance.COSINE,
                    'euclidean': Distance.EUCLID,
                    'dot': Distance.DOT
                }
                
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=distance_map.get(distance_metric, Distance.COSINE)
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up Qdrant collection: {str(e)}")
            raise
        
        return client
    
    def _create_chroma_client(self):
        """Create Chroma client and collection."""
        persist_directory = self.vector_db_config.get('persist_directory', './data/chroma')
        
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        try:
            collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Default to cosine similarity
            )
            logger.info(f"Using Chroma collection: {self.collection_name}")
            return client
            
        except Exception as e:
            logger.error(f"Error setting up Chroma collection: {str(e)}")
            raise
    
    def chunk_document(self, document: Document) -> List[TextChunk]:
        """
        Chunk a document into smaller text segments.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of TextChunk objects
        """
        if not document.content:
            logger.warning(f"Document {document.id} has no content to chunk")
            return []
        
        logger.info(f"Chunking document: {document.filename} ({len(document.content)} characters)")
        
        try:
            # Split text using configured splitter
            texts = self.text_splitter.split_text(document.content)
            
            chunks = []
            current_position = 0
            
            for i, chunk_text in enumerate(texts):
                # Find the position of this chunk in the original text
                chunk_start = document.content.find(chunk_text, current_position)
                if chunk_start == -1:
                    # Fallback if exact match not found
                    chunk_start = current_position
                
                chunk_end = chunk_start + len(chunk_text)
                
                # Calculate overlap with previous chunk
                overlap_size = 0
                if i > 0 and chunk_start < current_position:
                    overlap_size = current_position - chunk_start
                
                # Create TextChunk object
                chunk = TextChunk(
                    document_id=document.id,
                    content=chunk_text,
                    start_index=chunk_start,
                    end_index=chunk_end,
                    chunk_index=i,
                    chunk_size=len(chunk_text),
                    overlap_size=overlap_size,
                    metadata={
                        'document_filename': document.filename,
                        'document_type': document.document_type.value,
                        'chunk_strategy': self.chunking_strategy
                    }
                )
                
                chunks.append(chunk)
                current_position = chunk_end
            
            logger.info(f"Created {len(chunks)} chunks from document {document.filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {document.filename}: {str(e)}")
            raise
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of TextChunk objects with embeddings
        """
        if not chunks:
            return chunks
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.embedding_provider}")
        
        try:
            # Extract texts for embedding
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings based on provider
            if self.embedding_provider == 'sentence_transformers':
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=True,
                    normalize_embeddings=self.embedding_config.get('normalize', True)
                ).tolist()
            else:
                # For LangChain embeddings (OpenAI, HuggingFace)
                embeddings = self.embedding_model.embed_documents(texts)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.set_embedding(embedding, self.embedding_model_name)
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def store_in_vector_db(self, chunks: List[TextChunk]) -> bool:
        """
        Store chunks with embeddings in vector database.
        
        Args:
            chunks: List of TextChunk objects with embeddings
            
        Returns:
            Success status
        """
        if not chunks:
            return True
        
        # Filter chunks that have embeddings
        chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding is not None]
        
        if not chunks_with_embeddings:
            logger.warning("No chunks with embeddings to store")
            return False
        
        logger.info(f"Storing {len(chunks_with_embeddings)} chunks in {self.vector_db_provider} database")
        
        try:
            if self.vector_db_provider == 'qdrant':
                return self._store_in_qdrant(chunks_with_embeddings)
            elif self.vector_db_provider == 'chroma':
                return self._store_in_chroma(chunks_with_embeddings)
            else:
                raise ValueError(f"Unsupported vector database: {self.vector_db_provider}")
                
        except Exception as e:
            logger.error(f"Error storing chunks in vector database: {str(e)}")
            return False
    
    def _store_in_qdrant(self, chunks: List[TextChunk]) -> bool:
        """Store chunks in Qdrant."""
        points = []
        
        for chunk in chunks:
            payload = {
                'document_id': chunk.document_id,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'chunk_size': chunk.chunk_size,
                'metadata': chunk.metadata,
                'created_at': chunk.created_at.isoformat()
            }
            
            point = PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload=payload
            )
            points.append(point)
        
        # Batch upload points
        self.vector_store.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Successfully stored {len(points)} points in Qdrant")
        return True
    
    def _store_in_chroma(self, chunks: List[TextChunk]) -> bool:
        """Store chunks in Chroma."""
        collection = self.vector_store.get_collection(self.collection_name)
        
        # Prepare data for Chroma
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {
                'document_id': chunk.document_id,
                'chunk_index': chunk.chunk_index,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'chunk_size': chunk.chunk_size,
                'created_at': chunk.created_at.isoformat()
            }
            # Add custom metadata
            metadata.update(chunk.metadata)
            metadatas.append(metadata)
        
        # Upsert documents
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully stored {len(ids)} documents in Chroma")
        return True
    
    def process_document(self, document: Document) -> List[TextChunk]:
        """
        Complete processing pipeline: chunk, embed, and store document.
        
        Args:
            document: Document to process
            
        Returns:
            List of processed TextChunk objects
        """
        logger.info(f"Processing document: {document.filename}")
        
        try:
            # Update document status
            document.update_status(ProcessingStatus.PROCESSING)
            
            # Step 1: Chunk the document
            chunks = self.chunk_document(document)
            if not chunks:
                document.update_status(ProcessingStatus.FAILED, "No chunks created")
                return []
            
            # Step 2: Generate embeddings
            chunks_with_embeddings = self.generate_embeddings(chunks)
            
            # Step 3: Store in vector database
            stored_successfully = self.store_in_vector_db(chunks_with_embeddings)
            
            if stored_successfully:
                document.update_status(ProcessingStatus.INDEXED)
                logger.info(f"Successfully processed document {document.filename}: {len(chunks)} chunks created and stored")
            else:
                document.update_status(ProcessingStatus.FAILED, "Failed to store in vector database")
            
            return chunks_with_embeddings
            
        except Exception as e:
            error_msg = f"Error processing document {document.filename}: {str(e)}"
            logger.error(error_msg)
            document.update_status(ProcessingStatus.FAILED, str(e))
            raise
    
    def process_batch(self, documents: List[Document]) -> Dict[str, List[TextChunk]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Dictionary mapping document IDs to their chunks
        """
        logger.info(f"Processing batch of {len(documents)} documents")
        
        results = {}
        successful = 0
        
        for document in documents:
            try:
                chunks = self.process_document(document)
                results[document.id] = chunks
                if document.processing_status == ProcessingStatus.INDEXED:
                    successful += 1
            except Exception as e:
                logger.error(f"Failed to process document {document.filename}: {str(e)}")
                results[document.id] = []
        
        logger.info(f"Batch processing completed: {successful}/{len(documents)} documents successful")
        return results
    
    def get_embedding_for_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        try:
            if self.embedding_provider == 'sentence_transformers':
                embedding = self.embedding_model.encode(
                    [query],
                    normalize_embeddings=self.embedding_config.get('normalize', True)
                )[0].tolist()
            else:
                # For LangChain embeddings
                embedding = self.embedding_model.embed_query(query)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def update_chunk(self, chunk: TextChunk) -> bool:
        """
        Update an existing chunk in the vector database.
        
        Args:
            chunk: Updated chunk
            
        Returns:
            Success status
        """
        try:
            if self.vector_db_provider == 'qdrant':
                payload = {
                    'document_id': chunk.document_id,
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'chunk_size': chunk.chunk_size,
                    'metadata': chunk.metadata,
                    'created_at': chunk.created_at.isoformat()
                }
                
                point = PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload=payload
                )
                
                self.vector_store.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
            
            elif self.vector_db_provider == 'chroma':
                collection = self.vector_store.get_collection(self.collection_name)
                
                metadata = {
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'chunk_size': chunk.chunk_size,
                    'created_at': chunk.created_at.isoformat()
                }
                metadata.update(chunk.metadata)
                
                collection.upsert(
                    ids=[chunk.id],
                    embeddings=[chunk.embedding],
                    documents=[chunk.content],
                    metadatas=[metadata]
                )
            
            logger.info(f"Successfully updated chunk {chunk.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating chunk {chunk.id}: {str(e)}")
            return False
    
    def delete_document_chunks(self, document_id: str) -> bool:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Success status
        """
        try:
            if self.vector_db_provider == 'qdrant':
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
                
                self.vector_store.delete(
                    collection_name=self.collection_name,
                    points_selector=filter_condition
                )
            
            elif self.vector_db_provider == 'chroma':
                collection = self.vector_store.get_collection(self.collection_name)
                
                # Get all chunk IDs for this document
                results = collection.get(
                    where={"document_id": document_id},
                    include=["metadatas"]
                )
                
                if results['ids']:
                    collection.delete(ids=results['ids'])
            
            logger.info(f"Successfully deleted chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector collection.
        
        Returns:
            Collection information dictionary
        """
        try:
            if self.vector_db_provider == 'qdrant':
                collection_info = self.vector_store.get_collection(self.collection_name)
                return {
                    'provider': 'qdrant',
                    'collection_name': self.collection_name,
                    'vectors_count': collection_info.vectors_count,
                    'indexed_vectors_count': collection_info.indexed_vectors_count,
                    'points_count': collection_info.points_count,
                    'segments_count': collection_info.segments_count,
                    'vector_size': collection_info.config.params.vectors.size,
                    'distance': collection_info.config.params.vectors.distance.value
                }
            
            elif self.vector_db_provider == 'chroma':
                collection = self.vector_store.get_collection(self.collection_name)
                count = collection.count()
                return {
                    'provider': 'chroma',
                    'collection_name': self.collection_name,
                    'documents_count': count,
                    'embedding_dimension': self.embedding_dimension
                }
                
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {'error': str(e)}