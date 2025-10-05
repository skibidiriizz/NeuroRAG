"""
Retrieval Agent for RAG System

This agent handles semantic search over vector databases to retrieve relevant
text chunks for user queries. It supports multiple retrieval strategies and
reranking mechanisms.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

# Vector database imports
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, ScoredPoint
import chromadb

# Reranking imports
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from ..models.document import TextChunk, RetrievalResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalAgent:
    """
    Agent responsible for semantic search and retrieval of relevant text chunks.
    """
    
    def __init__(self, config: Dict[str, Any] = None, embedding_agent=None):
        """
        Initialize the Retrieval Agent.
        
        Args:
            config: Configuration dictionary
            embedding_agent: Reference to embedding agent for query encoding
        """
        self.config = config or {}
        self.embedding_agent = embedding_agent
        
        # Retrieval configuration
        self.retrieval_config = self.config.get('retrieval', {})
        self.top_k = self.retrieval_config.get('top_k', 5)
        self.score_threshold = self.retrieval_config.get('score_threshold', 0.7)
        self.reranking = self.retrieval_config.get('reranking', True)
        self.diversity_threshold = self.retrieval_config.get('diversity_threshold', 0.8)
        
        # Vector database configuration
        self.vector_db_config = self.config.get('vector_db', {})
        self.vector_db_provider = self.vector_db_config.get('provider', 'qdrant')
        self.collection_name = self.vector_db_config.get('collection_name', 'rag_documents')
        
        # Initialize components
        self.vector_store = None
        self.reranker = None
        
        self._initialize_components()
        
        logger.info(f"Retrieval Agent initialized with {self.vector_db_provider} backend")
    
    def _initialize_components(self):
        """Initialize vector store and reranker."""
        # Initialize vector store connection
        self.vector_store = self._create_vector_store()
        
        # Initialize reranker if enabled
        if self.reranking:
            self._initialize_reranker()
    
    def _create_vector_store(self):
        """Create vector store client."""
        if self.vector_db_provider == 'qdrant':
            return self._create_qdrant_client()
        elif self.vector_db_provider == 'chroma':
            return self._create_chroma_client()
        else:
            raise ValueError(f"Unsupported vector database provider: {self.vector_db_provider}")
    
    def _create_qdrant_client(self):
        """Create Qdrant client."""
        host = self.vector_db_config.get('host', 'localhost')
        port = self.vector_db_config.get('port', 6333)
        url = self.vector_db_config.get('url')
        api_key = self.vector_db_config.get('api_key')
        
        if url:
            client = QdrantClient(url=url, api_key=api_key)
        else:
            client = QdrantClient(host=host, port=port)
        
        return client
    
    def _create_chroma_client(self):
        """Create Chroma client."""
        persist_directory = self.vector_db_config.get('persist_directory', './data/chroma')
        client = chromadb.PersistentClient(path=persist_directory)
        return client
    
    def _initialize_reranker(self):
        """Initialize cross-encoder for reranking."""
        try:
            reranker_model = self.retrieval_config.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.reranker = CrossEncoder(reranker_model)
            logger.info(f"Reranker initialized: {reranker_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {str(e)}. Continuing without reranking.")
            self.reranking = False
            self.reranker = None
    
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                score_threshold: Optional[float] = None,
                filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of top results to return (optional)
            score_threshold: Minimum similarity score threshold (optional)
            filters: Additional filters for retrieval (optional)
            
        Returns:
            List of RetrievalResult objects
        """
        start_time = time.time()
        
        # Use provided parameters or defaults
        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold
        
        logger.info(f"Retrieving chunks for query: '{query[:50]}...' (top_k={top_k})")
        
        try:
            # Step 1: Get query embedding
            if not self.embedding_agent:
                raise ValueError("Embedding agent not provided - cannot encode query")
            
            query_embedding = self.embedding_agent.get_embedding_for_query(query)
            
            # Step 2: Perform vector search
            if self.vector_db_provider == 'qdrant':
                raw_results = self._search_qdrant(query_embedding, top_k * 2, filters)  # Get more for reranking
            elif self.vector_db_provider == 'chroma':
                raw_results = self._search_chroma(query_embedding, top_k * 2, filters)
            else:
                raise ValueError(f"Unsupported vector database: {self.vector_db_provider}")
            
            # Step 3: Convert to RetrievalResult objects
            results = self._process_raw_results(raw_results, query)
            
            # Step 4: Filter by score threshold
            filtered_results = [r for r in results if r.score >= score_threshold]
            
            # Step 5: Apply reranking if enabled
            if self.reranking and self.reranker and len(filtered_results) > 1:
                filtered_results = self._rerank_results(query, filtered_results)
            
            # Step 6: Apply diversity filtering
            if self.diversity_threshold < 1.0:
                filtered_results = self._apply_diversity_filter(filtered_results)
            
            # Step 7: Limit to top_k
            final_results = filtered_results[:top_k]
            
            # Step 8: Update rankings
            for i, result in enumerate(final_results):
                result.rank = i + 1
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(final_results)} chunks in {retrieval_time:.3f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def _search_qdrant(self, query_embedding: List[float], limit: int, 
                      filters: Optional[Dict[str, Any]] = None) -> List[ScoredPoint]:
        """Perform search in Qdrant."""
        search_params = {
            'collection_name': self.collection_name,
            'query_vector': query_embedding,
            'limit': limit,
            'with_payload': True,
            'score_threshold': 0.0  # We'll filter later
        }
        
        # Add filters if provided
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            
            if filter_conditions:
                search_params['query_filter'] = Filter(must=filter_conditions)
        
        try:
            results = self.vector_store.search(**search_params)
            logger.debug(f"Qdrant search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Qdrant search error: {str(e)}")
            raise
    
    def _search_chroma(self, query_embedding: List[float], limit: int,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform search in Chroma."""
        try:
            collection = self.vector_store.get_collection(self.collection_name)
            
            query_params = {
                'query_embeddings': [query_embedding],
                'n_results': limit,
                'include': ['metadatas', 'documents', 'distances']
            }
            
            # Add filters if provided
            if filters:
                query_params['where'] = filters
            
            results = collection.query(**query_params)
            logger.debug(f"Chroma search returned {len(results['ids'][0])} results")
            
            # Convert Chroma results to consistent format
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'payload': results['metadatas'][0][i],
                    'content': results['documents'][0][i]
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Chroma search error: {str(e)}")
            raise
    
    def _process_raw_results(self, raw_results: List[Any], query: str) -> List[RetrievalResult]:
        """Convert raw search results to RetrievalResult objects."""
        results = []
        
        for i, result in enumerate(raw_results):
            try:
                if self.vector_db_provider == 'qdrant':
                    chunk_data = self._extract_qdrant_chunk_data(result)
                else:  # chroma
                    chunk_data = self._extract_chroma_chunk_data(result)
                
                # Create TextChunk object
                chunk = TextChunk(
                    id=chunk_data['id'],
                    document_id=chunk_data['document_id'],
                    content=chunk_data['content'],
                    start_index=chunk_data['start_index'],
                    end_index=chunk_data['end_index'],
                    chunk_index=chunk_data['chunk_index'],
                    chunk_size=chunk_data['chunk_size'],
                    metadata=chunk_data.get('metadata', {}),
                    created_at=datetime.fromisoformat(chunk_data['created_at'])
                )
                
                # Create RetrievalResult
                retrieval_result = RetrievalResult(
                    chunk=chunk,
                    score=chunk_data['score'],
                    rank=i + 1,
                    metadata={
                        'query': query,
                        'retrieval_method': 'vector_search',
                        'database_provider': self.vector_db_provider
                    }
                )
                
                results.append(retrieval_result)
                
            except Exception as e:
                logger.warning(f"Error processing result {i}: {str(e)}")
                continue
        
        return results
    
    def _extract_qdrant_chunk_data(self, result: ScoredPoint) -> Dict[str, Any]:
        """Extract chunk data from Qdrant result."""
        return {
            'id': str(result.id),
            'score': float(result.score),
            'document_id': result.payload.get('document_id'),
            'content': result.payload.get('content'),
            'start_index': result.payload.get('start_index', 0),
            'end_index': result.payload.get('end_index', 0),
            'chunk_index': result.payload.get('chunk_index', 0),
            'chunk_size': result.payload.get('chunk_size', 0),
            'metadata': result.payload.get('metadata', {}),
            'created_at': result.payload.get('created_at')
        }
    
    def _extract_chroma_chunk_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract chunk data from Chroma result."""
        return {
            'id': result['id'],
            'score': result['score'],
            'document_id': result['payload'].get('document_id'),
            'content': result['content'],
            'start_index': result['payload'].get('start_index', 0),
            'end_index': result['payload'].get('end_index', 0),
            'chunk_index': result['payload'].get('chunk_index', 0),
            'chunk_size': result['payload'].get('chunk_size', 0),
            'metadata': {k: v for k, v in result['payload'].items() 
                        if k not in ['document_id', 'start_index', 'end_index', 'chunk_index', 'chunk_size']},
            'created_at': result['payload'].get('created_at')
        }
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using cross-encoder."""
        if not self.reranker or len(results) <= 1:
            return results
        
        logger.debug(f"Reranking {len(results)} results")
        
        try:
            # Prepare pairs for reranking
            pairs = [(query, result.chunk.content) for result in results]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update scores and resort
            for result, rerank_score in zip(results, rerank_scores):
                # Combine original score with rerank score (weighted average)
                original_weight = 0.3
                rerank_weight = 0.7
                result.score = (original_weight * result.score + 
                              rerank_weight * float(rerank_score))
                result.metadata['rerank_score'] = float(rerank_score)
                result.metadata['original_score'] = result.score
            
            # Sort by new combined score
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(f"Reranking completed")
            return results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}. Using original scores.")
            return results
    
    def _apply_diversity_filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply diversity filtering to reduce redundant results."""
        if len(results) <= 1:
            return results
        
        logger.debug(f"Applying diversity filter (threshold={self.diversity_threshold})")
        
        try:
            # Get embeddings for all chunks (if available) or use content similarity
            diverse_results = [results[0]]  # Always include top result
            
            for candidate in results[1:]:
                is_diverse = True
                
                for selected in diverse_results:
                    # Simple content-based diversity check
                    similarity = self._calculate_content_similarity(
                        candidate.chunk.content,
                        selected.chunk.content
                    )
                    
                    if similarity > self.diversity_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_results.append(candidate)
            
            logger.debug(f"Diversity filtering: {len(results)} -> {len(diverse_results)} results")
            return diverse_results
            
        except Exception as e:
            logger.warning(f"Diversity filtering failed: {str(e)}. Using all results.")
            return results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two text contents."""
        try:
            # Simple word overlap similarity
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def retrieve_by_document(self, document_id: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve chunks from a specific document.
        
        Args:
            document_id: ID of the document
            top_k: Number of chunks to return
            
        Returns:
            List of RetrievalResult objects
        """
        logger.info(f"Retrieving chunks from document: {document_id}")
        
        try:
            filters = {'document_id': document_id}
            
            if self.vector_db_provider == 'qdrant':
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                filter_condition = Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
                
                results = self.vector_store.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=top_k,
                    with_payload=True
                )
                
                # Convert to consistent format
                raw_results = []
                for point in results[0]:  # results[0] contains points, results[1] is next_page_offset
                    raw_results.append(point)
            
            elif self.vector_db_provider == 'chroma':
                collection = self.vector_store.get_collection(self.collection_name)
                
                results = collection.get(
                    where=filters,
                    include=['metadatas', 'documents'],
                    limit=top_k
                )
                
                # Convert to consistent format
                raw_results = []
                for i in range(len(results['ids'])):
                    raw_results.append({
                        'id': results['ids'][i],
                        'score': 1.0,  # No score for direct retrieval
                        'payload': results['metadatas'][i],
                        'content': results['documents'][i]
                    })
            
            # Process results
            retrieval_results = self._process_raw_results(raw_results, f"document:{document_id}")
            
            # Sort by chunk index
            retrieval_results.sort(key=lambda x: x.chunk.chunk_index)
            
            # Update rankings
            for i, result in enumerate(retrieval_results):
                result.rank = i + 1
            
            logger.info(f"Retrieved {len(retrieval_results)} chunks from document {document_id}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks from document {document_id}: {str(e)}")
            raise
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics and configuration.
        
        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                'provider': self.vector_db_provider,
                'collection_name': self.collection_name,
                'top_k': self.top_k,
                'score_threshold': self.score_threshold,
                'reranking_enabled': self.reranking,
                'diversity_threshold': self.diversity_threshold,
                'reranker_model': self.retrieval_config.get('reranker_model') if self.reranking else None
            }
            
            # Add collection info if available
            if self.vector_db_provider == 'qdrant':
                try:
                    collection_info = self.vector_store.get_collection(self.collection_name)
                    stats['total_vectors'] = collection_info.vectors_count
                    stats['indexed_vectors'] = collection_info.indexed_vectors_count
                except Exception as e:
                    stats['collection_error'] = str(e)
            
            elif self.vector_db_provider == 'chroma':
                try:
                    collection = self.vector_store.get_collection(self.collection_name)
                    stats['total_documents'] = collection.count()
                except Exception as e:
                    stats['collection_error'] = str(e)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {str(e)}")
            return {'error': str(e)}
    
    def update_configuration(self, **kwargs):
        """
        Update retrieval configuration at runtime.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if key == 'top_k':
                self.top_k = value
            elif key == 'score_threshold':
                self.score_threshold = value
            elif key == 'reranking':
                self.reranking = value
                if value and not self.reranker:
                    self._initialize_reranker()
            elif key == 'diversity_threshold':
                self.diversity_threshold = value
        
        logger.info(f"Retrieval configuration updated: {list(kwargs.keys())}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the vector database.
        
        Returns:
            Connection test results
        """
        try:
            if self.vector_db_provider == 'qdrant':
                # Test Qdrant connection
                collections = self.vector_store.get_collections()
                collection_exists = any(c.name == self.collection_name for c in collections.collections)
                
                return {
                    'status': 'connected',
                    'provider': 'qdrant',
                    'collection_exists': collection_exists,
                    'collections_count': len(collections.collections)
                }
            
            elif self.vector_db_provider == 'chroma':
                # Test Chroma connection
                collections = self.vector_store.list_collections()
                collection_exists = any(c.name == self.collection_name for c in collections)
                
                return {
                    'status': 'connected',
                    'provider': 'chroma',
                    'collection_exists': collection_exists,
                    'collections_count': len(collections)
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'provider': self.vector_db_provider,
                'error': str(e)
            }