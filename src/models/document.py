"""
Document data models for the RAG Agent System.

This module defines the data structures for representing documents,
text chunks, and system responses.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
import hashlib
import uuid


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Processing status for documents."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    INDEXED = "indexed"


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    creation_date: Optional[datetime] = Field(None, description="Creation date")
    modification_date: Optional[datetime] = Field(None, description="Last modification date")
    language: Optional[str] = Field("en", description="Document language")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages")
    word_count: Optional[int] = Field(None, description="Number of words")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Document keywords")
    tags: Optional[List[str]] = Field(default_factory=list, description="User-defined tags")
    subject: Optional[str] = Field(None, description="Document subject")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Document(BaseModel):
    """Document model representing a processed document."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document ID")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path to the document file")
    content: str = Field(..., description="Extracted text content")
    content_hash: str = Field("", description="Hash of the content for deduplication")
    document_type: DocumentType = Field(..., description="Type of document")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")
    processing_status: ProcessingStatus = Field(ProcessingStatus.PENDING, description="Processing status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    @validator('content_hash', always=True)
    def generate_content_hash(cls, v, values):
        """Generate content hash for deduplication."""
        if not v and 'content' in values:
            content = values['content']
            return hashlib.sha256(content.encode()).hexdigest()
        return v
    
    @validator('updated_at', always=True)
    def update_timestamp(cls, v):
        """Update timestamp on any change."""
        return datetime.utcnow()
    
    def update_status(self, status: ProcessingStatus, error_message: Optional[str] = None):
        """Update processing status."""
        self.processing_status = status
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
    
    def add_metadata(self, key: str, value: Any):
        """Add custom metadata."""
        self.metadata.extra[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get custom metadata value."""
        return self.metadata.extra.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict()
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TextChunk(BaseModel):
    """Text chunk model for storing processed text segments."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    content_hash: str = Field("", description="Hash of the chunk content")
    start_index: int = Field(..., description="Start position in original document")
    end_index: int = Field(..., description="End position in original document")
    chunk_index: int = Field(..., description="Sequential chunk number in document")
    chunk_size: int = Field(..., description="Size of the chunk in characters")
    overlap_size: int = Field(0, description="Overlap with previous chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk-specific metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    @validator('content_hash', always=True)
    def generate_content_hash(cls, v, values):
        """Generate content hash for the chunk."""
        if not v and 'content' in values:
            content = values['content']
            return hashlib.sha256(content.encode()).hexdigest()
        return v
    
    @validator('chunk_size', always=True)
    def calculate_chunk_size(cls, v, values):
        """Calculate chunk size if not provided."""
        if not v and 'content' in values:
            return len(values['content'])
        return v
    
    def set_embedding(self, embedding: List[float], model_name: str):
        """Set the embedding for this chunk."""
        self.embedding = embedding
        self.embedding_model = model_name
    
    def get_context_window(self, document_content: str, window_size: int = 100) -> str:
        """Get extended context around this chunk."""
        start = max(0, self.start_index - window_size)
        end = min(len(document_content), self.end_index + window_size)
        return document_content[start:end]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict(exclude={'embedding'})  # Exclude large embedding by default
    
    def to_dict_with_embedding(self) -> Dict[str, Any]:
        """Convert to dictionary with embedding included."""
        return self.dict()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RetrievalResult(BaseModel):
    """Result from retrieval operation."""
    
    chunk: TextChunk = Field(..., description="Retrieved text chunk")
    score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Rank in results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional retrieval metadata")
    
    def __lt__(self, other):
        """Enable sorting by score (descending)."""
        return self.score > other.score


class GenerationResponse(BaseModel):
    """Response from generation agent."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Response ID")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[RetrievalResult] = Field(default_factory=list, description="Source chunks used")
    context_used: str = Field("", description="Combined context sent to LLM")
    model_name: str = Field(..., description="Model used for generation")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    def add_source(self, chunk: TextChunk, score: float, rank: int):
        """Add a source chunk to the response."""
        result = RetrievalResult(chunk=chunk, score=score, rank=rank)
        self.sources.append(result)
    
    def get_source_documents(self) -> List[str]:
        """Get list of unique source document IDs."""
        return list(set(result.chunk.document_id for result in self.sources))
    
    def get_citations(self) -> str:
        """Generate citation text from sources."""
        citations = []
        for i, result in enumerate(self.sources, 1):
            chunk = result.chunk
            doc_id = chunk.document_id[:8]  # Shortened doc ID
            citations.append(f"[{i}] Doc {doc_id}, chunk {chunk.chunk_index}")
        return "; ".join(citations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for RAG responses."""
    
    response_id: str = Field(..., description="Response ID being evaluated")
    faithfulness_score: Optional[float] = Field(None, description="Faithfulness to context")
    relevance_score: Optional[float] = Field(None, description="Relevance to query")
    fluency_score: Optional[float] = Field(None, description="Language fluency")
    coherence_score: Optional[float] = Field(None, description="Response coherence")
    groundedness_score: Optional[float] = Field(None, description="How well grounded in sources")
    overall_score: Optional[float] = Field(None, description="Overall quality score")
    evaluator_model: Optional[str] = Field(None, description="Model used for evaluation")
    evaluation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp")
    
    @validator('overall_score', always=True)
    def calculate_overall_score(cls, v, values):
        """Calculate overall score from individual metrics."""
        if v is not None:
            return v
            
        scores = [
            values.get('faithfulness_score'),
            values.get('relevance_score'),
            values.get('fluency_score'),
            values.get('coherence_score'),
            values.get('groundedness_score')
        ]
        
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            return sum(valid_scores) / len(valid_scores)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemMetrics(BaseModel):
    """System-wide metrics and statistics."""
    
    total_documents: int = Field(0, description="Total number of documents")
    total_chunks: int = Field(0, description="Total number of chunks")
    total_queries: int = Field(0, description="Total number of queries processed")
    total_responses: int = Field(0, description="Total number of responses generated")
    avg_response_time: Optional[float] = Field(None, description="Average response time")
    avg_retrieval_score: Optional[float] = Field(None, description="Average retrieval score")
    avg_evaluation_score: Optional[float] = Field(None, description="Average evaluation score")
    system_uptime: Optional[float] = Field(None, description="System uptime in seconds")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    disk_usage: Optional[float] = Field(None, description="Disk usage in MB")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    def update_metrics(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }