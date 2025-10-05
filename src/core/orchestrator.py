"""
LangGraph Orchestration System for RAG Agent System

This module provides advanced workflow orchestration using LangGraph for 
complex agent interactions, conditional routing, and workflow management.
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from datetime import datetime
import json

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# Local imports
from ..models.document import Document, GenerationResponse, ProcessingStatus
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RAGWorkflowState(TypedDict):
    """State definition for RAG workflow."""
    # Input
    query: str
    documents: Optional[List[str]]  # Optional document paths to process
    workflow_type: str  # Type of workflow to execute
    
    # Processing state
    processed_documents: List[Document]
    retrieved_chunks: List[Any]
    generated_response: Optional[GenerationResponse]
    evaluation_results: Optional[Dict[str, Any]]
    
    # Control flow
    current_step: str
    error_message: Optional[str]
    retry_count: int
    
    # Metadata
    workflow_id: str
    start_time: datetime
    processing_times: Dict[str, float]
    intermediate_results: Dict[str, Any]


class RAGOrchestrator:
    """
    Advanced RAG system orchestrator using LangGraph for workflow management.
    """
    
    def __init__(self, rag_system, config: Dict[str, Any] = None):
        """
        Initialize the RAG Orchestrator.
        
        Args:
            rag_system: RAGSystem instance
            config: Configuration dictionary
        """
        self.rag_system = rag_system
        self.config = config or {}
        
        # Workflow configuration
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        self.enable_evaluation = self.config.get('enable_evaluation', True)
        
        # Initialize workflow graphs
        self.workflows = {}
        self._initialize_workflows()
        
        logger.info("RAG Orchestrator initialized with LangGraph")
    
    def _initialize_workflows(self):
        """Initialize different workflow types."""
        # Standard RAG workflow
        self.workflows['standard'] = self._create_standard_rag_workflow()
        
        # Document processing workflow
        self.workflows['document_processing'] = self._create_document_processing_workflow()
        
        # Evaluation workflow
        self.workflows['evaluation'] = self._create_evaluation_workflow()
        
        # Batch processing workflow
        self.workflows['batch_processing'] = self._create_batch_processing_workflow()
        
        # Interactive workflow (with human feedback)
        self.workflows['interactive'] = self._create_interactive_workflow()
    
    def _create_standard_rag_workflow(self) -> StateGraph:
        """Create the standard RAG workflow."""
        workflow = StateGraph(RAGWorkflowState)
        
        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("evaluate_response", self._evaluate_response)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define edges
        workflow.set_entry_point("validate_input")
        
        workflow.add_conditional_edges(
            "validate_input",
            self._should_continue_after_validation,
            {
                "continue": "retrieve_context",
                "error": "handle_error",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "retrieve_context",
            self._should_continue_after_retrieval,
            {
                "continue": "generate_response",
                "retry": "retrieve_context",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_response",
            self._should_continue_after_generation,
            {
                "evaluate": "evaluate_response",
                "complete": END,
                "retry": "generate_response",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("evaluate_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _create_document_processing_workflow(self) -> StateGraph:
        """Create workflow for document processing."""
        workflow = StateGraph(RAGWorkflowState)
        
        # Add nodes for document processing
        workflow.add_node("parse_documents", self._parse_documents)
        workflow.add_node("chunk_and_embed", self._chunk_and_embed)
        workflow.add_node("store_embeddings", self._store_embeddings)
        workflow.add_node("validate_storage", self._validate_storage)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define document processing flow
        workflow.set_entry_point("parse_documents")
        workflow.add_edge("parse_documents", "chunk_and_embed")
        workflow.add_edge("chunk_and_embed", "store_embeddings")
        workflow.add_edge("store_embeddings", "validate_storage")
        workflow.add_edge("validate_storage", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _create_evaluation_workflow(self) -> StateGraph:
        """Create workflow for response evaluation."""
        workflow = StateGraph(RAGWorkflowState)
        
        # Add evaluation nodes
        workflow.add_node("prepare_evaluation", self._prepare_evaluation)
        workflow.add_node("run_metrics", self._run_evaluation_metrics)
        workflow.add_node("analyze_results", self._analyze_evaluation_results)
        workflow.add_node("generate_report", self._generate_evaluation_report)
        
        # Define evaluation flow
        workflow.set_entry_point("prepare_evaluation")
        workflow.add_edge("prepare_evaluation", "run_metrics")
        workflow.add_edge("run_metrics", "analyze_results")
        workflow.add_edge("analyze_results", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def _create_batch_processing_workflow(self) -> StateGraph:
        """Create workflow for batch processing."""
        workflow = StateGraph(RAGWorkflowState)
        
        # Add batch processing nodes
        workflow.add_node("prepare_batch", self._prepare_batch)
        workflow.add_node("process_batch_items", self._process_batch_items)
        workflow.add_node("aggregate_results", self._aggregate_batch_results)
        workflow.add_node("generate_batch_report", self._generate_batch_report)
        
        # Define batch flow
        workflow.set_entry_point("prepare_batch")
        workflow.add_edge("prepare_batch", "process_batch_items")
        workflow.add_edge("process_batch_items", "aggregate_results")
        workflow.add_edge("aggregate_results", "generate_batch_report")
        workflow.add_edge("generate_batch_report", END)
        
        return workflow.compile()
    
    def _create_interactive_workflow(self) -> StateGraph:
        """Create interactive workflow with human feedback."""
        workflow = StateGraph(RAGWorkflowState)
        
        # Add interactive nodes
        workflow.add_node("initial_response", self._generate_initial_response)
        workflow.add_node("request_feedback", self._request_human_feedback)
        workflow.add_node("process_feedback", self._process_human_feedback)
        workflow.add_node("refine_response", self._refine_response)
        
        # Define interactive flow
        workflow.set_entry_point("initial_response")
        
        workflow.add_conditional_edges(
            "initial_response",
            self._should_request_feedback,
            {
                "request_feedback": "request_feedback",
                "complete": END
            }
        )
        
        workflow.add_edge("request_feedback", "process_feedback")
        workflow.add_edge("process_feedback", "refine_response")
        workflow.add_edge("refine_response", END)
        
        return workflow.compile()
    
    # Node implementations for standard RAG workflow
    async def _validate_input(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Validate input parameters."""
        logger.info("Validating workflow input")
        
        try:
            state["current_step"] = "validate_input"
            state["start_time"] = datetime.now()
            
            # Validate required fields
            if not state.get("query", "").strip():
                state["error_message"] = "Query is required but was empty"
                return state
            
            if not state.get("workflow_id"):
                state["workflow_id"] = f"workflow_{int(datetime.now().timestamp())}"
            
            # Initialize processing times
            if "processing_times" not in state:
                state["processing_times"] = {}
            
            logger.info(f"Input validation successful for workflow {state['workflow_id']}")
            return state
            
        except Exception as e:
            state["error_message"] = f"Input validation failed: {str(e)}"
            logger.error(f"Input validation error: {str(e)}")
            return state
    
    async def _retrieve_context(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Retrieve relevant context for the query."""
        logger.info("Retrieving context for query")
        
        try:
            import time
            start_time = time.time()
            
            state["current_step"] = "retrieve_context"
            
            query = state["query"]
            
            # Use retrieval agent to get relevant chunks
            if hasattr(self.rag_system, 'retrieval') and self.rag_system.retrieval:
                retrieval_results = self.rag_system.retrieval.retrieve(query)
                state["retrieved_chunks"] = retrieval_results
            else:
                state["error_message"] = "Retrieval agent not available"
                return state
            
            # Record processing time
            state["processing_times"]["retrieval"] = time.time() - start_time
            
            logger.info(f"Retrieved {len(retrieval_results)} relevant chunks")
            return state
            
        except Exception as e:
            state["error_message"] = f"Context retrieval failed: {str(e)}"
            logger.error(f"Context retrieval error: {str(e)}")
            return state
    
    async def _generate_response(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Generate response using the retrieved context."""
        logger.info("Generating response")
        
        try:
            import time
            start_time = time.time()
            
            state["current_step"] = "generate_response"
            
            query = state["query"]
            retrieval_results = state["retrieved_chunks"]
            
            # Use generation agent to create response
            if hasattr(self.rag_system, 'generation') and self.rag_system.generation:
                response = self.rag_system.generation.generate_response(
                    query=query,
                    retrieval_results=retrieval_results
                )
                state["generated_response"] = response
            else:
                state["error_message"] = "Generation agent not available"
                return state
            
            # Record processing time
            state["processing_times"]["generation"] = time.time() - start_time
            
            logger.info("Response generation completed")
            return state
            
        except Exception as e:
            state["error_message"] = f"Response generation failed: {str(e)}"
            logger.error(f"Response generation error: {str(e)}")
            return state
    
    async def _evaluate_response(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Evaluate the generated response."""
        logger.info("Evaluating response quality")
        
        try:
            import time
            start_time = time.time()
            
            state["current_step"] = "evaluate_response"
            
            response = state["generated_response"]
            
            # Use evaluation agent if available
            if hasattr(self.rag_system, 'evaluation') and self.rag_system.evaluation:
                metrics = self.rag_system.evaluation.evaluate_response(response)
                state["evaluation_results"] = metrics.to_dict()
            else:
                # Basic evaluation without evaluation agent
                state["evaluation_results"] = {
                    "response_length": len(response.answer),
                    "sources_count": len(response.sources),
                    "processing_time": response.processing_time
                }
            
            # Record processing time
            state["processing_times"]["evaluation"] = time.time() - start_time
            
            logger.info("Response evaluation completed")
            return state
            
        except Exception as e:
            state["error_message"] = f"Response evaluation failed: {str(e)}"
            logger.error(f"Response evaluation error: {str(e)}")
            return state
    
    async def _handle_error(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Handle errors that occur during workflow execution."""
        logger.error(f"Handling workflow error: {state.get('error_message')}")
        
        state["current_step"] = "handle_error"
        
        # Increment retry count
        state["retry_count"] = state.get("retry_count", 0) + 1
        
        # Log error details
        error_details = {
            "workflow_id": state.get("workflow_id"),
            "current_step": state.get("current_step"),
            "error_message": state.get("error_message"),
            "retry_count": state["retry_count"]
        }
        
        logger.error(f"Workflow error details: {json.dumps(error_details, indent=2)}")
        
        return state
    
    # Conditional edge functions
    def _should_continue_after_validation(self, state: RAGWorkflowState) -> str:
        """Determine next step after input validation."""
        if state.get("error_message"):
            return "error"
        return "continue"
    
    def _should_continue_after_retrieval(self, state: RAGWorkflowState) -> str:
        """Determine next step after context retrieval."""
        if state.get("error_message"):
            if state.get("retry_count", 0) < self.max_retries:
                return "retry"
            return "error"
        
        if not state.get("retrieved_chunks"):
            return "error"
        
        return "continue"
    
    def _should_continue_after_generation(self, state: RAGWorkflowState) -> str:
        """Determine next step after response generation."""
        if state.get("error_message"):
            if state.get("retry_count", 0) < self.max_retries:
                return "retry"
            return "error"
        
        if not state.get("generated_response"):
            return "error"
        
        # Decide whether to evaluate or complete
        if self.enable_evaluation:
            return "evaluate"
        else:
            return "complete"
    
    # Document processing nodes
    async def _parse_documents(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Parse documents in the workflow."""
        logger.info("Parsing documents")
        
        try:
            state["current_step"] = "parse_documents"
            
            document_paths = state.get("documents", [])
            processed_docs = []
            
            for doc_path in document_paths:
                doc = self.rag_system.add_document(doc_path)
                processed_docs.append(doc)
            
            state["processed_documents"] = processed_docs
            
            logger.info(f"Parsed {len(processed_docs)} documents")
            return state
            
        except Exception as e:
            state["error_message"] = f"Document parsing failed: {str(e)}"
            return state
    
    async def _chunk_and_embed(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Chunk and embed processed documents."""
        logger.info("Chunking and embedding documents")
        
        try:
            state["current_step"] = "chunk_and_embed"
            
            processed_docs = state["processed_documents"]
            
            for doc in processed_docs:
                if hasattr(self.rag_system, 'chunking_embedding'):
                    chunks = self.rag_system.chunking_embedding.process_document(doc)
                    state["intermediate_results"][f"chunks_{doc.id}"] = len(chunks)
            
            logger.info("Chunking and embedding completed")
            return state
            
        except Exception as e:
            state["error_message"] = f"Chunking and embedding failed: {str(e)}"
            return state
    
    async def _store_embeddings(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Store embeddings in vector database."""
        logger.info("Storing embeddings")
        
        try:
            state["current_step"] = "store_embeddings"
            
            # Embeddings are stored as part of the chunking process
            # This step can perform additional validation or operations
            
            logger.info("Embeddings stored successfully")
            return state
            
        except Exception as e:
            state["error_message"] = f"Embedding storage failed: {str(e)}"
            return state
    
    async def _validate_storage(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Validate that embeddings were stored correctly."""
        logger.info("Validating embedding storage")
        
        try:
            state["current_step"] = "validate_storage"
            
            # Perform validation checks
            if hasattr(self.rag_system, 'chunking_embedding'):
                collection_info = self.rag_system.chunking_embedding.get_collection_info()
                state["intermediate_results"]["collection_info"] = collection_info
            
            logger.info("Storage validation completed")
            return state
            
        except Exception as e:
            state["error_message"] = f"Storage validation failed: {str(e)}"
            return state
    
    # Additional workflow node implementations would go here...
    # (Placeholder implementations for other workflows)
    
    async def _prepare_evaluation(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Prepare for evaluation workflow."""
        state["current_step"] = "prepare_evaluation"
        return state
    
    async def _run_evaluation_metrics(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Run evaluation metrics."""
        state["current_step"] = "run_metrics"
        return state
    
    async def _analyze_evaluation_results(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Analyze evaluation results."""
        state["current_step"] = "analyze_results"
        return state
    
    async def _generate_evaluation_report(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Generate evaluation report."""
        state["current_step"] = "generate_report"
        return state
    
    # Public interface methods
    async def execute_workflow(self, 
                             workflow_type: str,
                             query: str = None,
                             documents: List[str] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Execute a specific workflow type.
        
        Args:
            workflow_type: Type of workflow to execute
            query: Query string (for query-based workflows)
            documents: List of document paths (for document workflows)
            **kwargs: Additional workflow parameters
            
        Returns:
            Workflow execution results
        """
        logger.info(f"Executing {workflow_type} workflow")
        
        if workflow_type not in self.workflows:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        # Initialize workflow state
        initial_state = RAGWorkflowState(
            query=query or "",
            documents=documents or [],
            workflow_type=workflow_type,
            processed_documents=[],
            retrieved_chunks=[],
            generated_response=None,
            evaluation_results=None,
            current_step="initialized",
            error_message=None,
            retry_count=0,
            workflow_id=f"{workflow_type}_{int(datetime.now().timestamp())}",
            start_time=datetime.now(),
            processing_times={},
            intermediate_results={}
        )
        
        # Update state with any additional parameters
        for key, value in kwargs.items():
            if key in RAGWorkflowState.__annotations__:
                initial_state[key] = value
        
        try:
            # Execute the workflow
            workflow = self.workflows[workflow_type]
            result = await workflow.ainvoke(initial_state)
            
            # Calculate total execution time
            if "start_time" in result:
                total_time = (datetime.now() - result["start_time"]).total_seconds()
                result["processing_times"]["total_workflow"] = total_time
            
            logger.info(f"Workflow {workflow_type} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_type} execution failed: {str(e)}")
            return {
                "workflow_type": workflow_type,
                "error_message": str(e),
                "status": "failed"
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the status of a running workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Workflow status information
        """
        # This would typically query a workflow state store
        # For now, return basic information
        return {
            "workflow_id": workflow_id,
            "status": "This feature would track running workflows"
        }
    
    def list_available_workflows(self) -> List[str]:
        """List all available workflow types."""
        return list(self.workflows.keys())
    
    def get_workflow_schema(self, workflow_type: str) -> Dict[str, Any]:
        """
        Get the schema for a specific workflow type.
        
        Args:
            workflow_type: Type of workflow
            
        Returns:
            Workflow schema information
        """
        if workflow_type not in self.workflows:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        schemas = {
            "standard": {
                "description": "Standard RAG query processing",
                "required_inputs": ["query"],
                "optional_inputs": ["top_k", "score_threshold", "prompt_template"],
                "outputs": ["generated_response", "evaluation_results"]
            },
            "document_processing": {
                "description": "Process and index documents",
                "required_inputs": ["documents"],
                "optional_inputs": ["chunk_size", "chunk_overlap"],
                "outputs": ["processed_documents", "intermediate_results"]
            },
            "evaluation": {
                "description": "Evaluate system responses",
                "required_inputs": ["generated_response"],
                "optional_inputs": ["ground_truth", "metrics"],
                "outputs": ["evaluation_results"]
            },
            "batch_processing": {
                "description": "Process multiple queries in batch",
                "required_inputs": ["queries"],
                "optional_inputs": ["batch_size", "parallel_processing"],
                "outputs": ["batch_results"]
            },
            "interactive": {
                "description": "Interactive workflow with human feedback",
                "required_inputs": ["query"],
                "optional_inputs": ["feedback_enabled", "max_iterations"],
                "outputs": ["refined_response", "feedback_history"]
            }
        }
        
        return schemas.get(workflow_type, {})
    
    # Placeholder implementations for additional workflow nodes
    async def _prepare_batch(self, state: RAGWorkflowState) -> RAGWorkflowState:
        state["current_step"] = "prepare_batch"
        return state
    
    async def _process_batch_items(self, state: RAGWorkflowState) -> RAGWorkflowState:
        state["current_step"] = "process_batch_items"
        return state
    
    async def _aggregate_batch_results(self, state: RAGWorkflowState) -> RAGWorkflowState:
        state["current_step"] = "aggregate_results"
        return state
    
    async def _generate_batch_report(self, state: RAGWorkflowState) -> RAGWorkflowState:
        state["current_step"] = "generate_batch_report"
        return state
    
    async def _generate_initial_response(self, state: RAGWorkflowState) -> RAGWorkflowState:
        state["current_step"] = "initial_response"
        return state
    
    async def _request_human_feedback(self, state: RAGWorkflowState) -> RAGWorkflowState:
        state["current_step"] = "request_feedback"
        return state
    
    async def _process_human_feedback(self, state: RAGWorkflowState) -> RAGWorkflowState:
        state["current_step"] = "process_feedback"
        return state
    
    async def _refine_response(self, state: RAGWorkflowState) -> RAGWorkflowState:
        state["current_step"] = "refine_response"
        return state
    
    def _should_request_feedback(self, state: RAGWorkflowState) -> str:
        """Determine if human feedback should be requested."""
        # This could be based on confidence scores, user preferences, etc.
        return "request_feedback" if state.get("request_feedback", False) else "complete"