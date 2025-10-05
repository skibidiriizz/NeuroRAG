"""
Generation Agent for RAG System

This agent handles LLM-powered response generation using retrieved context.
It supports various LLM providers and includes prompt engineering for grounded responses.
"""

import time
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# LangChain imports
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.callbacks import get_openai_callback

# Transformers for local models
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Local imports
from ..models.document import RetrievalResult, GenerationResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GenerationAgent:
    """
    Agent responsible for generating grounded responses using LLMs and retrieved context.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Generation Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # LLM configuration
        self.llm_config = self.config.get('llm', {})
        self.provider = self.llm_config.get('provider', 'openai')
        self.model_name = self.llm_config.get('model_name', 'gpt-3.5-turbo')
        self.temperature = self.llm_config.get('temperature', 0.1)
        self.max_tokens = self.llm_config.get('max_tokens', 1500)
        self.top_p = self.llm_config.get('top_p', 0.9)
        
        # Generation configuration
        self.generation_config = self.config.get('generation', {})
        self.max_context_length = self.generation_config.get('max_context_length', 4000)
        self.prompt_template_name = self.generation_config.get('prompt_template', 'default')
        self.include_sources = self.generation_config.get('include_sources', True)
        
        # Initialize components
        self.llm = None
        self.prompt_templates = {}
        
        self._initialize_components()
        
        logger.info(f"Generation Agent initialized with {self.provider} provider ({self.model_name})")
    
    def _initialize_components(self):
        """Initialize LLM and prompt templates."""
        # Initialize LLM
        self.llm = self._create_llm()
        
        # Initialize prompt templates
        self._initialize_prompt_templates()
    
    def _create_llm(self):
        """Create LLM instance based on configuration."""
        if self.provider == 'openai':
            api_key = self.llm_config.get('api_key')
            
            if self.model_name.startswith('gpt-'):
                # Use ChatOpenAI for GPT models
                return ChatOpenAI(
                    openai_api_key=api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    model_kwargs={'top_p': self.top_p}
                )
            else:
                # Use OpenAI for completion models
                return OpenAI(
                    openai_api_key=api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )
        
        elif self.provider == 'huggingface_local':
            return self._create_local_llm()
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _create_local_llm(self):
        """Create local HuggingFace LLM."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available for local LLM")
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=self.model_name,
                device=device,
                max_length=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=50256  # Common pad token ID
            )
            
            # Wrap in LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
            
            logger.info(f"Local LLM loaded: {self.model_name} on {device}")
            return llm
            
        except Exception as e:
            logger.error(f"Error loading local LLM: {str(e)}")
            raise
    
    def _initialize_prompt_templates(self):
        """Initialize prompt templates for different use cases."""
        
        # Default RAG prompt template
        self.prompt_templates['default'] = PromptTemplate(
            template="""You are a helpful AI assistant that provides accurate and informative answers based on the given context.

Context Information:
{context}

Question: {question}

Instructions:
- Answer the question using only the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Provide specific details and examples when available
- Keep your answer concise but comprehensive
- If you reference specific information, indicate which source it came from

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Chat-style prompt template
        self.prompt_templates['chat'] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant that answers questions based on provided context. 
Always base your answers on the given context and cite sources when possible."""),
            HumanMessage(content="""Context:
{context}

Question: {question}""")
        ])
        
        # Detailed analysis prompt
        self.prompt_templates['detailed'] = PromptTemplate(
            template="""You are an expert analyst providing detailed responses based on the given context.

Relevant Information:
{context}

User Query: {question}

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. Uses specific information from the provided context
3. Explains the reasoning behind your conclusions
4. Mentions any limitations or uncertainties
5. Cites the sources of information used

Detailed Response:""",
            input_variables=["context", "question"]
        )
        
        # Summary prompt template
        self.prompt_templates['summary'] = PromptTemplate(
            template="""Summarize the key information from the following context that relates to the user's question.

Context:
{context}

Question: {question}

Provide a concise summary that captures the most relevant points:""",
            input_variables=["context", "question"]
        )
        
        # Factual QA prompt
        self.prompt_templates['factual'] = PromptTemplate(
            template="""Answer the following question using only factual information from the provided context.

Context:
{context}

Question: {question}

Requirements:
- Provide only factual, verifiable information
- If uncertain, state your level of confidence
- Cite specific sources when mentioning facts
- If the context doesn't contain the answer, state that clearly

Factual Answer:""",
            input_variables=["context", "question"]
        )
    
    def generate_response(self, query: str, retrieval_results: List[RetrievalResult],
                         prompt_template: Optional[str] = None,
                         max_context_length: Optional[int] = None) -> GenerationResponse:
        """
        Generate a response using LLM and retrieved context.
        
        Args:
            query: User query
            retrieval_results: List of retrieved chunks
            prompt_template: Prompt template to use (optional)
            max_context_length: Maximum context length (optional)
            
        Returns:
            GenerationResponse object
        """
        start_time = time.time()
        
        # Use provided parameters or defaults
        template_name = prompt_template or self.prompt_template_name
        max_context = max_context_length or self.max_context_length
        
        logger.info(f"Generating response for query: '{query[:50]}...' using {len(retrieval_results)} sources")
        
        try:
            # Step 1: Prepare context from retrieval results
            context = self._prepare_context(retrieval_results, max_context)
            
            # Step 2: Select and format prompt
            prompt = self._format_prompt(query, context, template_name)
            
            # Step 3: Generate response with LLM
            if self.provider == 'openai' and self.model_name.startswith('gpt-'):
                # Use chat completion for GPT models
                with get_openai_callback() as cb:
                    response_text = self._generate_chat_response(prompt, query, context)
                    token_usage = {
                        'prompt_tokens': cb.prompt_tokens,
                        'completion_tokens': cb.completion_tokens,
                        'total_tokens': cb.total_tokens,
                        'total_cost': cb.total_cost
                    }
            else:
                # Use completion for other models
                response_text = self._generate_completion_response(prompt)
                token_usage = {}
            
            # Step 4: Post-process response
            cleaned_response = self._post_process_response(response_text)
            
            # Step 5: Create GenerationResponse object
            processing_time = time.time() - start_time
            
            generation_response = GenerationResponse(
                query=query,
                answer=cleaned_response,
                sources=retrieval_results,
                context_used=context,
                model_name=self.model_name,
                generation_metadata={
                    'prompt_template': template_name,
                    'context_length': len(context),
                    'num_sources': len(retrieval_results),
                    'provider': self.provider,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'token_usage': token_usage
                },
                processing_time=processing_time
            )
            
            logger.info(f"Generated response in {processing_time:.3f}s ({len(cleaned_response)} characters)")
            return generation_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Return error response
            return GenerationResponse(
                query=query,
                answer=f"I apologize, but I encountered an error while generating the response: {str(e)}",
                sources=[],
                context_used="",
                model_name=self.model_name,
                generation_metadata={'error': str(e)},
                processing_time=time.time() - start_time
            )
    
    def _prepare_context(self, retrieval_results: List[RetrievalResult], 
                        max_length: int) -> str:
        """Prepare context string from retrieval results."""
        if not retrieval_results:
            return "No relevant information found."
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(retrieval_results):
            # Create source reference
            source_ref = f"Source {i + 1}"
            if result.chunk.metadata.get('document_filename'):
                source_ref += f" ({result.chunk.metadata['document_filename']})"
            
            # Format chunk content with source
            chunk_text = f"{source_ref}:\n{result.chunk.content}\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_length and context_parts:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n".join(context_parts)
        
        # Truncate if still too long
        if len(context) > max_length:
            context = context[:max_length] + "... [truncated]"
        
        logger.debug(f"Prepared context: {len(context)} characters from {len(context_parts)} sources")
        return context
    
    def _format_prompt(self, query: str, context: str, template_name: str) -> str:
        """Format prompt using selected template."""
        if template_name not in self.prompt_templates:
            logger.warning(f"Template '{template_name}' not found, using 'default'")
            template_name = 'default'
        
        template = self.prompt_templates[template_name]
        
        if isinstance(template, ChatPromptTemplate):
            # For chat templates, return the formatted messages
            messages = template.format_messages(context=context, question=query)
            return messages
        else:
            # For regular templates, return formatted string
            return template.format(context=context, question=query)
    
    def _generate_chat_response(self, prompt, query: str, context: str) -> str:
        """Generate response using chat completion."""
        try:
            if isinstance(prompt, str):
                # Convert string prompt to messages
                messages = [
                    SystemMessage(content="You are a helpful AI assistant that provides accurate answers based on given context."),
                    HumanMessage(content=prompt)
                ]
            else:
                messages = prompt
            
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    def _generate_completion_response(self, prompt: str) -> str:
        """Generate response using text completion."""
        try:
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error in text completion: {str(e)}")
            raise
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response."""
        if not response:
            return "I apologize, but I couldn't generate a response."
        
        # Clean up the response
        cleaned = response.strip()
        
        # Remove any repeated newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove any trailing incomplete sentences if the response was truncated
        if cleaned.endswith('...') or cleaned.endswith('..'):
            # Find the last complete sentence
            sentences = re.split(r'[.!?]', cleaned)
            if len(sentences) > 1:
                cleaned = '.'.join(sentences[:-1]) + '.'
        
        return cleaned
    
    def generate_streaming_response(self, query: str, retrieval_results: List[RetrievalResult],
                                  prompt_template: Optional[str] = None) -> GenerationResponse:
        """
        Generate a streaming response (placeholder for future implementation).
        
        Args:
            query: User query
            retrieval_results: List of retrieved chunks
            prompt_template: Prompt template to use (optional)
            
        Returns:
            GenerationResponse object
        """
        # For now, fall back to regular generation
        # In the future, this could implement streaming for real-time response generation
        logger.info("Streaming not yet implemented, using regular generation")
        return self.generate_response(query, retrieval_results, prompt_template)
    
    def generate_with_citations(self, query: str, retrieval_results: List[RetrievalResult],
                              prompt_template: Optional[str] = None) -> GenerationResponse:
        """
        Generate response with explicit source citations.
        
        Args:
            query: User query
            retrieval_results: List of retrieved chunks
            prompt_template: Prompt template to use (optional)
            
        Returns:
            GenerationResponse object with citations
        """
        # Use detailed template for better citations
        template_name = prompt_template or 'detailed'
        
        response = self.generate_response(query, retrieval_results, template_name)
        
        # Add explicit citations at the end
        if self.include_sources and retrieval_results:
            citations = "\n\nSources:\n"
            for i, result in enumerate(retrieval_results, 1):
                filename = result.chunk.metadata.get('document_filename', 'Unknown Document')
                citations += f"{i}. {filename} (Score: {result.score:.3f})\n"
            
            response.answer += citations
        
        return response
    
    def get_available_templates(self) -> List[str]:
        """Get list of available prompt templates."""
        return list(self.prompt_templates.keys())
    
    def add_custom_template(self, name: str, template: str, input_variables: List[str]):
        """
        Add a custom prompt template.
        
        Args:
            name: Template name
            template: Template string
            input_variables: List of input variable names
        """
        try:
            self.prompt_templates[name] = PromptTemplate(
                template=template,
                input_variables=input_variables
            )
            logger.info(f"Added custom template: {name}")
        except Exception as e:
            logger.error(f"Error adding custom template {name}: {str(e)}")
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a given text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token on average
        return len(text) // 4
    
    def validate_context_length(self, context: str, query: str) -> bool:
        """
        Validate if context and query fit within model limits.
        
        Args:
            context: Context text
            query: Query text
            
        Returns:
            True if within limits, False otherwise
        """
        total_tokens = self.estimate_tokens(context + query)
        
        # Leave room for response generation
        max_input_tokens = self.max_tokens * 0.7  # Use 70% for input, 30% for output
        
        return total_tokens <= max_input_tokens
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get generation statistics and configuration.
        
        Returns:
            Statistics dictionary
        """
        return {
            'provider': self.provider,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_context_length': self.max_context_length,
            'available_templates': list(self.prompt_templates.keys()),
            'current_template': self.prompt_template_name,
            'include_sources': self.include_sources
        }
    
    def update_configuration(self, **kwargs):
        """
        Update generation configuration at runtime.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        updated_params = []
        
        for key, value in kwargs.items():
            if key == 'temperature':
                self.temperature = value
                updated_params.append(key)
            elif key == 'max_tokens':
                self.max_tokens = value
                updated_params.append(key)
            elif key == 'max_context_length':
                self.max_context_length = value
                updated_params.append(key)
            elif key == 'prompt_template':
                if value in self.prompt_templates:
                    self.prompt_template_name = value
                    updated_params.append(key)
                else:
                    logger.warning(f"Template '{value}' not found")
            elif key == 'include_sources':
                self.include_sources = value
                updated_params.append(key)
        
        # Reinitialize LLM if model parameters changed
        if any(param in ['temperature', 'max_tokens'] for param in updated_params):
            logger.info("Reinitializing LLM with new parameters")
            self.llm = self._create_llm()
        
        if updated_params:
            logger.info(f"Generation configuration updated: {updated_params}")
    
    def test_generation(self, test_query: str = "What is artificial intelligence?") -> Dict[str, Any]:
        """
        Test the generation capability with a simple query.
        
        Args:
            test_query: Query to test with
            
        Returns:
            Test results
        """
        try:
            start_time = time.time()
            
            # Create a dummy retrieval result for testing
            from ..models.document import TextChunk
            
            test_chunk = TextChunk(
                document_id="test",
                content="Artificial Intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence.",
                start_index=0,
                end_index=150,
                chunk_index=0,
                chunk_size=150
            )
            
            test_result = RetrievalResult(
                chunk=test_chunk,
                score=0.95,
                rank=1
            )
            
            # Generate response
            response = self.generate_response(test_query, [test_result])
            
            test_time = time.time() - start_time
            
            return {
                'status': 'success',
                'query': test_query,
                'response_length': len(response.answer),
                'processing_time': test_time,
                'model_used': response.model_name,
                'template_used': response.generation_metadata.get('prompt_template'),
                'sample_response': response.answer[:200] + "..." if len(response.answer) > 200 else response.answer
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }