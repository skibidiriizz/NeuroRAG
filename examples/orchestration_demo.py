"""
Demonstration of LangGraph Orchestration System

This script shows how to use the RAG Orchestrator for advanced workflow management,
including different workflow types and async execution patterns.
"""

import asyncio
import json
from pathlib import Path
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_system import RAGSystem
from src.core.orchestrator import RAGOrchestrator
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def demo_standard_rag_workflow():
    """Demonstrate the standard RAG workflow using orchestration."""
    print("\n🔄 Testing Standard RAG Workflow")
    print("=" * 50)
    
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Initialize orchestrator
        orchestrator = RAGOrchestrator(
            rag_system=rag,
            config={
                'max_retries': 2,
                'timeout_seconds': 120,
                'enable_evaluation': True
            }
        )
        
        # Add some sample documents first
        sample_docs = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Natural language processing enables computers to understand and process human language.",
            "Deep learning uses neural networks with multiple layers to learn complex patterns."
        ]
        
        for i, content in enumerate(sample_docs):
            doc_path = f"temp_doc_{i}.txt"
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(content)
            rag.add_document(doc_path)
            os.remove(doc_path)  # Clean up
        
        # Execute standard workflow
        result = await orchestrator.execute_workflow(
            workflow_type="standard",
            query="What is machine learning and how does it relate to AI?"
        )
        
        print(f"✅ Workflow completed successfully")
        print(f"📊 Processing times: {result.get('processing_times', {})}")
        
        if result.get('generated_response'):
            response = result['generated_response']
            print(f"🤖 Generated Answer: {response.answer}")
            print(f"📚 Sources used: {len(response.sources)}")
        
        if result.get('evaluation_results'):
            eval_results = result['evaluation_results']
            print(f"📈 Evaluation metrics: {json.dumps(eval_results, indent=2)}")
            
    except Exception as e:
        logger.error(f"Standard workflow demo failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


async def demo_document_processing_workflow():
    """Demonstrate document processing workflow."""
    print("\n📄 Testing Document Processing Workflow")
    print("=" * 50)
    
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Initialize orchestrator
        orchestrator = RAGOrchestrator(rag_system=rag)
        
        # Create sample documents
        sample_documents = []
        for i in range(3):
            doc_path = f"sample_doc_{i}.txt"
            content = f"This is sample document {i} with content about topic {i}. " \
                     f"It contains information relevant to testing document processing workflows."
            
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            sample_documents.append(doc_path)
        
        # Execute document processing workflow
        result = await orchestrator.execute_workflow(
            workflow_type="document_processing",
            documents=sample_documents
        )
        
        print(f"✅ Document processing completed")
        print(f"📊 Processing times: {result.get('processing_times', {})}")
        
        if result.get('processed_documents'):
            docs = result['processed_documents']
            print(f"📚 Processed {len(docs)} documents")
        
        if result.get('intermediate_results'):
            intermediate = result['intermediate_results']
            print(f"🔍 Intermediate results: {json.dumps(intermediate, indent=2)}")
        
        # Clean up
        for doc_path in sample_documents:
            if os.path.exists(doc_path):
                os.remove(doc_path)
                
    except Exception as e:
        logger.error(f"Document processing workflow demo failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


async def demo_batch_processing_workflow():
    """Demonstrate batch processing workflow."""
    print("\n🔄 Testing Batch Processing Workflow")
    print("=" * 50)
    
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Initialize orchestrator
        orchestrator = RAGOrchestrator(rag_system=rag)
        
        # Add sample data first
        sample_content = "Artificial intelligence is transforming various industries including healthcare, finance, and transportation."
        doc_path = "ai_sample.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        rag.add_document(doc_path)
        os.remove(doc_path)
        
        # Define batch queries
        batch_queries = [
            "What is artificial intelligence?",
            "How is AI used in healthcare?",
            "What industries are being transformed by AI?"
        ]
        
        # Execute batch processing workflow
        result = await orchestrator.execute_workflow(
            workflow_type="batch_processing",
            queries=batch_queries,
            batch_size=2
        )
        
        print(f"✅ Batch processing completed")
        print(f"📊 Processing times: {result.get('processing_times', {})}")
        
        if result.get('batch_results'):
            batch_results = result['batch_results']
            print(f"📦 Processed {len(batch_queries)} queries in batch")
            
    except Exception as e:
        logger.error(f"Batch processing workflow demo failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


async def demo_workflow_introspection():
    """Demonstrate workflow introspection capabilities."""
    print("\n🔍 Testing Workflow Introspection")
    print("=" * 50)
    
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Initialize orchestrator
        orchestrator = RAGOrchestrator(rag_system=rag)
        
        # List available workflows
        workflows = orchestrator.list_available_workflows()
        print(f"📋 Available workflows: {workflows}")
        
        # Get schemas for each workflow
        for workflow_type in workflows:
            schema = orchestrator.get_workflow_schema(workflow_type)
            print(f"\n🔧 Schema for '{workflow_type}':")
            print(f"   Description: {schema.get('description', 'N/A')}")
            print(f"   Required inputs: {schema.get('required_inputs', [])}")
            print(f"   Optional inputs: {schema.get('optional_inputs', [])}")
            print(f"   Outputs: {schema.get('outputs', [])}")
            
    except Exception as e:
        logger.error(f"Workflow introspection demo failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


async def demo_error_handling():
    """Demonstrate error handling in workflows."""
    print("\n⚠️ Testing Error Handling")
    print("=" * 50)
    
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Initialize orchestrator
        orchestrator = RAGOrchestrator(rag_system=rag)
        
        # Execute workflow with invalid input (empty query)
        result = await orchestrator.execute_workflow(
            workflow_type="standard",
            query=""  # Invalid empty query
        )
        
        print(f"🔍 Error handling test result:")
        print(f"   Status: {'failed' if result.get('error_message') else 'success'}")
        if result.get('error_message'):
            print(f"   Error: {result['error_message']}")
        
        # Execute workflow with unknown type
        try:
            result = await orchestrator.execute_workflow(
                workflow_type="unknown_workflow",
                query="test query"
            )
        except ValueError as e:
            print(f"✅ Correctly caught unknown workflow error: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error handling demo failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


async def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n📊 Testing Performance Monitoring")
    print("=" * 50)
    
    try:
        # Initialize the RAG system
        rag = RAGSystem()
        
        # Initialize orchestrator with performance monitoring
        orchestrator = RAGOrchestrator(
            rag_system=rag,
            config={
                'enable_evaluation': True,
                'timeout_seconds': 60
            }
        )
        
        # Add sample content
        content = "Performance monitoring is crucial for understanding system behavior and optimization opportunities."
        doc_path = "performance_doc.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        rag.add_document(doc_path)
        os.remove(doc_path)
        
        # Execute workflow and measure performance
        import time
        start_time = time.time()
        
        result = await orchestrator.execute_workflow(
            workflow_type="standard",
            query="What is performance monitoring?"
        )
        
        total_time = time.time() - start_time
        
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        
        if result.get('processing_times'):
            processing_times = result['processing_times']
            print(f"🔍 Detailed timing breakdown:")
            for step, duration in processing_times.items():
                print(f"   {step}: {duration:.3f}s")
        
        # Calculate performance metrics
        if result.get('generated_response'):
            response = result['generated_response']
            print(f"📈 Response metrics:")
            print(f"   Length: {len(response.answer)} characters")
            print(f"   Sources: {len(response.sources)}")
            print(f"   Processing time: {response.processing_time:.3f}s")
            
    except Exception as e:
        logger.error(f"Performance monitoring demo failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


async def main():
    """Run all orchestration demonstrations."""
    print("🚀 RAG Orchestration System Demo")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrations = [
        ("Standard RAG Workflow", demo_standard_rag_workflow),
        ("Document Processing Workflow", demo_document_processing_workflow),
        ("Batch Processing Workflow", demo_batch_processing_workflow),
        ("Workflow Introspection", demo_workflow_introspection),
        ("Error Handling", demo_error_handling),
        ("Performance Monitoring", demo_performance_monitoring),
    ]
    
    for demo_name, demo_func in demonstrations:
        try:
            print(f"\n🎯 Running: {demo_name}")
            await demo_func()
            print(f"✅ {demo_name} completed successfully")
        except Exception as e:
            logger.error(f"{demo_name} failed: {str(e)}")
            print(f"❌ {demo_name} failed: {str(e)}")
        
        # Brief pause between demonstrations
        await asyncio.sleep(1)
    
    print(f"\n🎉 All orchestration demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())