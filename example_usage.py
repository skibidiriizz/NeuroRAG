#!/usr/bin/env python3
"""
Example usage of the RAG Agent System

This script demonstrates how to use the RAG system for document processing,
embedding generation, and basic functionality.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.rag_system import RAGSystem, create_rag_system, quick_setup


def main():
    """Main example function."""
    print("🤖 RAG Agent System - Example Usage")
    print("=" * 50)
    
    try:
        # Method 1: Basic initialization
        print("\\n1. Creating RAG System...")
        rag = RAGSystem()
        
        # Check system status
        print("\\n2. System Status:")
        status = rag.get_system_status()
        print(f"   System: {status['system_info']['name']} v{status['system_info']['version']}")
        print(f"   Environment: {status['system_info']['environment']}")
        print(f"   Embedding Provider: {status['configuration']['embedding_provider']}")
        print(f"   Vector DB Provider: {status['configuration']['vector_db_provider']}")
        
        # Check supported formats
        print("\\n3. Supported Document Formats:")
        formats = rag.get_supported_formats()
        print(f"   {', '.join(formats)}")
        
        # Health check
        print("\\n4. System Health Check:")
        health = rag.health_check()
        print(f"   Overall Status: {health['overall_status']}")
        for component, status in health['components'].items():
            print(f"   {component}: {status}")
        
        # Create some sample documents for testing
        print("\\n5. Creating Sample Documents...")
        sample_docs_dir = Path("data/raw")
        sample_docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample text document
        sample_txt = sample_docs_dir / "sample.txt"
        sample_txt.write_text(
            "This is a sample document for testing the RAG system. "
            "It contains information about artificial intelligence and machine learning. "
            "RAG (Retrieval-Augmented Generation) is a powerful technique that combines "
            "information retrieval with text generation to provide more accurate and "
            "contextual responses. The system can process various document formats "
            "including PDF, DOCX, TXT, and HTML files."
        )
        
        # Sample HTML document
        sample_html = sample_docs_dir / "sample.html"
        sample_html.write_text("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG System Documentation</title>
            <meta name="author" content="AI Assistant">
            <meta name="description" content="Documentation for RAG system">
        </head>
        <body>
            <h1>RAG System Overview</h1>
            <p>The RAG Agent System is a comprehensive solution for document processing 
            and question answering. It consists of multiple specialized agents:</p>
            <ul>
                <li>Document Parser Agent</li>
                <li>Chunking & Embedding Agent</li>
                <li>Retrieval Agent</li>
                <li>Generation Agent</li>
                <li>Evaluation Agent</li>
            </ul>
            <p>Each agent has a specific role in the overall pipeline.</p>
        </body>
        </html>
        """)
        
        print(f"   Created sample documents in: {sample_docs_dir}")
        
        # Add documents to the system
        print("\\n6. Processing Documents...")
        
        # Process individual documents
        doc1 = rag.add_document(str(sample_txt))
        print(f"   Processed TXT: {doc1.filename} -> {doc1.processing_status}")
        
        doc2 = rag.add_document(str(sample_html))
        print(f"   Processed HTML: {doc2.filename} -> {doc2.processing_status}")
        
        # Process documents from directory
        documents = rag.add_documents("data/raw", ["*.txt", "*.html"])
        successful = len([d for d in documents if d.processing_status.value == "indexed"])
        print(f"   Batch processed: {successful}/{len(documents)} documents successful")
        
        # Updated system status
        print("\\n7. Updated System Metrics:")
        status = rag.get_system_status()
        metrics = status['metrics']
        print(f"   Total Documents: {metrics['total_documents']}")
        print(f"   Total Chunks: {metrics['total_chunks']}")
        if metrics['avg_response_time']:
            print(f"   Avg Processing Time: {metrics['avg_response_time']:.2f}s")
        
        # Vector database info
        if 'vector_database' in status:
            vector_info = status['vector_database']
            if 'error' not in vector_info:
                print("\\n8. Vector Database Info:")
                print(f"   Provider: {vector_info.get('provider', 'Unknown')}")
                print(f"   Collection: {vector_info.get('collection_name', 'Unknown')}")
                if 'vectors_count' in vector_info:
                    print(f"   Vectors Count: {vector_info['vectors_count']}")
                elif 'documents_count' in vector_info:
                    print(f"   Documents Count: {vector_info['documents_count']}")
            else:
                print(f"\\n8. Vector Database Error: {vector_info['error']}")
        
        # Test query (placeholder)
        print("\\n9. Testing Query (Placeholder):")
        question = "What is RAG and how does it work?"
        response = rag.query(question)
        print(f"   Question: {question}")
        print(f"   Response: {response}")
        
        # Configuration info
        print("\\n10. System Configuration:")
        config = rag.get_configuration()
        print(f"    App: {config['app']['name']} v{config['app']['version']}")
        print(f"    Chunking Strategy: {config['chunking'].get('strategy', 'default')}")
        print(f"    Chunk Size: {config['chunking'].get('chunk_size', 'default')}")
        print(f"    Embedding Model: {config['embeddings']['model_name']}")
        
        print("\\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


def quick_start_example():
    """Demonstrate quick start functionality."""
    print("\\n" + "=" * 50)
    print("🚀 Quick Start Example")
    print("=" * 50)
    
    try:
        # Quick setup with documents directory
        rag = quick_setup("data/raw")
        
        # Show status
        status = rag.get_system_status()
        print(f"\\nQuick Setup Results:")
        print(f"Documents Processed: {status['metrics']['total_documents']}")
        print(f"Chunks Created: {status['metrics']['total_chunks']}")
        
    except Exception as e:
        print(f"Quick start error: {str(e)}")


def validation_example():
    """Demonstrate document validation."""
    print("\\n" + "=" * 50)
    print("🔍 Document Validation Example")
    print("=" * 50)
    
    rag = RAGSystem()
    
    # Test files
    test_files = [
        "data/raw/sample.txt",
        "data/raw/sample.html",
        "nonexistent.pdf",
        "README.md"
    ]
    
    for file_path in test_files:
        print(f"\\nValidating: {file_path}")
        result = rag.validate_document(file_path)
        
        if result['valid']:
            print("   ✅ Valid")
            print(f"   Size: {result['file_info']['size_mb']:.2f} MB")
            print(f"   Type: {result['file_info']['detected_type']}")
        else:
            print("   ❌ Invalid")
            for error in result['errors']:
                print(f"   Error: {error}")
            for warning in result['warnings']:
                print(f"   Warning: {warning}")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run additional examples
    quick_start_example()
    validation_example()
    
    print("\\n🎉 All examples completed!")
    print("\\nNext steps:")
    print("1. Set up your API keys in .env file")
    print("2. Start a vector database (Qdrant: docker run -p 6333:6333 qdrant/qdrant)")
    print("3. Add your own documents to data/raw/")
    print("4. Run: python example_usage.py")
    print("5. Try the Streamlit dashboard: streamlit run dashboards/main_dashboard.py")