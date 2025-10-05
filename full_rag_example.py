#!/usr/bin/env python3
"""
Complete RAG Pipeline Example

This script demonstrates the full RAG pipeline including document processing,
embedding generation, retrieval, and response generation.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.rag_system import RAGSystem


def main():
    """Demonstrate the complete RAG pipeline."""
    print("🚀 Complete RAG Pipeline Example")
    print("=" * 60)
    
    try:
        # Initialize RAG system
        print("\n1. Initializing RAG System...")
        rag = RAGSystem()
        
        # Check system health
        print("\n2. System Health Check:")
        health = rag.health_check()
        print(f"   Overall Status: {health['overall_status']}")
        
        for component, status in health['components'].items():
            status_icon = "✅" if "healthy" in status else "❌" if "error" in status else "⚠️"
            print(f"   {status_icon} {component}: {status}")
        
        if health['overall_status'] != 'healthy':
            print(f"\n⚠️ System not fully healthy. Some features may not work correctly.")
            print("   Make sure you have:")
            print("   - Set OPENAI_API_KEY in .env file (for generation)")
            print("   - Started Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        
        # Create comprehensive sample documents
        print("\n3. Creating Sample Documents...")
        sample_docs_dir = Path("data/raw")
        sample_docs_dir.mkdir(parents=True, exist_ok=True)
        
        # AI/ML content
        ai_content = sample_docs_dir / "artificial_intelligence.txt"
        ai_content.write_text("""
Artificial Intelligence: A Comprehensive Overview

Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior. The field encompasses various subfields including machine learning, natural language processing, computer vision, and robotics.

Machine Learning
Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. There are three main types of machine learning:

1. Supervised Learning: Uses labeled data to train models for prediction or classification tasks.
2. Unsupervised Learning: Finds patterns in data without labeled examples.
3. Reinforcement Learning: Learns through interaction with an environment using rewards and penalties.

Deep Learning
Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. It has revolutionized fields such as:
- Image recognition and computer vision
- Natural language processing and translation
- Speech recognition
- Autonomous vehicles

Applications of AI
AI has numerous applications across industries:
- Healthcare: Medical diagnosis, drug discovery, personalized treatment
- Finance: Fraud detection, algorithmic trading, credit scoring
- Transportation: Self-driving cars, route optimization
- Entertainment: Recommendation systems, content generation
- Customer Service: Chatbots, automated support systems

Challenges and Ethics
AI development faces several challenges:
- Data privacy and security concerns
- Algorithmic bias and fairness
- Job displacement due to automation
- The need for explainable AI systems
- Ensuring AI safety and alignment with human values
        """)
        
        # RAG systems content
        rag_content = sample_docs_dir / "rag_systems.txt"
        rag_content.write_text("""
Retrieval-Augmented Generation (RAG) Systems

RAG systems combine the power of large language models with external knowledge retrieval to provide more accurate and up-to-date responses. This approach addresses limitations of pure language models such as knowledge cutoffs and hallucinations.

How RAG Works
1. Document Processing: Raw documents are parsed and cleaned
2. Text Chunking: Documents are split into smaller, manageable pieces
3. Embedding Generation: Text chunks are converted to vector representations
4. Vector Storage: Embeddings are stored in specialized vector databases
5. Query Processing: User queries are converted to embeddings
6. Similarity Search: Relevant chunks are retrieved based on semantic similarity
7. Context Formation: Retrieved chunks are combined into context
8. Response Generation: LLM generates responses using the retrieved context

Components of RAG Systems
- Document Parser: Extracts text from various file formats (PDF, DOCX, HTML, etc.)
- Text Splitter: Divides documents into chunks with optional overlap
- Embedding Model: Converts text to dense vector representations
- Vector Database: Stores and indexes embeddings for fast retrieval
- Retrieval System: Performs semantic search over the knowledge base
- Language Model: Generates human-like responses using retrieved context

Benefits of RAG
- Provides access to current, domain-specific information
- Reduces hallucinations by grounding responses in real data
- Enables knowledge base updates without retraining models
- Offers transparency through source attribution
- Scales efficiently with growing knowledge bases

Vector Databases for RAG
Popular vector databases include:
- Qdrant: High-performance vector search engine
- Chroma: Lightweight, easy-to-use vector database
- Pinecone: Managed vector database service
- Weaviate: Open-source vector database with GraphQL API
        """)
        
        # Programming content
        python_content = sample_docs_dir / "python_programming.txt"
        python_content.write_text("""
Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity and versatility. Created by Guido van Rossum in 1991, Python emphasizes code readability and allows developers to express concepts in fewer lines of code.

Key Features of Python
- Simple and readable syntax
- Interpreted language (no compilation needed)
- Dynamic typing
- Extensive standard library
- Large ecosystem of third-party packages
- Cross-platform compatibility
- Strong community support

Python Data Types
Python supports various built-in data types:

1. Numbers: integers (int), floating-point numbers (float), complex numbers
2. Strings: sequences of characters, immutable
3. Lists: ordered, mutable collections
4. Tuples: ordered, immutable collections  
5. Dictionaries: unordered key-value pairs
6. Sets: unordered collections of unique elements
7. Booleans: True or False values

Control Flow
Python provides standard control flow statements:
- if/elif/else statements for conditional execution
- for loops for iteration over sequences
- while loops for conditional repetition
- break and continue statements for loop control
- try/except for exception handling

Functions and Classes
- Functions are defined using the 'def' keyword
- Classes use the 'class' keyword
- Python supports object-oriented programming
- Multiple inheritance is supported
- Special methods (dunder methods) customize object behavior

Popular Libraries
- NumPy: numerical computing
- Pandas: data manipulation and analysis
- Matplotlib/Seaborn: data visualization
- Scikit-learn: machine learning
- TensorFlow/PyTorch: deep learning
- Django/Flask: web development
- Requests: HTTP library
- Beautiful Soup: web scraping
        """)
        
        print(f"   Created 3 sample documents in: {sample_docs_dir}")
        
        # Add documents to the system
        print("\n4. Processing Documents...")
        documents = rag.add_documents(str(sample_docs_dir))
        successful = len([d for d in documents if d.processing_status.value == "indexed"])
        print(f"   Successfully processed: {successful}/{len(documents)} documents")
        
        if successful == 0:
            print("   ⚠️ No documents were successfully processed. Check vector database connection.")
            return
        
        # System status after processing
        print("\n5. Updated System Status:")
        status = rag.get_system_status()
        print(f"   Documents: {status['metrics']['total_documents']}")
        print(f"   Chunks: {status['metrics']['total_chunks']}")
        print(f"   Vector DB: {status.get('vector_database', {}).get('provider', 'Not available')}")
        
        # Test different types of queries
        print("\n6. Testing RAG Pipeline with Different Query Types:")
        print("-" * 60)
        
        queries = [
            {
                "question": "What is artificial intelligence and what are its main applications?",
                "type": "General"
            },
            {
                "question": "How do RAG systems work and what are their benefits?",
                "type": "Technical"
            },
            {
                "question": "What are the different types of machine learning?",
                "type": "Educational"
            },
            {
                "question": "What Python libraries are popular for machine learning?",
                "type": "Programming"
            }
        ]
        
        for i, query_info in enumerate(queries, 1):
            question = query_info["question"]
            query_type = query_info["type"]
            
            print(f"\n   Query {i} ({query_type}):")
            print(f"   Q: {question}")
            
            try:
                # Test simple query
                answer = rag.ask(question)
                print(f"   A: {answer[:200]}..." if len(answer) > 200 else f"   A: {answer}")
                
                # Get detailed information
                detailed = rag.query_with_sources(question, top_k=3)
                print(f"   Sources used: {len(detailed['sources'])}")
                print(f"   Processing time: {detailed['processing_time']:.3f}s")
                
                if detailed['sources']:
                    print("   Top source:")
                    top_source = detailed['sources'][0]
                    print(f"   - Document: {top_source['document']}")
                    print(f"   - Similarity: {top_source['score']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        # Test different prompt templates
        print(f"\n7. Testing Different Prompt Templates:")
        print("-" * 60)
        
        test_question = "What are the challenges in AI development?"
        
        templates = ["default", "detailed", "summary", "factual"]
        for template in templates:
            print(f"\n   Template: {template}")
            try:
                if template == "summary":
                    response = rag.summarize_query(test_question)
                elif template == "detailed":
                    response = rag.detailed_query(test_question)
                elif template == "factual":
                    response = rag.factual_query(test_question)
                else:
                    response = rag.query(test_question, prompt_template=template)
                
                print(f"   Response length: {len(response.answer)} characters")
                print(f"   Sources: {len(response.sources)}")
                print(f"   Sample: {response.answer[:150]}...")
                
            except Exception as e:
                print(f"   ❌ Error with {template} template: {str(e)}")
        
        # Final system metrics
        print(f"\n8. Final System Metrics:")
        print("-" * 60)
        final_status = rag.get_system_status()
        metrics = final_status['metrics']
        
        print(f"   📊 Total Queries Processed: {metrics['total_queries']}")
        print(f"   💾 Documents in System: {metrics['total_documents']}")
        print(f"   🔗 Text Chunks: {metrics['total_chunks']}")
        if metrics.get('avg_response_time'):
            print(f"   ⏱️ Average Response Time: {metrics['avg_response_time']:.3f}s")
        
        # Configuration summary
        config = rag.get_configuration()
        print(f"\n   🔧 Configuration:")
        print(f"   - Embedding Model: {config['embeddings']['model_name']}")
        print(f"   - Vector DB: {config['vector_db']['provider']}")
        print(f"   - Chunk Size: {config['chunking']['chunk_size']}")
        print(f"   - Chunk Overlap: {config['chunking']['chunk_overlap']}")
        
        print(f"\n✅ RAG Pipeline demonstration completed successfully!")
        
        # Usage recommendations
        print(f"\n💡 Next Steps:")
        print("   1. Add your own documents to data/raw/ directory")
        print("   2. Experiment with different chunk sizes and overlap")
        print("   3. Try different embedding models")
        print("   4. Adjust retrieval parameters (top_k, score_threshold)")
        print("   5. Create custom prompt templates for specific use cases")
        print("   6. Run the Streamlit dashboard: streamlit run dashboards/main_dashboard.py")
        
    except Exception as e:
        print(f"\n❌ Error during RAG pipeline demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting:")
        print("   1. Ensure vector database is running (Qdrant)")
        print("   2. Check your .env file has OPENAI_API_KEY set")
        print("   3. Verify all dependencies are installed")
        print("   4. Check logs in logs/ directory")


if __name__ == "__main__":
    main()