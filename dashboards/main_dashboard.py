"""
Main Streamlit Dashboard for RAG Agent System

This dashboard provides an interactive interface for managing documents,
querying the system, monitoring performance, and visualizing results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from src.core.rag_system import RAGSystem
from src.agents.evaluation import EvaluationAgent
from src.models.document import GenerationResponse, ProcessingStatus


# Page configuration
st.set_page_config(
    page_title="RAG Agent System Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-healthy { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached for performance)."""
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None


@st.cache_resource
def initialize_evaluation_agent():
    """Initialize the evaluation agent (cached for performance)."""
    try:
        return EvaluationAgent()
    except Exception as e:
        st.warning(f"Failed to initialize evaluation agent: {str(e)}")
        return None


def display_system_header():
    """Display the main system header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Agent System Dashboard</h1>
        <p>Comprehensive Retrieval-Augmented Generation System</p>
    </div>
    """, unsafe_allow_html=True)


def display_system_status(rag_system):
    """Display system status and health information."""
    st.header("🔍 System Status")
    
    if rag_system is None:
        st.error("RAG System not initialized!")
        return
    
    # Get system status
    with st.spinner("Checking system status..."):
        status = rag_system.get_system_status()
        health = rag_system.health_check()
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Health", 
            health['overall_status'].upper(),
            delta="Operational" if health['overall_status'] == 'healthy' else "Issues Detected"
        )
    
    with col2:
        st.metric(
            "Documents", 
            status['metrics']['total_documents'],
            delta=f"{status['metrics']['total_chunks']} chunks"
        )
    
    with col3:
        st.metric(
            "Queries Processed", 
            status['metrics']['total_queries'],
            delta=f"{status['metrics']['total_responses']} responses"
        )
    
    with col4:
        avg_time = status['metrics'].get('avg_response_time')
        if avg_time:
            st.metric(
                "Avg Response Time", 
                f"{avg_time:.3f}s",
                delta="Performance"
            )
        else:
            st.metric("Avg Response Time", "N/A", delta="No queries yet")
    
    # Component status
    st.subheader("Component Health")
    
    components_df = pd.DataFrame([
        {
            "Component": component.replace('_', ' ').title(),
            "Status": status,
            "Health": "🟢" if "healthy" in status.lower() else "🟡" if "warning" in status.lower() else "🔴"
        }
        for component, status in health['components'].items()
    ])
    
    st.dataframe(components_df, hide_index=True, use_container_width=True)
    
    # Configuration info
    with st.expander("📋 Configuration Details"):
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write("**Embedding Configuration:**")
            st.write(f"- Provider: {status['configuration']['embedding_provider']}")
            st.write(f"- Model: {status['configuration']['embedding_model']}")
            st.write(f"- Chunking: {status['configuration']['chunking_strategy']}")
        
        with config_col2:
            st.write("**Vector Database:**")
            st.write(f"- Provider: {status['configuration']['vector_db_provider']}")
            if 'vector_database' in status:
                db_info = status['vector_database']
                if 'vectors_count' in db_info:
                    st.write(f"- Vectors: {db_info['vectors_count']}")
                elif 'documents_count' in db_info:
                    st.write(f"- Documents: {db_info['documents_count']}")


def document_management_tab(rag_system):
    """Document management interface."""
    st.header("📄 Document Management")
    
    if rag_system is None:
        st.error("RAG System not initialized!")
        return
    
    # File upload section
    st.subheader("Upload Documents")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["File Upload", "Directory Path", "URL"]
    )
    
    if upload_method == "File Upload":
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'html']
        )
        
        if uploaded_files and st.button("Process Uploaded Files"):
            with st.spinner("Processing uploaded files..."):
                results = []
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    try:
                        document = rag_system.add_document(temp_path)
                        results.append({
                            "File": uploaded_file.name,
                            "Status": document.processing_status.value,
                            "Size": len(uploaded_file.getvalue()),
                            "Error": document.error_message or "None"
                        })
                    except Exception as e:
                        results.append({
                            "File": uploaded_file.name,
                            "Status": "failed",
                            "Size": len(uploaded_file.getvalue()),
                            "Error": str(e)
                        })
                    
                    # Cleanup
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                successful = len([r for r in results if r["Status"] == "indexed"])
                st.success(f"Processing complete! {successful}/{len(results)} files processed successfully.")
    
    elif upload_method == "Directory Path":
        directory_path = st.text_input("Directory path:", placeholder="/path/to/documents/")
        file_patterns = st.multiselect(
            "File patterns:", 
            ["*.pdf", "*.docx", "*.txt", "*.html"],
            default=["*.pdf", "*.docx", "*.txt", "*.html"]
        )
        
        if directory_path and st.button("Process Directory"):
            if os.path.exists(directory_path):
                with st.spinner("Processing directory..."):
                    try:
                        documents = rag_system.add_documents(directory_path, file_patterns)
                        
                        results_data = []
                        for doc in documents:
                            results_data.append({
                                "File": doc.filename,
                                "Status": doc.processing_status.value,
                                "Type": doc.document_type.value,
                                "Error": doc.error_message or "None"
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        successful = len([d for d in documents if d.processing_status == ProcessingStatus.INDEXED])
                        st.success(f"Processing complete! {successful}/{len(documents)} documents processed successfully.")
                        
                    except Exception as e:
                        st.error(f"Error processing directory: {str(e)}")
            else:
                st.error(f"Directory not found: {directory_path}")
    
    elif upload_method == "URL":
        url = st.text_input("URL:", placeholder="https://example.com/document.html")
        
        if url and st.button("Process URL"):
            with st.spinner("Processing URL..."):
                try:
                    document = rag_system.add_url(url)
                    
                    if document.processing_status == ProcessingStatus.INDEXED:
                        st.success(f"✅ Successfully processed: {document.filename}")
                        st.write(f"**Content length:** {len(document.content)} characters")
                        st.write(f"**Document type:** {document.document_type.value}")
                    else:
                        st.error(f"❌ Failed to process URL: {document.error_message}")
                        
                except Exception as e:
                    st.error(f"Error processing URL: {str(e)}")
    
    # Document validation
    st.subheader("🔍 Document Validation")
    
    validation_file = st.text_input("File path to validate:", placeholder="/path/to/document.pdf")
    
    if validation_file and st.button("Validate File"):
        try:
            validation_result = rag_system.validate_document(validation_file)
            
            if validation_result['valid']:
                st.success("✅ File is valid and can be processed")
                st.write(f"**Size:** {validation_result['file_info']['size_mb']:.2f} MB")
                st.write(f"**Type:** {validation_result['file_info']['detected_type']}")
            else:
                st.error("❌ File validation failed")
                for error in validation_result['errors']:
                    st.error(f"- {error}")
                
                for warning in validation_result['warnings']:
                    st.warning(f"- {warning}")
                    
        except Exception as e:
            st.error(f"Validation error: {str(e)}")


def query_interface_tab(rag_system, evaluation_agent):
    """Query interface for interacting with the RAG system."""
    st.header("💬 Query Interface")
    
    if rag_system is None:
        st.error("RAG System not initialized!")
        return
    
    # Query input
    query_text = st.text_area(
        "Enter your question:",
        placeholder="What is artificial intelligence and how does it work?",
        height=100
    )
    
    # Query options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        query_type = st.selectbox(
            "Query Type:",
            ["Standard", "Detailed Analysis", "Summary", "Factual", "Custom"]
        )
    
    with col2:
        top_k = st.slider("Number of sources:", 1, 20, 5)
    
    with col3:
        score_threshold = st.slider("Score threshold:", 0.0, 1.0, 0.7, 0.1)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        prompt_template = None
        if query_type == "Custom":
            prompt_template = st.text_area(
                "Custom prompt template:",
                placeholder="You are an expert assistant. Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            )
        
        include_evaluation = st.checkbox("Include response evaluation", value=True)
        show_sources = st.checkbox("Show detailed sources", value=True)
    
    # Query execution
    if st.button("🔍 Ask Question", type="primary") and query_text.strip():
        with st.spinner("Processing your question..."):
            start_time = time.time()
            
            try:
                # Execute query based on type
                if query_type == "Standard":
                    response = rag_system.query(query_text, top_k=top_k, score_threshold=score_threshold)
                elif query_type == "Detailed Analysis":
                    response = rag_system.detailed_query(query_text, top_k=top_k)
                elif query_type == "Summary":
                    response = rag_system.summarize_query(query_text, top_k=top_k)
                elif query_type == "Factual":
                    response = rag_system.factual_query(query_text, top_k=top_k, score_threshold=score_threshold)
                elif query_type == "Custom" and prompt_template:
                    # Add custom template to generation agent
                    rag_system.generation.add_custom_template("custom", prompt_template, ["context", "question"])
                    response = rag_system.query(query_text, top_k=top_k, prompt_template="custom")
                else:
                    response = rag_system.query(query_text, top_k=top_k, score_threshold=score_threshold)
                
                processing_time = time.time() - start_time
                
                # Display response
                st.success("✅ Query completed successfully!")
                
                # Main answer
                st.subheader("📝 Answer")
                st.write(response.answer)
                
                # Response metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sources Used", len(response.sources))
                with col2:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                with col3:
                    st.metric("Context Length", len(response.context_used))
                with col4:
                    st.metric("Response Length", len(response.answer))
                
                # Sources section
                if show_sources and response.sources:
                    st.subheader("📚 Sources")
                    
                    for i, result in enumerate(response.sources, 1):
                        with st.expander(f"Source {i} (Score: {result.score:.3f})"):
                            st.write(f"**Document:** {result.chunk.metadata.get('document_filename', 'Unknown')}")
                            st.write(f"**Chunk Index:** {result.chunk.chunk_index}")
                            st.write(f"**Content:**")
                            st.write(result.chunk.content)
                
                # Evaluation section
                if include_evaluation and evaluation_agent:
                    st.subheader("📊 Response Evaluation")
                    
                    with st.spinner("Evaluating response quality..."):
                        try:
                            metrics = evaluation_agent.evaluate_response(response)
                            
                            # Display metrics
                            eval_col1, eval_col2, eval_col3 = st.columns(3)
                            
                            with eval_col1:
                                if metrics.faithfulness_score is not None:
                                    st.metric("Faithfulness", f"{metrics.faithfulness_score:.3f}")
                                if metrics.fluency_score is not None:
                                    st.metric("Fluency", f"{metrics.fluency_score:.3f}")
                            
                            with eval_col2:
                                if metrics.relevance_score is not None:
                                    st.metric("Relevance", f"{metrics.relevance_score:.3f}")
                                if metrics.coherence_score is not None:
                                    st.metric("Coherence", f"{metrics.coherence_score:.3f}")
                            
                            with eval_col3:
                                if metrics.groundedness_score is not None:
                                    st.metric("Groundedness", f"{metrics.groundedness_score:.3f}")
                                if metrics.overall_score is not None:
                                    st.metric("Overall Score", f"{metrics.overall_score:.3f}")
                            
                            # Additional metrics
                            if metrics.evaluation_metadata:
                                with st.expander("📈 Additional Metrics"):
                                    for key, value in metrics.evaluation_metadata.items():
                                        if isinstance(value, (int, float)):
                                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        except Exception as e:
                            st.warning(f"Evaluation failed: {str(e)}")
                
                # Save query to session state for analytics
                if 'query_history' not in st.session_state:
                    st.session_state.query_history = []
                
                st.session_state.query_history.append({
                    'timestamp': datetime.now(),
                    'query': query_text,
                    'response_length': len(response.answer),
                    'sources_count': len(response.sources),
                    'processing_time': processing_time,
                    'query_type': query_type
                })
            
            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
    
    # Query history
    if 'query_history' in st.session_state and st.session_state.query_history:
        st.subheader("📜 Query History")
        
        history_df = pd.DataFrame(st.session_state.query_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Display recent queries
        recent_queries = history_df.tail(5).sort_values('timestamp', ascending=False)
        
        for _, query_record in recent_queries.iterrows():
            with st.expander(f"🕐 {query_record['timestamp'].strftime('%H:%M:%S')} - {query_record['query'][:50]}..."):
                st.write(f"**Query Type:** {query_record['query_type']}")
                st.write(f"**Processing Time:** {query_record['processing_time']:.2f}s")
                st.write(f"**Sources Used:** {query_record['sources_count']}")
                st.write(f"**Response Length:** {query_record['response_length']} characters")


def analytics_tab():
    """Analytics and performance monitoring."""
    st.header("📊 Analytics & Performance")
    
    # Check if we have query history
    if 'query_history' not in st.session_state or not st.session_state.query_history:
        st.info("No query data available. Run some queries to see analytics!")
        return
    
    history_df = pd.DataFrame(st.session_state.query_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    
    # Performance metrics
    st.subheader("⚡ Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        avg_response_time = history_df['processing_time'].mean()
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    with perf_col2:
        total_queries = len(history_df)
        st.metric("Total Queries", total_queries)
    
    with perf_col3:
        avg_sources = history_df['sources_count'].mean()
        st.metric("Avg Sources Used", f"{avg_sources:.1f}")
    
    with perf_col4:
        avg_response_length = history_df['response_length'].mean()
        st.metric("Avg Response Length", f"{avg_response_length:.0f} chars")
    
    # Visualizations
    st.subheader("📈 Query Analytics")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Response time over time
        fig_time = px.line(
            history_df, 
            x='timestamp', 
            y='processing_time',
            title='Response Time Over Time',
            labels={'processing_time': 'Response Time (s)', 'timestamp': 'Time'}
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with viz_col2:
        # Query types distribution
        query_type_counts = history_df['query_type'].value_counts()
        fig_types = px.pie(
            values=query_type_counts.values,
            names=query_type_counts.index,
            title='Query Types Distribution'
        )
        fig_types.update_layout(height=400)
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Sources utilization
    fig_sources = px.histogram(
        history_df,
        x='sources_count',
        title='Sources Count Distribution',
        labels={'sources_count': 'Number of Sources Used', 'count': 'Frequency'}
    )
    st.plotly_chart(fig_sources, use_container_width=True)
    
    # Response length distribution
    fig_length = px.box(
        history_df,
        y='response_length',
        x='query_type',
        title='Response Length by Query Type',
        labels={'response_length': 'Response Length (characters)', 'query_type': 'Query Type'}
    )
    st.plotly_chart(fig_length, use_container_width=True)


def settings_tab(rag_system):
    """System settings and configuration."""
    st.header("⚙️ Settings & Configuration")
    
    if rag_system is None:
        st.error("RAG System not initialized!")
        return
    
    # Current configuration
    st.subheader("📋 Current Configuration")
    
    try:
        current_config = rag_system.get_configuration()
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write("**Application Settings:**")
            for key, value in current_config['app'].items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
            
            st.write("**Chunking Settings:**")
            for key, value in current_config['chunking'].items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        with config_col2:
            st.write("**Embeddings Settings:**")
            for key, value in current_config['embeddings'].items():
                if key != 'api_key':  # Don't show sensitive info
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
            
            st.write("**Vector Database:**")
            for key, value in current_config['vector_db'].items():
                if key not in ['api_key', 'url']:  # Don't show sensitive info
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
    
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
    
    # Runtime configuration updates
    st.subheader("🔧 Runtime Configuration")
    
    st.info("⚠️ Configuration changes will affect new queries only.")
    
    with st.form("config_form"):
        st.write("**Retrieval Settings:**")
        new_top_k = st.slider("Default Top-K", 1, 20, 5)
        new_score_threshold = st.slider("Default Score Threshold", 0.0, 1.0, 0.7, 0.1)
        
        st.write("**Generation Settings:**")
        new_temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.1)
        new_max_tokens = st.number_input("Max Tokens", 100, 4000, 1500)
        
        if st.form_submit_button("Update Configuration"):
            try:
                # Update retrieval settings
                rag_system.retrieval.update_configuration(
                    top_k=new_top_k,
                    score_threshold=new_score_threshold
                )
                
                # Update generation settings
                rag_system.generation.update_configuration(
                    temperature=new_temperature,
                    max_tokens=new_max_tokens
                )
                
                st.success("✅ Configuration updated successfully!")
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to update configuration: {str(e)}")
    
    # System actions
    st.subheader("🔄 System Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Reload Configuration"):
            try:
                rag_system.reload_configuration()
                st.success("✅ Configuration reloaded!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Reload failed: {str(e)}")
    
    with action_col2:
        if st.button("🩺 Run Health Check"):
            with st.spinner("Running health check..."):
                try:
                    health = rag_system.health_check()
                    if health['overall_status'] == 'healthy':
                        st.success("✅ All systems healthy!")
                    else:
                        st.warning(f"⚠️ System status: {health['overall_status']}")
                        for component, status in health['components'].items():
                            if 'error' in status.lower():
                                st.error(f"- {component}: {status}")
                except Exception as e:
                    st.error(f"❌ Health check failed: {str(e)}")
    
    with action_col3:
        if st.button("📊 Export Logs"):
            try:
                # This would export system logs
                st.info("🔄 Log export functionality would be implemented here")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")


def main():
    """Main dashboard application."""
    
    # Initialize system
    rag_system = initialize_rag_system()
    evaluation_agent = initialize_evaluation_agent()
    
    # Display header
    display_system_header()
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>🧭 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    tab_selection = st.sidebar.radio(
        "Select Dashboard Tab:",
        ["System Status", "Document Management", "Query Interface", "Analytics", "Settings"],
        index=0
    )
    
    # System info in sidebar
    if rag_system:
        with st.sidebar.expander("ℹ️ System Info"):
            status = rag_system.get_system_status()
            st.write(f"**Version:** {status['system_info']['version']}")
            st.write(f"**Environment:** {status['system_info']['environment']}")
            st.write(f"**Documents:** {status['metrics']['total_documents']}")
            st.write(f"**Queries:** {status['metrics']['total_queries']}")
    
    # Display selected tab
    if tab_selection == "System Status":
        display_system_status(rag_system)
    elif tab_selection == "Document Management":
        document_management_tab(rag_system)
    elif tab_selection == "Query Interface":
        query_interface_tab(rag_system, evaluation_agent)
    elif tab_selection == "Analytics":
        analytics_tab()
    elif tab_selection == "Settings":
        settings_tab(rag_system)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>🤖 RAG Agent System Dashboard - Built with Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()