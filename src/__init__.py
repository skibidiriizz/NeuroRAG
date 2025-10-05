"""
RAG Agent System - A comprehensive Retrieval-Augmented Generation system.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Comprehensive RAG system with intelligent agents"

from .core.rag_system import RAGSystem, create_rag_system, quick_setup
from .core.config_manager import ConfigManager

__all__ = [
    "RAGSystem",
    "create_rag_system", 
    "quick_setup",
    "ConfigManager"
]