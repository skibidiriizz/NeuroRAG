"""
Configuration Manager for RAG Agent System

This module handles loading and managing configuration settings from YAML files
and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseSettings, Field
from pydantic_settings import SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    provider: str = Field(default="qdrant", description="Vector database provider")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=6333, description="Database port")
    collection_name: str = Field(default="rag_documents", description="Collection name")
    distance_metric: str = Field(default="cosine", description="Distance metric")
    url: Optional[str] = Field(default=None, description="Database URL")
    api_key: Optional[str] = Field(default=None, description="API key for cloud databases")

    model_config = SettingsConfigDict(env_prefix="QDRANT_")


class LLMConfig(BaseSettings):
    """Large Language Model configuration settings."""
    
    provider: str = Field(default="openai", description="LLM provider")
    model_name: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=1500, description="Maximum tokens")
    top_p: float = Field(default=0.9, description="Top-p sampling")
    api_key: Optional[str] = Field(default=None, description="API key")

    model_config = SettingsConfigDict(env_prefix="OPENAI_")


class EmbeddingsConfig(BaseSettings):
    """Embeddings configuration settings."""
    
    provider: str = Field(default="sentence_transformers", description="Embeddings provider")
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Model name")
    dimension: int = Field(default=384, description="Embedding dimension")
    normalize: bool = Field(default=True, description="Normalize embeddings")
    api_key: Optional[str] = Field(default=None, description="API key if needed")

    model_config = SettingsConfigDict(env_prefix="EMBEDDINGS_")


class AppConfig(BaseSettings):
    """Application configuration settings."""
    
    name: str = Field(default="RAG Agent System", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Environment")

    model_config = SettingsConfigDict(env_prefix="APP_")


class ConfigManager:
    """
    Centralized configuration manager that loads settings from YAML files
    and environment variables.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path or self._find_config_file()
        self._config = self._load_config()
        
        # Initialize typed configuration objects
        self.app = AppConfig()
        self.llm = LLMConfig()
        self.embeddings = EmbeddingsConfig()
        self.database = DatabaseConfig()
        
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        possible_paths = [
            "config/config.yaml",
            "../config/config.yaml",
            "../../config/config.yaml",
            os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError("Configuration file not found in standard locations")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config if config else {}
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation (e.g., 'llm.temperature')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary containing section configuration
        """
        return self._config.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_env(self) -> None:
        """Update configuration with environment variables."""
        # Load .env file if it exists
        env_path = Path(".env")
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
        
        # Update specific configurations
        self.app = AppConfig()
        self.llm = LLMConfig()
        self.embeddings = EmbeddingsConfig()
        self.database = DatabaseConfig()
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            path: Path to save configuration (defaults to current config path)
        """
        save_path = path or self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    def reload_config(self) -> None:
        """Reload configuration from file and environment."""
        self._config = self._load_config()
        self.update_from_env()
    
    def validate_config(self) -> Dict[str, str]:
        """
        Validate configuration and return any errors.
        
        Returns:
            Dictionary of validation errors
        """
        errors = {}
        
        # Validate required API keys based on providers
        if self.llm.provider == "openai" and not self.llm.api_key:
            errors["llm.api_key"] = "OpenAI API key is required"
        
        if self.embeddings.provider == "openai" and not self.embeddings.api_key:
            errors["embeddings.api_key"] = "OpenAI API key is required for embeddings"
        
        # Validate database configuration
        if not self.database.host and not self.database.url:
            errors["database"] = "Either host or url must be specified"
        
        # Validate file paths
        required_dirs = ["data/raw", "data/processed", "data/embeddings"]
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
        
        return errors
    
    def get_database_connection_string(self) -> str:
        """Get database connection string based on configuration."""
        if self.database.url:
            return self.database.url
        else:
            return f"http://{self.database.host}:{self.database.port}"
    
    def get_model_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get model configuration for specific agent type.
        
        Args:
            agent_type: Type of agent (llm, embeddings, etc.)
            
        Returns:
            Model configuration dictionary
        """
        if agent_type == "llm":
            return {
                "provider": self.llm.provider,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "top_p": self.llm.top_p,
                "api_key": self.llm.api_key
            }
        elif agent_type == "embeddings":
            return {
                "provider": self.embeddings.provider,
                "model_name": self.embeddings.model_name,
                "dimension": self.embeddings.dimension,
                "normalize": self.embeddings.normalize,
                "api_key": self.embeddings.api_key
            }
        else:
            return {}
    
    def __str__(self) -> str:
        """String representation of configuration (without sensitive data)."""
        safe_config = self._config.copy()
        
        # Remove sensitive information
        sensitive_keys = ["api_key", "token", "password", "secret"]
        
        def remove_sensitive(obj, keys):
            if isinstance(obj, dict):
                for key in list(obj.keys()):
                    if any(sensitive in key.lower() for sensitive in keys):
                        obj[key] = "***HIDDEN***"
                    else:
                        remove_sensitive(obj[key], keys)
        
        remove_sensitive(safe_config, sensitive_keys)
        return yaml.dump(safe_config, default_flow_style=False, indent=2)


# Global configuration instance
config = ConfigManager()