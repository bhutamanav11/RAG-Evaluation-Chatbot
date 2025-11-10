import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)  # Optional
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Required
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Required
    
    # Database
    DATABASE_URL = "sqlite:///./rag_evaluation.db"
    CHROMA_PERSIST_DIR = "./chroma_db"
    
    # Model Configurations - FREE TIER FRIENDLY
    MODELS_CONFIG = {
        "gemini-pro": {  # Working model name
            "provider": "google",
            "max_tokens": 8192,
            "temperature": 0.7,
            "cost_per_1k_tokens": 0.0005,  # Very cheap!
            "enabled": True
        },
        "claude-3-haiku-20240307": {  # Full model name
            "provider": "anthropic",
            "max_tokens": 4096,
            "temperature": 0.7,
            "cost_per_1k_tokens": 0.00025,  # Very cheap!
            "enabled": True
        },
        "gpt-3.5-turbo": {
            "provider": "openai",
            "max_tokens": 4096,
            "temperature": 0.7,
            "cost_per_1k_tokens": 0.002,
            "enabled": True  # Disabled by default - enable after testing
        },
        "gpt-4": {
            "provider": "openai", 
            "max_tokens": 8192,
            "temperature": 0.7,
            "cost_per_1k_tokens": 0.03,
            "enabled": False  # Disabled by default - expensive
        }
    }
    
    # Get only enabled models
    @classmethod
    def get_enabled_models(cls):
        """Return list of enabled model names"""
        return [name for name, config in cls.MODELS_CONFIG.items() if config.get('enabled', True)]
    
    # Chunking Configuration - IMPROVED
    CHUNK_SIZE = 250  # Reduced from 1000 to 250 (200-300 tokens)
    CHUNK_OVERLAP = 40  # 15-20% overlap
    CHUNK_SIZES = [200, 250, 300, 350]  # For testing different sizes
    
    # Embedding Model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Evaluation Settings
    TOP_K_RETRIEVAL = 30
    MAX_CONVERSATION_TURNS = 25
    EVALUATION_BATCH_SIZE = 10