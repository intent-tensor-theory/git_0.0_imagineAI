"""
embeddings.py - Sentence Embedding Interface

Provides embedding functionality for imagineAI using various backends:
- Sentence-Transformers (recommended)
- GloVe/Word2Vec (simpler, faster)
- API-based (OpenAI, HuggingFace)

The embeddings become the Φ field substrate.
"""

import numpy as np
from typing import List, Union, Optional
from abc import ABC, abstractmethod


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends"""
    
    @abstractmethod
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text to vectors"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension"""
        pass


class SentenceTransformerBackend(EmbeddingBackend):
    """
    Sentence-Transformers backend (recommended).
    
    Models:
    - all-MiniLM-L6-v2: Fast, good quality (384 dim)
    - all-mpnet-base-v2: Better quality, slower (768 dim)
    - multi-qa-MiniLM-L6-cos-v1: Optimized for QA
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class GloVeBackend(EmbeddingBackend):
    """
    GloVe word vectors backend.
    
    Simpler than sentence transformers - averages word vectors.
    Good for testing and low-resource environments.
    """
    
    def __init__(self, model_name: str = "glove-wiki-gigaword-100"):
        """
        Args:
            model_name: Gensim model name
                - glove-wiki-gigaword-100 (100 dim, fast)
                - glove-wiki-gigaword-300 (300 dim, better)
                - word2vec-google-news-300 (300 dim)
        """
        try:
            import gensim.downloader as api
            self.wv = api.load(model_name)
            self._dimension = self.wv.vector_size
        except ImportError:
            raise ImportError("Install gensim: pip install gensim")
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            return self._encode_single(text)
        return np.array([self._encode_single(t) for t in text])
    
    def _encode_single(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vectors = []
        for w in words:
            # Clean punctuation
            w = w.strip(".,!?;:\"'()[]{}")
            if w in self.wv:
                vectors.append(self.wv[w])
        
        if vectors:
            return np.mean(vectors, axis=0).astype(np.float32)
        return np.zeros(self._dimension, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class HuggingFaceBackend(EmbeddingBackend):
    """
    HuggingFace Transformers backend.
    
    Uses any HuggingFace model for embeddings.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto"
    ):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            self.model.to(self.device)
            self.model.eval()
            
            # Get dimension from model config
            self._dimension = self.model.config.hidden_size
            
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        import torch
        
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over tokens
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy().astype(np.float32)
    
    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_backend(
    backend_type: str = "auto",
    model_name: Optional[str] = None
) -> EmbeddingBackend:
    """
    Get an embedding backend.
    
    Args:
        backend_type: "sentence-transformers", "glove", "huggingface", or "auto"
        model_name: Specific model name (optional)
        
    Returns:
        EmbeddingBackend instance
    """
    if backend_type == "auto":
        # Try sentence-transformers first, fall back to GloVe
        try:
            return SentenceTransformerBackend(model_name or "all-MiniLM-L6-v2")
        except ImportError:
            try:
                return GloVeBackend(model_name or "glove-wiki-gigaword-100")
            except ImportError:
                raise ImportError(
                    "No embedding backend available. Install one of:\n"
                    "  pip install sentence-transformers  # recommended\n"
                    "  pip install gensim  # simpler"
                )
    
    elif backend_type == "sentence-transformers":
        return SentenceTransformerBackend(model_name or "all-MiniLM-L6-v2")
    
    elif backend_type == "glove":
        return GloVeBackend(model_name or "glove-wiki-gigaword-100")
    
    elif backend_type == "huggingface":
        return HuggingFaceBackend(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def create_phi_field(
    backend_type: str = "auto",
    model_name: Optional[str] = None
):
    """
    Create a PhiField with the specified embedding backend.
    
    Args:
        backend_type: "sentence-transformers", "glove", "huggingface", or "auto"
        model_name: Specific model name (optional)
        
    Returns:
        PhiField configured with the embedding backend
    """
    from ..core.phi_field import PhiField
    
    backend = get_embedding_backend(backend_type, model_name)
    return PhiField(embedding_model=backend)
