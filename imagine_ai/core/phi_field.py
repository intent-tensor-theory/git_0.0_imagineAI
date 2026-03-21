"""
phi_field.py - The Semantic Substrate

The Φ field in imagineAI is the embedding space where meaning lives.
Each word, sentence, or concept is a point in this high-dimensional space.

From ITT: Φ = A·e^{iθ} where A is amplitude and θ is phase.
In embeddings: A = ||v||, θ = angle in embedding space.
"""

import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class PhiState:
    """
    A state in the Φ field.
    
    Attributes:
        vector: The embedding vector (the "position" in semantic space)
        text: The original text that was embedded
        amplitude: ||vector|| - magnitude of the embedding
        phase: Normalized direction vector (unit sphere position)
    """
    vector: np.ndarray
    text: str
    
    @property
    def amplitude(self) -> float:
        """A in Φ = A·e^{iθ} - how much structure exists"""
        return np.linalg.norm(self.vector)
    
    @property
    def phase(self) -> np.ndarray:
        """θ direction - normalized position on unit hypersphere"""
        norm = self.amplitude
        if norm > 0:
            return self.vector / norm
        return self.vector
    
    def distance_to(self, other: 'PhiState') -> float:
        """Cosine distance to another state"""
        dot = np.dot(self.vector, other.vector)
        norms = self.amplitude * other.amplitude
        if norms > 0:
            return 1.0 - (dot / norms)
        return 1.0
    
    def similarity_to(self, other: 'PhiState') -> float:
        """Cosine similarity to another state"""
        return 1.0 - self.distance_to(other)


class PhiField:
    """
    The embedding substrate for imagineAI.
    
    This is the Φ field from ITT - the scalar potential that holds semantic meaning.
    Currently uses pre-trained embeddings (Path A). 
    Future: Generate Φ from recursive collapse (Path B).
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize the Φ field.
        
        Args:
            embedding_model: A model with .encode() method (sentence-transformers style)
                            If None, will use a simple word-vector approach.
        """
        self.model = embedding_model
        self.dimension = None
        self._cache = {}  # Cache embeddings for efficiency
        
    def set_model(self, model):
        """Set the embedding model after initialization"""
        self.model = model
        self.dimension = None
        self._cache = {}
        
    def embed(self, text: Union[str, List[str]]) -> Union[PhiState, List[PhiState]]:
        """
        Convert text to Φ field state(s).
        
        Args:
            text: Single string or list of strings
            
        Returns:
            PhiState or list of PhiStates
        """
        if isinstance(text, str):
            return self._embed_single(text)
        return [self._embed_single(t) for t in text]
    
    def _embed_single(self, text: str) -> PhiState:
        """Embed a single text string"""
        # Check cache
        if text in self._cache:
            return self._cache[text]
        
        if self.model is None:
            raise ValueError("No embedding model set. Call set_model() first.")
        
        # Get embedding from model
        vector = self.model.encode(text)
        if hasattr(vector, 'numpy'):
            vector = vector.numpy()
        vector = np.array(vector, dtype=np.float32)
        
        # Set dimension on first embed
        if self.dimension is None:
            self.dimension = len(vector)
        
        state = PhiState(vector=vector, text=text)
        self._cache[text] = state
        return state
    
    def i_zero(self) -> PhiState:
        """
        The imaginary anchor point i₀.
        
        In ITT, i₀ is the pre-emergence seed - a point that exists but
        cannot be reached by real field values. In embeddings, we represent
        this as the zero vector (null semantic content).
        
        The field can approach i₀ but never equal it - there's always
        some semantic content in any real text.
        """
        if self.dimension is None:
            raise ValueError("Dimension unknown. Embed something first.")
        return PhiState(
            vector=np.zeros(self.dimension, dtype=np.float32),
            text="[i₀]"  # The imaginary null state
        )
    
    def distance_matrix(self, states: List[PhiState]) -> np.ndarray:
        """
        Compute pairwise distances between all states.
        
        Returns NxN matrix where M[i,j] = distance(states[i], states[j])
        """
        n = len(states)
        vectors = np.array([s.vector for s in states])
        
        # Compute all pairwise cosine distances
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normalized = vectors / norms
        
        similarities = normalized @ normalized.T
        distances = 1.0 - similarities
        return distances
    
    def centroid(self, states: List[PhiState]) -> PhiState:
        """
        Compute the centroid (mean) of multiple states.
        
        This is useful for finding the "center" of a conversation
        or the average meaning of multiple concepts.
        """
        vectors = np.array([s.vector for s in states])
        mean_vector = np.mean(vectors, axis=0)
        
        # Create text representation
        texts = [s.text[:20] for s in states[:3]]
        text = f"centroid({', '.join(texts)}{'...' if len(states) > 3 else ''})"
        
        return PhiState(vector=mean_vector, text=text)
    
    def interpolate(self, state1: PhiState, state2: PhiState, t: float) -> PhiState:
        """
        Linear interpolation between two states.
        
        Args:
            state1: Starting state
            state2: Ending state
            t: Interpolation factor (0 = state1, 1 = state2)
            
        Returns:
            Interpolated PhiState
        """
        vector = (1 - t) * state1.vector + t * state2.vector
        text = f"interp({state1.text[:15]}, {state2.text[:15]}, {t:.2f})"
        return PhiState(vector=vector, text=text)
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache = {}


# Convenience function for quick testing
def create_simple_field():
    """
    Create a simple PhiField using word vectors.
    Useful for testing without loading large models.
    """
    try:
        import gensim.downloader as api
        word_vectors = api.load("glove-wiki-gigaword-100")  # 100-dim for speed
        
        class SimpleEmbedder:
            def __init__(self, wv):
                self.wv = wv
                
            def encode(self, text):
                words = text.lower().split()
                vectors = []
                for w in words:
                    if w in self.wv:
                        vectors.append(self.wv[w])
                if vectors:
                    return np.mean(vectors, axis=0)
                return np.zeros(self.wv.vector_size)
        
        field = PhiField(SimpleEmbedder(word_vectors))
        return field
        
    except ImportError:
        raise ImportError("Install gensim: pip install gensim")
