"""
semantic.py - GloVe Semantic Substrate

We inherit GloVe's pre-trained 300D semantic space.
We don't train anything - we use it as geometric substrate.

GloVe captures the structure of meaning:
- Similar words are close
- Analogies work: king - man + woman ≈ queen
- Concepts can be composed via vector algebra
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class SemanticPoint:
    """A point in semantic space."""
    vector: np.ndarray      # 300D GloVe vector
    text: str               # The text this represents
    words: List[str] = None # Component words
    
    def __post_init__(self):
        self.vector = self.vector.astype(np.float32)
        if self.words is None:
            self.words = self.text.lower().split()
    
    def cosine_similarity(self, other: 'SemanticPoint') -> float:
        """Cosine similarity to another point."""
        n1 = np.linalg.norm(self.vector)
        n2 = np.linalg.norm(other.vector)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (n1 * n2))
    
    def distance(self, other: 'SemanticPoint') -> float:
        """Euclidean distance to another point."""
        return float(np.linalg.norm(self.vector - other.vector))


class GloVeSubstrate:
    """
    The GloVe semantic substrate.
    
    This provides the 300D geometry of meaning.
    We don't train it - we inherit it and navigate it.
    """
    
    def __init__(self, model_name: str = "glove-wiki-gigaword-300"):
        """
        Load GloVe model.
        
        Args:
            model_name: Gensim model name
        """
        self.model_name = model_name
        self.model = None
        self.dimension = 300
        self._load_model()
    
    def _load_model(self):
        """Load the GloVe model via Gensim."""
        try:
            import gensim.downloader as api
            print(f"[GloVe] Loading {self.model_name}...")
            self.model = api.load(self.model_name)
            self.dimension = self.model.vector_size
            print(f"[GloVe] Loaded. Dimension: {self.dimension}, Vocab: {len(self.model)}")
        except Exception as e:
            print(f"[GloVe] Failed to load: {e}")
            print("[GloVe] Falling back to random vectors (for testing only)")
            self.model = None
    
    def word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a single word."""
        if self.model is None:
            # Fallback: deterministic random based on word
            np.random.seed(hash(word) % (2**32))
            return np.random.randn(self.dimension).astype(np.float32)
        
        word = word.lower().strip()
        if word in self.model:
            return self.model[word].astype(np.float32)
        return None
    
    def sentence_vector(self, sentence: str) -> np.ndarray:
        """
        Get vector for a sentence.
        
        Method: Mean of word vectors (simple but effective).
        """
        words = sentence.lower().split()
        vectors = []
        
        for word in words:
            # Clean punctuation
            word = ''.join(c for c in word if c.isalnum())
            if not word:
                continue
            
            vec = self.word_vector(word)
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            return np.zeros(self.dimension, dtype=np.float32)
        
        return np.mean(vectors, axis=0).astype(np.float32)
    
    def embed(self, text: str) -> SemanticPoint:
        """Embed text into semantic space."""
        vec = self.sentence_vector(text)
        return SemanticPoint(vector=vec, text=text)
    
    def embed_batch(self, texts: List[str]) -> List[SemanticPoint]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]
    
    def most_similar_words(self, word: str, n: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words."""
        if self.model is None:
            return []
        
        word = word.lower().strip()
        if word not in self.model:
            return []
        
        return self.model.most_similar(word, topn=n)
    
    def concept_algebra(
        self,
        positive: List[str],
        negative: List[str] = None
    ) -> np.ndarray:
        """
        Compose concepts via vector algebra.
        
        Example: concept_algebra(['king', 'woman'], ['man']) ≈ 'queen'
        
        Args:
            positive: Words to add
            negative: Words to subtract
            
        Returns:
            Resulting concept vector
        """
        result = np.zeros(self.dimension, dtype=np.float32)
        
        for word in positive:
            vec = self.word_vector(word)
            if vec is not None:
                result += vec
        
        if negative:
            for word in negative:
                vec = self.word_vector(word)
                if vec is not None:
                    result -= vec
        
        return result
    
    def find_nearest(
        self,
        vector: np.ndarray,
        candidates: List[SemanticPoint],
        n: int = 5
    ) -> List[Tuple[SemanticPoint, float]]:
        """
        Find nearest points to a vector.
        
        Args:
            vector: Query vector
            candidates: Points to search
            n: How many to return
            
        Returns:
            List of (point, similarity) pairs
        """
        query = SemanticPoint(vector=vector, text="[query]")
        
        scored = []
        for point in candidates:
            sim = query.cosine_similarity(point)
            scored.append((point, sim))
        
        scored.sort(key=lambda x: -x[1])
        return scored[:n]


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GloVe Semantic Substrate Test")
    print("=" * 60)
    
    substrate = GloVeSubstrate()
    
    # Test word vectors
    print("\nWord vectors:")
    for word in ["king", "queen", "man", "woman"]:
        vec = substrate.word_vector(word)
        if vec is not None:
            print(f"  {word}: norm={np.linalg.norm(vec):.4f}")
    
    # Test concept algebra
    print("\nConcept algebra (king - man + woman):")
    concept = substrate.concept_algebra(['king', 'woman'], ['man'])
    if substrate.model:
        similar = substrate.model.most_similar([concept], topn=5)
        for word, score in similar:
            print(f"  {word}: {score:.4f}")
    
    # Test sentence embedding
    print("\nSentence embeddings:")
    sentences = [
        "What is the capital of Mississippi?",
        "Jackson is the capital of Mississippi.",
        "The moon is made of cheese.",
    ]
    
    points = substrate.embed_batch(sentences)
    
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i < j:
                sim = p1.cosine_similarity(p2)
                print(f"  [{i}]↔[{j}]: sim={sim:.4f}")
    
    print("\n✓ GloVe substrate working")
