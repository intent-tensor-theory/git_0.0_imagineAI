"""
filament.py - Semantic Filaments

A sentence is not a POINT. It's a PATH through semantic space.
The GRADIENTS between words encode how meaning FLOWS.

This is the core insight:
    Meaning is in the TRANSITIONS, not the POSITIONS.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import re


@dataclass
class Filament:
    """
    A semantic filament - a path through GloVe space.
    
    Attributes:
        text: Original sentence
        words: Tokenized words
        vectors: GloVe vectors for each word (N × 300)
        gradients: Gradient tensor (N-1 × 300)
        gradient_magnitudes: |∇ᵢ| for each gradient
    """
    text: str
    words: List[str]
    vectors: np.ndarray          # Shape: (N, 300)
    gradients: np.ndarray        # Shape: (N-1, 300)
    gradient_magnitudes: np.ndarray  # Shape: (N-1,)
    
    @property
    def length(self) -> int:
        """Number of words."""
        return len(self.words)
    
    @property
    def num_gradients(self) -> int:
        """Number of gradients (transitions)."""
        return len(self.gradients)
    
    @property
    def total_path_length(self) -> float:
        """Total distance traveled through semantic space."""
        return float(np.sum(self.gradient_magnitudes))
    
    @property
    def mean_gradient_magnitude(self) -> float:
        """Average gradient magnitude."""
        if self.num_gradients == 0:
            return 0.0
        return float(np.mean(self.gradient_magnitudes))
    
    @property 
    def curvature(self) -> float:
        """
        Total curvature of the path.
        
        Curvature = sum of angle changes between consecutive gradients.
        High curvature = lots of direction changes = complex meaning structure.
        """
        if self.num_gradients < 2:
            return 0.0
        
        total_curve = 0.0
        for i in range(len(self.gradients) - 1):
            g1 = self.gradients[i]
            g2 = self.gradients[i + 1]
            
            n1 = np.linalg.norm(g1)
            n2 = np.linalg.norm(g2)
            
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            
            # Cosine of angle between consecutive gradients
            cos_angle = np.dot(g1, g2) / (n1 * n2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Curvature contribution: 1 - cos(θ) 
            # = 0 for straight, 2 for reversal
            total_curve += (1.0 - cos_angle)
        
        return float(total_curve)
    
    def gradient_cosine(self, i: int, j: int) -> float:
        """Cosine similarity between gradients i and j."""
        g1 = self.gradients[i]
        g2 = self.gradients[j]
        
        n1 = np.linalg.norm(g1)
        n2 = np.linalg.norm(g2)
        
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        
        return float(np.dot(g1, g2) / (n1 * n2))


class FilamentFactory:
    """
    Creates filaments from text using GloVe embeddings.
    """
    
    def __init__(self, glove_model):
        """
        Args:
            glove_model: Gensim KeyedVectors (GloVe)
        """
        self.glove = glove_model
        self.dim = glove_model.vector_size if glove_model else 300
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.
        
        Lowercase, split on whitespace, remove punctuation.
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation but keep words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split and filter empty
        words = [w.strip() for w in text.split() if w.strip()]
        
        return words
    
    def word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get GloVe vector for a word."""
        if self.glove is None:
            # Fallback: deterministic random
            np.random.seed(hash(word) % (2**32))
            return np.random.randn(self.dim).astype(np.float32)
        
        if word in self.glove:
            return self.glove[word].astype(np.float32)
        
        return None
    
    def create(self, text: str) -> Filament:
        """
        Create a filament from text.
        
        Pipeline:
        1. Tokenize
        2. Get GloVe vectors
        3. Compute gradients
        """
        words = self.tokenize(text)
        
        # Get vectors (skip words not in vocabulary)
        valid_words = []
        vectors = []
        
        for word in words:
            vec = self.word_vector(word)
            if vec is not None:
                valid_words.append(word)
                vectors.append(vec)
        
        if len(vectors) == 0:
            # Empty filament
            return Filament(
                text=text,
                words=[],
                vectors=np.zeros((0, self.dim), dtype=np.float32),
                gradients=np.zeros((0, self.dim), dtype=np.float32),
                gradient_magnitudes=np.zeros(0, dtype=np.float32)
            )
        
        vectors = np.array(vectors, dtype=np.float32)
        
        # Compute gradients: ∇ᵢ = vᵢ₊₁ - vᵢ
        if len(vectors) > 1:
            gradients = np.diff(vectors, axis=0)
            magnitudes = np.linalg.norm(gradients, axis=1)
        else:
            gradients = np.zeros((0, self.dim), dtype=np.float32)
            magnitudes = np.zeros(0, dtype=np.float32)
        
        return Filament(
            text=text,
            words=valid_words,
            vectors=vectors,
            gradients=gradients,
            gradient_magnitudes=magnitudes
        )
    
    def create_batch(self, texts: List[str]) -> List[Filament]:
        """Create multiple filaments."""
        return [self.create(t) for t in texts]


# =============================================================================
# Filament Comparison (without DTW - simple version)
# =============================================================================

def gradient_overlap(f1: Filament, f2: Filament, threshold: float = 0.7) -> int:
    """
    Count how many gradient pairs are similar.
    
    This is a simple matching before we implement full DTW.
    """
    if f1.num_gradients == 0 or f2.num_gradients == 0:
        return 0
    
    count = 0
    for i, g1 in enumerate(f1.gradients):
        n1 = np.linalg.norm(g1)
        if n1 < 1e-8:
            continue
        
        for j, g2 in enumerate(f2.gradients):
            n2 = np.linalg.norm(g2)
            if n2 < 1e-8:
                continue
            
            cos = np.dot(g1, g2) / (n1 * n2)
            if cos >= threshold:
                count += 1
                break  # Count each g1 at most once
    
    return count


def filament_similarity_simple(f1: Filament, f2: Filament) -> float:
    """
    Simple filament similarity based on gradient overlap.
    
    Returns value in [0, 1].
    """
    if f1.num_gradients == 0 or f2.num_gradients == 0:
        return 0.0
    
    overlap = gradient_overlap(f1, f2, threshold=0.6)
    max_possible = min(f1.num_gradients, f2.num_gradients)
    
    return overlap / max_possible if max_possible > 0 else 0.0


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Semantic Filament Test")
    print("=" * 60)
    
    # Load GloVe
    try:
        import gensim.downloader as api
        print("\n[GloVe] Loading...")
        glove = api.load("glove-wiki-gigaword-300")
        print(f"[GloVe] Loaded. Vocab: {len(glove)}")
    except Exception as e:
        print(f"[GloVe] Failed: {e}")
        glove = None
    
    factory = FilamentFactory(glove)
    
    # Test filaments
    sentences = [
        "What is the capital of Mississippi?",
        "Jackson is the capital of Mississippi.",
        "Jupiter is the largest planet.",
        "How fast does light travel?",
        "The speed of light is very fast.",
    ]
    
    filaments = factory.create_batch(sentences)
    
    print("\nFilament Properties:")
    for f in filaments:
        print(f"\n'{f.text}'")
        print(f"  Words: {f.words}")
        print(f"  Length: {f.length}, Gradients: {f.num_gradients}")
        print(f"  Path length: {f.total_path_length:.2f}")
        print(f"  Curvature: {f.curvature:.4f}")
    
    print("\nGradient Overlap (threshold=0.6):")
    for i, f1 in enumerate(filaments):
        for j, f2 in enumerate(filaments):
            if i < j:
                overlap = gradient_overlap(f1, f2, threshold=0.6)
                sim = filament_similarity_simple(f1, f2)
                print(f"  [{i}]↔[{j}]: overlap={overlap}, sim={sim:.3f}")
    
    print("\n✓ Filament creation working")
