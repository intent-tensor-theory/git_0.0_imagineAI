"""
dtw.py - Dynamic Time Warping for Semantic Filaments

DTW aligns two sequences of different lengths optimally.
This is NOT a neural network. It's pure dynamic programming.

For filaments:
- Input: Two gradient sequences
- Output: Alignment cost (lower = more similar)

This respects SEQUENTIAL STRUCTURE that averaging destroys.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .filament import Filament


@dataclass
class DTWResult:
    """Result of DTW alignment."""
    distance: float                          # Total DTW distance
    normalized_distance: float               # Distance / alignment length
    path: List[Tuple[int, int]]             # Optimal alignment path
    cost_matrix: Optional[np.ndarray] = None # Full cost matrix (if requested)


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine distance between two vectors.
    
    Returns value in [0, 2]:
    - 0 = identical direction
    - 1 = perpendicular
    - 2 = opposite direction
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    if n1 < 1e-8 or n2 < 1e-8:
        return 1.0  # Undefined → perpendicular
    
    cos = np.dot(v1, v2) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    
    return 1.0 - cos


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Euclidean distance."""
    return float(np.linalg.norm(v1 - v2))


def dtw_gradients(
    seq1: np.ndarray,
    seq2: np.ndarray,
    distance_fn: callable = cosine_distance,
    return_path: bool = True
) -> DTWResult:
    """
    Dynamic Time Warping on gradient sequences.
    
    Args:
        seq1: First gradient sequence (N1 × D)
        seq2: Second gradient sequence (N2 × D)
        distance_fn: Distance function for vectors
        return_path: Whether to compute alignment path
        
    Returns:
        DTWResult with distance and optional path
    """
    n1, n2 = len(seq1), len(seq2)
    
    # Handle edge cases
    if n1 == 0 and n2 == 0:
        return DTWResult(distance=0.0, normalized_distance=0.0, path=[])
    if n1 == 0:
        return DTWResult(distance=float('inf'), normalized_distance=float('inf'), path=[])
    if n2 == 0:
        return DTWResult(distance=float('inf'), normalized_distance=float('inf'), path=[])
    
    # Cost matrix
    cost = np.full((n1 + 1, n2 + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    
    # Fill cost matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            d = distance_fn(seq1[i-1], seq2[j-1])
            cost[i, j] = d + min(
                cost[i-1, j],     # Insertion
                cost[i, j-1],     # Deletion
                cost[i-1, j-1]    # Match
            )
    
    total_distance = cost[n1, n2]
    
    # Normalize by path length
    path_length = n1 + n2  # Approximate
    normalized = total_distance / path_length if path_length > 0 else 0.0
    
    # Backtrack to find path
    path = []
    if return_path:
        i, j = n1, n2
        while i > 0 or j > 0:
            path.append((i-1, j-1))
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Find which direction we came from
                candidates = [
                    (cost[i-1, j-1], i-1, j-1),  # Diagonal
                    (cost[i-1, j], i-1, j),      # Up
                    (cost[i, j-1], i, j-1),      # Left
                ]
                _, i, j = min(candidates, key=lambda x: x[0])
        
        path.reverse()
        # Filter out invalid indices
        path = [(i, j) for i, j in path if i >= 0 and j >= 0 and i < n1 and j < n2]
    
    return DTWResult(
        distance=float(total_distance),
        normalized_distance=float(normalized),
        path=path
    )


def filament_dtw_distance(
    f1: Filament,
    f2: Filament,
    use_cosine: bool = True
) -> float:
    """
    DTW distance between two filaments.
    
    Uses gradient sequences for comparison.
    
    Args:
        f1: First filament
        f2: Second filament
        use_cosine: Use cosine distance (True) or Euclidean (False)
        
    Returns:
        Normalized DTW distance
    """
    if f1.num_gradients == 0 or f2.num_gradients == 0:
        return float('inf')
    
    dist_fn = cosine_distance if use_cosine else euclidean_distance
    
    result = dtw_gradients(f1.gradients, f2.gradients, dist_fn, return_path=False)
    
    return result.normalized_distance


def filament_sigma(f1: Filament, f2: Filament) -> float:
    """
    σ (semantic distance) between filaments.
    
    This is the core metric for imagineAI v0.4.
    
    σ = DTW_normalized(f1.gradients, f2.gradients)
    
    Lower σ = more similar gradient flow = more related semantically.
    """
    return filament_dtw_distance(f1, f2, use_cosine=True)


# =============================================================================
# Batch Operations
# =============================================================================

def find_minimum_sigma(
    query: Filament,
    candidates: List[Filament],
    n: int = 5
) -> List[Tuple[int, Filament, float]]:
    """
    Find candidates with minimum σ to query.
    
    Args:
        query: Query filament
        candidates: List of candidate filaments
        n: How many to return
        
    Returns:
        List of (index, filament, sigma) sorted by σ ascending
    """
    scored = []
    
    for i, cand in enumerate(candidates):
        sigma = filament_sigma(query, cand)
        scored.append((i, cand, sigma))
    
    scored.sort(key=lambda x: x[2])
    
    return scored[:n]


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DTW for Filaments Test")
    print("=" * 60)
    
    # Create simple test sequences
    np.random.seed(42)
    
    # Sequence 1: [a, b, c, d]
    seq1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ], dtype=np.float32)
    
    # Sequence 2: [a, a, b, c, d] (same but with repeat)
    seq2 = np.array([
        [1, 0, 0],
        [1, 0, 0],  # Repeat
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ], dtype=np.float32)
    
    # Sequence 3: [x, y, z] (different)
    seq3 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ], dtype=np.float32)
    
    print("\nDTW distances:")
    
    r12 = dtw_gradients(seq1, seq2)
    r13 = dtw_gradients(seq1, seq3)
    r23 = dtw_gradients(seq2, seq3)
    
    print(f"  seq1 ↔ seq2 (similar): {r12.normalized_distance:.4f}")
    print(f"  seq1 ↔ seq3 (different): {r13.normalized_distance:.4f}")
    print(f"  seq2 ↔ seq3 (different): {r23.normalized_distance:.4f}")
    
    print(f"\nAlignment path seq1↔seq2: {r12.path[:5]}...")
    
    # Test with real filaments
    print("\n--- Real Filament Test ---")
    
    try:
        import gensim.downloader as api
        from .filament import FilamentFactory
        
        print("[GloVe] Loading...")
        glove = api.load("glove-wiki-gigaword-300")
        
        factory = FilamentFactory(glove)
        
        f_q = factory.create("What is the capital of Mississippi?")
        f_a = factory.create("Jackson is the capital of Mississippi.")
        f_x = factory.create("Jupiter is the largest planet.")
        
        sigma_qa = filament_sigma(f_q, f_a)
        sigma_qx = filament_sigma(f_q, f_x)
        
        print(f"\nσ(Q, correct_answer): {sigma_qa:.4f}")
        print(f"σ(Q, wrong_answer):   {sigma_qx:.4f}")
        print(f"Correct has lower σ: {sigma_qa < sigma_qx}")
        
    except Exception as e:
        print(f"Skipping real test: {e}")
    
    print("\n✓ DTW working")
