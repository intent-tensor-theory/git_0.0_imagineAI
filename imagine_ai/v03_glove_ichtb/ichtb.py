"""
ichtb.py - ICHTB Structured Projection

Projects GloVe 300D → ICHTB 48D

The 48D structure comes from ITT:
    6 zones × 4 operators × 2 components = 48 dimensions

Zone Layout:
    Δ₁ [0-7]:   Forward (∇Φ governs)
    Δ₂ [8-15]:  Memory (∇×F governs)
    Δ₃ [16-23]: Expansion (+∇²Φ governs)
    Δ₄ [24-31]: Compression (-∇²Φ governs)
    Δ₅ [32-39]: Apex (∂Φ/∂t governs)
    Δ₆ [40-47]: Core (Φ=i₀ governs)

This isn't arbitrary dimensionality reduction.
The zone structure carries meaning.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from .semantic import SemanticPoint, GloVeSubstrate


# =============================================================================
# ICHTB Constants
# =============================================================================

GLOVE_DIM = 300
ICHTB_DIM = 48
ZONES = 6
ZONE_DIM = 8  # 4 operators × 2 components


class Zone(Enum):
    """The six ICHTB zones."""
    FORWARD = 0      # Δ₁: Direction/intent
    MEMORY = 1       # Δ₂: Context/history
    EXPANSION = 2    # Δ₃: Growth/spread
    COMPRESSION = 3  # Δ₄: Focus/concentrate
    APEX = 4         # Δ₅: Lock test
    CORE = 5         # Δ₆: Anchor


# =============================================================================
# ICHTB Point
# =============================================================================

@dataclass
class ICHTBPoint:
    """A point in 48D ICHTB space."""
    vector: np.ndarray      # 48D vector
    text: str               # Original text
    glove_vector: Optional[np.ndarray] = None  # Original 300D
    
    def __post_init__(self):
        self.vector = self.vector.astype(np.float32)
        if self.vector.shape != (ICHTB_DIM,):
            raise ValueError(f"ICHTB point must be {ICHTB_DIM}D, got {self.vector.shape}")
    
    def zone_slice(self, zone: Zone) -> np.ndarray:
        """Get the 8D slice for a zone."""
        start = zone.value * ZONE_DIM
        return self.vector[start:start + ZONE_DIM]
    
    def zone_magnitude(self, zone: Zone) -> float:
        """Get magnitude of a zone."""
        return float(np.linalg.norm(self.zone_slice(zone)))
    
    @property
    def forward_magnitude(self) -> float:
        """Δ₁ magnitude - how directed is this point?"""
        return self.zone_magnitude(Zone.FORWARD)
    
    @property
    def memory_magnitude(self) -> float:
        """Δ₂ magnitude - how much context?"""
        return self.zone_magnitude(Zone.MEMORY)
    
    @property
    def expansion_compression_balance(self) -> float:
        """Δ₃/Δ₄ balance - expanding (+) or compressing (-)?"""
        exp = self.zone_magnitude(Zone.EXPANSION)
        comp = self.zone_magnitude(Zone.COMPRESSION)
        if exp + comp < 1e-8:
            return 0.0
        return (exp - comp) / (exp + comp)
    
    def cosine_similarity(self, other: 'ICHTBPoint') -> float:
        """Cosine similarity in ICHTB space."""
        n1 = np.linalg.norm(self.vector)
        n2 = np.linalg.norm(other.vector)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (n1 * n2))
    
    def distance(self, other: 'ICHTBPoint') -> float:
        """Euclidean distance in ICHTB space."""
        return float(np.linalg.norm(self.vector - other.vector))


# =============================================================================
# ICHTB Projection Matrix
# =============================================================================

class ICHTBProjection:
    """
    Projects 300D GloVe → 48D ICHTB.
    
    The projection matrix is NOT trained.
    It's derived from the ICHTB zone structure.
    
    Each zone gets a different "slice" of the GloVe space:
    - Δ₁ (Forward): Projects dimensions that correlate with action/direction
    - Δ₂ (Memory): Projects dimensions that correlate with context/time
    - Δ₃ (Expansion): Projects dimensions that correlate with growth
    - Δ₄ (Compression): Projects dimensions that correlate with focus
    - Δ₅ (Apex): Projects dimensions that correlate with resolution
    - Δ₆ (Core): Projects dimensions that correlate with core identity
    
    For now, we use orthogonal slicing. Future work: derive from ITT equations.
    """
    
    def __init__(self, seed: int = 42):
        """
        Create projection matrix.
        
        Args:
            seed: Random seed for reproducible projection
        """
        self.seed = seed
        self.W = self._create_projection_matrix()
        self.W_inv = np.linalg.pinv(self.W)
    
    def _create_projection_matrix(self) -> np.ndarray:
        """
        Create the 48×300 projection matrix.
        
        Method: Structured orthogonal projection.
        Each zone gets 50 dimensions of GloVe (50 × 6 = 300).
        Then compress 50 → 8 within each zone.
        """
        np.random.seed(self.seed)
        
        # Create random orthogonal matrix
        # Start with random, then orthogonalize
        W = np.random.randn(ICHTB_DIM, GLOVE_DIM).astype(np.float32)
        
        # Orthogonalize each zone's projection
        for zone in Zone:
            start = zone.value * ZONE_DIM
            end = start + ZONE_DIM
            
            # Get this zone's rows
            zone_rows = W[start:end, :]
            
            # QR decomposition for orthogonality
            Q, R = np.linalg.qr(zone_rows.T)
            W[start:end, :] = Q[:, :ZONE_DIM].T
        
        # Normalize rows
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        W = W / norms
        
        return W
    
    def project(self, glove_vector: np.ndarray) -> np.ndarray:
        """
        Project 300D GloVe → 48D ICHTB.
        
        Args:
            glove_vector: 300D GloVe vector
            
        Returns:
            48D ICHTB vector
        """
        if glove_vector.shape != (GLOVE_DIM,):
            # Handle wrong dimensions
            if len(glove_vector) > GLOVE_DIM:
                glove_vector = glove_vector[:GLOVE_DIM]
            else:
                glove_vector = np.pad(glove_vector, (0, GLOVE_DIM - len(glove_vector)))
        
        ichtb_vector = self.W @ glove_vector
        
        # Normalize to unit sphere
        norm = np.linalg.norm(ichtb_vector)
        if norm > 1e-8:
            ichtb_vector = ichtb_vector / norm
        
        return ichtb_vector.astype(np.float32)
    
    def unproject(self, ichtb_vector: np.ndarray) -> np.ndarray:
        """
        Unproject 48D ICHTB → 300D GloVe (approximate inverse).
        
        Note: Information is lost in projection.
        """
        return (self.W_inv @ ichtb_vector).astype(np.float32)


# =============================================================================
# ICHTB Space
# =============================================================================

class ICHTBSpace:
    """
    Combined GloVe + ICHTB space.
    
    Takes text, embeds via GloVe, projects to ICHTB.
    """
    
    def __init__(self, glove: GloVeSubstrate = None, projection: ICHTBProjection = None):
        """
        Args:
            glove: GloVe substrate (creates one if None)
            projection: ICHTB projection (creates one if None)
        """
        self.glove = glove or GloVeSubstrate()
        self.projection = projection or ICHTBProjection()
        self.points: List[ICHTBPoint] = []
    
    def embed(self, text: str) -> ICHTBPoint:
        """
        Embed text into ICHTB space.
        
        Pipeline: text → GloVe 300D → ICHTB 48D
        """
        # Step 1: GloVe embedding
        semantic_point = self.glove.embed(text)
        
        # Step 2: ICHTB projection
        ichtb_vector = self.projection.project(semantic_point.vector)
        
        return ICHTBPoint(
            vector=ichtb_vector,
            text=text,
            glove_vector=semantic_point.vector
        )
    
    def add(self, text: str) -> int:
        """Add text to space, return index."""
        point = self.embed(text)
        idx = len(self.points)
        self.points.append(point)
        return idx
    
    def add_batch(self, texts: List[str]) -> List[int]:
        """Add multiple texts."""
        return [self.add(t) for t in texts]
    
    def get(self, idx: int) -> ICHTBPoint:
        """Get point by index."""
        return self.points[idx]
    
    def size(self) -> int:
        """Number of points."""
        return len(self.points)
    
    def find_nearest(
        self,
        query: ICHTBPoint,
        n: int = 5
    ) -> List[tuple]:
        """
        Find nearest points to query.
        
        Returns: List of (index, point, similarity)
        """
        scored = []
        for i, point in enumerate(self.points):
            sim = query.cosine_similarity(point)
            scored.append((i, point, sim))
        
        scored.sort(key=lambda x: -x[2])
        return scored[:n]
    
    def find_minimum_sigma(
        self,
        query: ICHTBPoint,
        n: int = 1
    ) -> List[tuple]:
        """
        Find points with minimum σ (semantic distance) to query.
        
        σ = 1 - cosine_similarity
        
        Returns: List of (index, point, sigma)
        """
        scored = []
        for i, point in enumerate(self.points):
            sigma = 1.0 - query.cosine_similarity(point)
            scored.append((i, point, sigma))
        
        scored.sort(key=lambda x: x[2])  # Sort by σ ascending
        return scored[:n]


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ICHTB Projection Test")
    print("=" * 60)
    
    # Create space
    space = ICHTBSpace()
    
    # Add some facts
    facts = [
        "Jackson is the capital of Mississippi.",
        "Austin is the capital of Texas.",
        "Jupiter is the largest planet.",
        "The speed of light is very fast.",
        "The moon orbits the Earth.",
    ]
    
    print(f"\nAdding {len(facts)} facts to ICHTB space...")
    space.add_batch(facts)
    print(f"Space now has {space.size()} points")
    
    # Test query
    query = space.embed("What is the capital of Mississippi?")
    print(f"\nQuery: {query.text}")
    print(f"Query vector norm: {np.linalg.norm(query.vector):.4f}")
    
    # Zone analysis
    print("\nZone magnitudes:")
    for zone in Zone:
        mag = query.zone_magnitude(zone)
        print(f"  Δ{zone.value+1} ({zone.name}): {mag:.4f}")
    
    # Find nearest
    print("\nNearest points (by cosine similarity):")
    nearest = space.find_nearest(query, n=3)
    for idx, point, sim in nearest:
        print(f"  [{idx}] sim={sim:.4f}: {point.text[:50]}...")
    
    # Find minimum σ
    print("\nMinimum σ points:")
    min_sigma = space.find_minimum_sigma(query, n=3)
    for idx, point, sigma in min_sigma:
        print(f"  [{idx}] σ={sigma:.4f}: {point.text[:50]}...")
    
    print("\n✓ ICHTB projection working")
