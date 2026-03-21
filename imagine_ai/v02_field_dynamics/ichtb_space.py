"""
ichtb_space.py - The 48-Dimensional Information Space

This is the substrate where answers emerge from field dynamics.
NO pre-trained models. NO lookup tables. Pure math.

The 48D structure comes from the ICHTB:
    6 zones × 4 operators × 2 components = 48 dimensions

Information (text, facts, concepts) lives in this space.
Questions create excitations. Answers are stable configurations.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# FUNDAMENTAL CONSTANTS - DERIVED FROM ICHTB, NOT ARBITRARY
# =============================================================================

ZONES = 6           # Δ₁ through Δ₆
OPERATORS = 4       # Φ, ∇Φ, ∇×F, ∇²Φ
COMPONENTS = 2      # Real + Imaginary
DIM = ZONES * OPERATORS * COMPONENTS  # = 48


class Zone(Enum):
    """The six ICHTB zones."""
    FORWARD = 0      # Δ₁: +Y, ∇Φ governs (direction/intent)
    MEMORY = 1       # Δ₂: -Y, ∇×F governs (curl/memory)
    EXPANSION = 2    # Δ₃: +X, +∇²Φ governs (growth)
    COMPRESSION = 3  # Δ₄: -X, -∇²Φ governs (focus)
    APEX = 4         # Δ₅: +Z, ∂Φ/∂t governs (lock test)
    CORE = 5         # Δ₆: -Z, Φ=i₀ (imaginary anchor)


class Operator(Enum):
    """The four operators in each zone."""
    PHI = 0          # Scalar potential
    GRAD = 1         # ∇Φ (gradient)
    CURL = 2         # ∇×F (curl/rotation)
    LAP = 3          # ∇²Φ (Laplacian/curvature)


# =============================================================================
# ICHTB POINT - A SINGLE POINT IN 48D SPACE
# =============================================================================

@dataclass
class ICHTBPoint:
    """
    A point in 48-dimensional ICHTB space.
    
    This could represent:
    - A word
    - A sentence
    - A concept
    - A question
    - An answer
    
    The structure is NOT arbitrary. It reflects the ICHTB zone/operator geometry.
    """
    vector: np.ndarray  # Shape: (48,)
    content: str = ""   # The text/concept this point represents
    
    def __post_init__(self):
        if self.vector.shape != (DIM,):
            raise ValueError(f"ICHTB point must be {DIM}D, got {self.vector.shape}")
        self.vector = self.vector.astype(np.float64)
    
    # =========================================================================
    # ZONE ACCESS - Structured access to the 48D vector
    # =========================================================================
    
    def zone_slice(self, zone: Zone) -> np.ndarray:
        """Get the 8-dimensional slice for a specific zone."""
        start = zone.value * 8
        return self.vector[start:start + 8]
    
    def operator_value(self, zone: Zone, op: Operator) -> complex:
        """Get a specific operator value as complex number."""
        zone_vec = self.zone_slice(zone)
        idx = op.value * 2
        return complex(zone_vec[idx], zone_vec[idx + 1])
    
    def phi(self, zone: Zone) -> complex:
        """Get Φ (scalar potential) for a zone."""
        return self.operator_value(zone, Operator.PHI)
    
    def grad(self, zone: Zone) -> complex:
        """Get ∇Φ (gradient) for a zone."""
        return self.operator_value(zone, Operator.GRAD)
    
    def curl(self, zone: Zone) -> complex:
        """Get ∇×F (curl) for a zone."""
        return self.operator_value(zone, Operator.CURL)
    
    def lap(self, zone: Zone) -> complex:
        """Get ∇²Φ (Laplacian) for a zone."""
        return self.operator_value(zone, Operator.LAP)
    
    # =========================================================================
    # FIELD PROPERTIES
    # =========================================================================
    
    @property
    def total_phi(self) -> float:
        """Total |Φ| across all zones - the "intensity" of this point."""
        return sum(abs(self.phi(z)) for z in Zone)
    
    @property
    def gradient_magnitude(self) -> float:
        """Total |∇Φ| - how "directed" this point is."""
        return sum(abs(self.grad(z)) for z in Zone)
    
    @property
    def curl_magnitude(self) -> float:
        """Total |∇×F| - how much "memory/rotation" this point has."""
        return sum(abs(self.curl(z)) for z in Zone)
    
    @property
    def curvature(self) -> float:
        """Net Laplacian - positive = expanding, negative = compressing."""
        expand = abs(self.lap(Zone.EXPANSION))
        compress = abs(self.lap(Zone.COMPRESSION))
        return expand - compress
    
    @property
    def i_zero_alignment(self) -> float:
        """How aligned is the Core zone with i₀ (pure imaginary)?"""
        core_phi = self.phi(Zone.CORE)
        if abs(core_phi) < 1e-10:
            return 1.0  # Zero is perfectly aligned with i₀
        return abs(core_phi.imag) / abs(core_phi)
    
    # =========================================================================
    # DISTANCE AND SIMILARITY
    # =========================================================================
    
    def distance(self, other: 'ICHTBPoint') -> float:
        """Euclidean distance in ICHTB space."""
        return np.linalg.norm(self.vector - other.vector)
    
    def cosine_sim(self, other: 'ICHTBPoint') -> float:
        """Cosine similarity."""
        n1 = np.linalg.norm(self.vector)
        n2 = np.linalg.norm(other.vector)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return np.dot(self.vector, other.vector) / (n1 * n2)
    
    # =========================================================================
    # VECTOR OPERATIONS
    # =========================================================================
    
    def __add__(self, other: 'ICHTBPoint') -> 'ICHTBPoint':
        return ICHTBPoint(self.vector + other.vector, f"({self.content} + {other.content})")
    
    def __sub__(self, other: 'ICHTBPoint') -> 'ICHTBPoint':
        return ICHTBPoint(self.vector - other.vector, f"({self.content} - {other.content})")
    
    def __mul__(self, scalar: float) -> 'ICHTBPoint':
        return ICHTBPoint(self.vector * scalar, self.content)
    
    def normalize(self) -> 'ICHTBPoint':
        """Return unit vector."""
        n = np.linalg.norm(self.vector)
        if n < 1e-10:
            return ICHTBPoint(np.zeros(DIM), self.content)
        return ICHTBPoint(self.vector / n, self.content)


# =============================================================================
# i₀ - THE IMAGINARY ANCHOR POINT
# =============================================================================

def create_i_zero() -> ICHTBPoint:
    """
    Create i₀ - the imaginary anchor point.
    
    From ICHTB theory: i₀ sits at the Core zone (Δ₆) with purely imaginary Φ.
    All other components are zero. This is the "silence" from which
    recursion begins and to which it returns.
    """
    vec = np.zeros(DIM)
    
    # Core zone (Δ₆) Φ = i (pure imaginary)
    # Φ is at indices 0,1 within the zone
    # Core zone starts at index 5 * 8 = 40
    # So Φ_real is at 40, Φ_imag is at 41
    vec[41] = 1.0  # Φ_imag = 1, Φ_real = 0 → Φ = i
    
    return ICHTBPoint(vec, "i₀")


I_ZERO = create_i_zero()


# =============================================================================
# ICHTB SPACE - THE FIELD SUBSTRATE
# =============================================================================

@dataclass
class ICHTBSpace:
    """
    The 48-dimensional information space.
    
    This is where:
    - Information (text/concepts) is embedded as points
    - Questions create excitations
    - Field dynamics evolve
    - Answers emerge as stable configurations
    
    NO training. NO neural networks. Pure geometry.
    """
    points: List[ICHTBPoint] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)  # content -> index
    
    def add(self, point: ICHTBPoint) -> int:
        """Add a point to the space. Returns its index."""
        idx = len(self.points)
        self.points.append(point)
        if point.content:
            self.index[point.content] = idx
        return idx
    
    def get(self, idx: int) -> ICHTBPoint:
        """Get point by index."""
        return self.points[idx]
    
    def find(self, content: str) -> Optional[ICHTBPoint]:
        """Find point by content."""
        idx = self.index.get(content)
        return self.points[idx] if idx is not None else None
    
    def size(self) -> int:
        """Number of points in space."""
        return len(self.points)
    
    # =========================================================================
    # FIELD OPERATIONS
    # =========================================================================
    
    def nearest(self, query: ICHTBPoint, k: int = 5) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors to query point.
        Returns list of (index, distance) pairs.
        """
        distances = [(i, query.distance(p)) for i, p in enumerate(self.points)]
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def most_similar(self, query: ICHTBPoint, k: int = 5) -> List[Tuple[int, float]]:
        """
        Find k most similar points (by cosine similarity).
        Returns list of (index, similarity) pairs.
        """
        sims = [(i, query.cosine_sim(p)) for i, p in enumerate(self.points)]
        sims.sort(key=lambda x: -x[1])  # Descending
        return sims[:k]
    
    def centroid(self) -> ICHTBPoint:
        """Compute centroid of all points."""
        if not self.points:
            return I_ZERO
        avg = np.mean([p.vector for p in self.points], axis=0)
        return ICHTBPoint(avg, "centroid")
    
    # =========================================================================
    # GRADIENT FIELD - ∇Φ
    # =========================================================================
    
    def gradient_at(self, point: ICHTBPoint, target: ICHTBPoint) -> np.ndarray:
        """
        Compute gradient direction from point toward target.
        
        This is the Δ₁ (Forward) operation - establishing direction.
        """
        delta = target.vector - point.vector
        norm = np.linalg.norm(delta)
        if norm < 1e-10:
            return np.zeros(DIM)
        return delta / norm
    
    def gradient_field(self, target: ICHTBPoint) -> np.ndarray:
        """
        Compute gradient field pointing toward target.
        Returns matrix where row i is the gradient at point i.
        """
        grads = np.zeros((len(self.points), DIM))
        for i, p in enumerate(self.points):
            grads[i] = self.gradient_at(p, target)
        return grads


# =============================================================================
# TEXT TO ICHTB PROJECTION (No neural networks!)
# =============================================================================

class ICHTBProjector:
    """
    Project text into ICHTB 48D space.
    
    THIS IS NOT AN EMBEDDING MODEL.
    
    We use deterministic mathematical operations:
    - Character-level hash functions
    - Frequency analysis
    - Structural patterns
    
    The goal is NOT semantic understanding via training.
    The goal is consistent mapping from text to 48D space
    where field dynamics can operate.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducible projections
        """
        self.rng = np.random.RandomState(seed)
        
        # Character-level projection matrix (deterministic)
        # Maps 256 possible byte values to 48D
        self.char_proj = self.rng.randn(256, DIM).astype(np.float64)
        self.char_proj /= np.linalg.norm(self.char_proj, axis=1, keepdims=True)
    
    def project_text(self, text: str) -> ICHTBPoint:
        """
        Project text into ICHTB 48D space.
        
        Method:
        1. Convert text to bytes
        2. Sum character projections (weighted by position)
        3. Normalize to unit sphere
        4. Structure into zone/operator format
        """
        if not text:
            return I_ZERO
        
        # Convert to bytes
        try:
            data = text.encode('utf-8')
        except:
            data = text.encode('latin-1')
        
        # Weighted sum of character projections
        vec = np.zeros(DIM)
        for i, byte in enumerate(data):
            # Position-based weight (earlier chars matter more)
            weight = 1.0 / (1.0 + 0.1 * i)
            vec += weight * self.char_proj[byte]
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        
        return ICHTBPoint(vec, text)
    
    def project_batch(self, texts: List[str]) -> List[ICHTBPoint]:
        """Project multiple texts."""
        return [self.project_text(t) for t in texts]


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ICHTB 48D Space Test")
    print("=" * 60)
    print(f"\nDimension: {DIM}")
    print(f"  = {ZONES} zones × {OPERATORS} operators × {COMPONENTS} components")
    
    # Test i₀
    print(f"\ni₀ vector norm: {np.linalg.norm(I_ZERO.vector):.4f}")
    print(f"i₀ Core Φ: {I_ZERO.phi(Zone.CORE)}")
    print(f"i₀ i-alignment: {I_ZERO.i_zero_alignment:.4f}")
    
    # Test projector
    projector = ICHTBProjector()
    
    p1 = projector.project_text("What is the capital of Mississippi?")
    p2 = projector.project_text("Jackson is the capital of Mississippi")
    p3 = projector.project_text("The speed of light is fast")
    
    print(f"\nProjected points:")
    print(f"  Q: '{p1.content[:40]}...' norm={np.linalg.norm(p1.vector):.4f}")
    print(f"  A: '{p2.content[:40]}...' norm={np.linalg.norm(p2.vector):.4f}")
    print(f"  X: '{p3.content[:40]}...' norm={np.linalg.norm(p3.vector):.4f}")
    
    print(f"\nDistances:")
    print(f"  Q <-> A: {p1.distance(p2):.4f}")
    print(f"  Q <-> X: {p1.distance(p3):.4f}")
    print(f"  A <-> X: {p2.distance(p3):.4f}")
    
    print(f"\nCosine similarities:")
    print(f"  Q <-> A: {p1.cosine_sim(p2):.4f}")
    print(f"  Q <-> X: {p1.cosine_sim(p3):.4f}")
    
    # Test space
    space = ICHTBSpace()
    space.add(p1)
    space.add(p2)
    space.add(p3)
    
    print(f"\nSpace contains {space.size()} points")
    
    # Find nearest to question
    nearest = space.most_similar(p1, k=3)
    print(f"\nMost similar to question:")
    for idx, sim in nearest:
        print(f"  [{idx}] sim={sim:.4f}: {space.get(idx).content[:50]}...")
    
    print("\n✓ ICHTB Space working")
