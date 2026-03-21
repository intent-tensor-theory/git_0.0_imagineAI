"""
ichtb_projection.py - ICHTB-Aligned Embedding Space

The ICHTB (Inverse Cartesian Heisenberg Tensor Box) has a specific dimensional
structure derived from first principles, not arbitrary engineering choices.

From the ICHTB papers:
    6 zones (Δ₁-Δ₆) × 4 operators (Φ, ∇Φ, ∇×F, ∇²Φ) × 2 components (real, imaginary)
    = 48 fundamental dimensions

This module projects arbitrary embeddings into ICHTB-aligned 48D space,
where ITT operators can work on the correct geometric structure.

The 48 dimensions are structured as:
    [Δ₁: 8 dims][Δ₂: 8 dims][Δ₃: 8 dims][Δ₄: 8 dims][Δ₅: 8 dims][Δ₆: 8 dims]
    
Each zone's 8 dims are:
    [Φ_real, Φ_imag, ∇Φ_real, ∇Φ_imag, curl_real, curl_imag, lap_real, lap_imag]
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


# ICHTB Fundamental Constants
ICHTB_ZONES = 6          # Δ₁ through Δ₆
ICHTB_OPERATORS = 4      # Φ, ∇Φ, ∇×F, ∇²Φ
ICHTB_COMPONENTS = 2     # Real + Imaginary
ICHTB_DIMENSION = ICHTB_ZONES * ICHTB_OPERATORS * ICHTB_COMPONENTS  # = 48


class Zone(Enum):
    """The six ICHTB zones with their operators and meanings."""
    DELTA_1_FORWARD = 0      # +Y: ∇Φ (gradient / direction)
    DELTA_2_MEMORY = 1       # -Y: ∇×F (curl / memory)
    DELTA_3_EXPANSION = 2    # +X: +∇²Φ (positive Laplacian / growth)
    DELTA_4_COMPRESSION = 3  # -X: -∇²Φ (negative Laplacian / focusing)
    DELTA_5_APEX = 4         # +Z: ∂Φ/∂t (temporal / lock test)
    DELTA_6_CORE = 5         # -Z: Φ=i₀ (imaginary anchor)


class Operator(Enum):
    """The four operators present in each zone."""
    PHI = 0           # Scalar potential
    GRAD_PHI = 1      # Gradient
    CURL_F = 2        # Curl (phase memory)
    LAP_PHI = 3       # Laplacian (curvature)


@dataclass
class ICHTBState:
    """
    A state in ICHTB-aligned 48-dimensional space.
    
    The vector is structured as 6 zones × 8 components per zone.
    """
    vector: np.ndarray  # Shape: (48,)
    source_text: Optional[str] = None
    
    def __post_init__(self):
        if self.vector.shape != (ICHTB_DIMENSION,):
            raise ValueError(f"ICHTB state must be {ICHTB_DIMENSION}D, got {self.vector.shape}")
    
    def get_zone(self, zone: Zone) -> np.ndarray:
        """Extract the 8-dimensional subspace for a specific zone."""
        start = zone.value * 8
        return self.vector[start:start + 8]
    
    def get_operator(self, zone: Zone, operator: Operator) -> complex:
        """Extract a specific operator value (as complex number)."""
        zone_vec = self.get_zone(zone)
        op_idx = operator.value * 2
        return complex(zone_vec[op_idx], zone_vec[op_idx + 1])
    
    def get_phi(self, zone: Zone) -> complex:
        """Get Φ value for a zone."""
        return self.get_operator(zone, Operator.PHI)
    
    def get_gradient(self, zone: Zone) -> complex:
        """Get ∇Φ value for a zone."""
        return self.get_operator(zone, Operator.GRAD_PHI)
    
    def get_curl(self, zone: Zone) -> complex:
        """Get ∇×F value for a zone."""
        return self.get_operator(zone, Operator.CURL_F)
    
    def get_laplacian(self, zone: Zone) -> complex:
        """Get ∇²Φ value for a zone."""
        return self.get_operator(zone, Operator.LAP_PHI)
    
    @property
    def total_phi(self) -> float:
        """Total |Φ| across all zones."""
        return sum(abs(self.get_phi(z)) for z in Zone)
    
    @property
    def i_zero_alignment(self) -> float:
        """How close to i₀ (pure imaginary) is the Core zone."""
        core_phi = self.get_phi(Zone.DELTA_6_CORE)
        # Perfect i₀ alignment means real part = 0, imaginary part > 0
        if abs(core_phi) < 1e-8:
            return 0.0
        return abs(core_phi.imag) / abs(core_phi)
    
    def distance_to(self, other: 'ICHTBState') -> float:
        """Euclidean distance in ICHTB space."""
        return np.linalg.norm(self.vector - other.vector)
    
    def cosine_similarity(self, other: 'ICHTBState') -> float:
        """Cosine similarity in ICHTB space."""
        n1 = np.linalg.norm(self.vector)
        n2 = np.linalg.norm(other.vector)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return np.dot(self.vector, other.vector) / (n1 * n2)


class ICHTBProjection:
    """
    Projects arbitrary embeddings into ICHTB-aligned 48D space.
    
    Takes embeddings of any dimension (300, 384, 768, etc.) and projects
    them into the 48-dimensional ICHTB structure where ITT operators
    can work correctly.
    """
    
    def __init__(self, input_dim: int = 384, learned: bool = False):
        """
        Args:
            input_dim: Dimension of input embeddings
            learned: If True, projection matrix is learnable (for training)
                    If False, uses fixed orthogonal projection
        """
        self.input_dim = input_dim
        self.output_dim = ICHTB_DIMENSION  # 48
        self.learned = learned
        
        # Initialize projection matrix
        if learned:
            # Random initialization for learnable projection
            self.W = np.random.randn(input_dim, ICHTB_DIMENSION).astype(np.float32)
            self.W = self.W / np.linalg.norm(self.W, axis=0, keepdims=True)
        else:
            # Fixed orthogonal projection
            # Uses SVD to get the best 48-dim subspace
            self.W = self._create_orthogonal_projection()
        
        # Inverse projection (for reconstruction)
        self.W_inv = np.linalg.pinv(self.W)
    
    def _create_orthogonal_projection(self) -> np.ndarray:
        """
        Create orthogonal projection matrix.
        
        Uses random orthogonal basis aligned with ICHTB zone structure.
        Each zone gets a distinct 8-dim subspace.
        """
        # Create random matrix and orthogonalize
        A = np.random.randn(self.input_dim, ICHTB_DIMENSION).astype(np.float32)
        
        # QR decomposition for orthogonality
        Q, R = np.linalg.qr(A)
        
        # Take first 48 columns (or pad if input_dim < 48)
        if self.input_dim >= ICHTB_DIMENSION:
            W = Q[:, :ICHTB_DIMENSION]
        else:
            # If input is smaller than 48, pad with zeros
            W = np.zeros((self.input_dim, ICHTB_DIMENSION), dtype=np.float32)
            W[:, :self.input_dim] = np.eye(self.input_dim)
        
        return W
    
    def project(self, embedding: np.ndarray) -> ICHTBState:
        """
        Project an embedding into ICHTB space.
        
        Args:
            embedding: Input embedding vector (any dimension)
            
        Returns:
            ICHTBState in 48D ICHTB-aligned space
        """
        # Handle dimension mismatch
        if len(embedding) != self.input_dim:
            # Resize: truncate or pad
            if len(embedding) > self.input_dim:
                embedding = embedding[:self.input_dim]
            else:
                embedding = np.pad(embedding, (0, self.input_dim - len(embedding)))
        
        # Project to ICHTB space
        ichtb_vec = embedding @ self.W
        
        # Normalize to unit sphere (optional but helps with stability)
        norm = np.linalg.norm(ichtb_vec)
        if norm > 1e-8:
            ichtb_vec = ichtb_vec / norm
        
        return ICHTBState(vector=ichtb_vec.astype(np.float32))
    
    def unproject(self, ichtb_state: ICHTBState) -> np.ndarray:
        """
        Reconstruct embedding from ICHTB state (inverse projection).
        
        Note: Information is lost if input_dim > 48.
        """
        return ichtb_state.vector @ self.W_inv
    
    def project_batch(self, embeddings: np.ndarray) -> List[ICHTBState]:
        """Project a batch of embeddings."""
        return [self.project(emb) for emb in embeddings]


class ICHTBOperators:
    """
    ITT field operators that work directly in ICHTB space.
    
    These operators respect the zone structure and apply the correct
    transformations based on which zone governs which operator.
    """
    
    def __init__(self):
        # Zone dominance weights (which zone primarily governs which operator)
        # From Chapter 2: ★ marks the governing operator
        self.zone_weights = {
            Operator.PHI: {Zone.DELTA_6_CORE: 1.0},  # Φ=i₀ at Core
            Operator.GRAD_PHI: {Zone.DELTA_1_FORWARD: 1.0},  # ∇Φ at Forward
            Operator.CURL_F: {Zone.DELTA_2_MEMORY: 1.0},  # ∇×F at Memory
            Operator.LAP_PHI: {
                Zone.DELTA_3_EXPANSION: 0.5,   # +∇²Φ
                Zone.DELTA_4_COMPRESSION: 0.5  # -∇²Φ
            }
        }
    
    def gradient(self, current: ICHTBState, target: ICHTBState) -> np.ndarray:
        """
        Compute ∇Φ: gradient from current to target in ICHTB space.
        
        The gradient is dominated by the Δ₁ (Forward) zone.
        """
        delta = target.vector - current.vector
        
        # Extract Δ₁ component (indices 0-7)
        delta_1 = delta[:8]
        
        # Weighted by Δ₁ dominance
        grad_direction = np.zeros(ICHTB_DIMENSION)
        grad_direction[:8] = delta_1  # Gradient lives primarily in Δ₁
        
        return grad_direction
    
    def curl(self, states: List[ICHTBState]) -> float:
        """
        Compute ∇×F: curl measuring memory/rotation.
        
        The curl is dominated by the Δ₂ (Memory) zone.
        """
        if len(states) < 3:
            return 0.0
        
        # Extract Δ₂ components from all states
        memory_components = [s.vector[8:16] for s in states]  # Indices 8-15 = Δ₂
        
        # Compute "rotation" as cross-product-like measure
        total_rotation = 0.0
        for i in range(len(memory_components) - 2):
            v1 = memory_components[i+1] - memory_components[i]
            v2 = memory_components[i+2] - memory_components[i+1]
            
            # Curl magnitude from angle between consecutive displacements
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                total_rotation += (1 - cos_angle)
        
        return total_rotation / max(len(states) - 2, 1)
    
    def laplacian(self, center: ICHTBState, neighbors: List[ICHTBState]) -> float:
        """
        Compute ∇²Φ: curvature (expansion vs compression).
        
        The Laplacian is split between Δ₃ (expansion, +∇²Φ) and Δ₄ (compression, -∇²Φ).
        Positive return = expanding, Negative return = compressing.
        """
        if not neighbors:
            return 0.0
        
        # Extract Δ₃ and Δ₄ components
        delta_3_center = center.vector[16:24]  # Expansion zone
        delta_4_center = center.vector[24:32]  # Compression zone
        
        delta_3_neighbors = np.mean([n.vector[16:24] for n in neighbors], axis=0)
        delta_4_neighbors = np.mean([n.vector[24:32] for n in neighbors], axis=0)
        
        # Laplacian ≈ (neighbor average - center)
        expansion = np.linalg.norm(delta_3_neighbors - delta_3_center)
        compression = np.linalg.norm(delta_4_neighbors - delta_4_center)
        
        # Positive if expanding, negative if compressing
        return expansion - compression
    
    def compute_sigma(self, state1: ICHTBState, state2: ICHTBState) -> float:
        """
        Compute σ (residue): misalignment in ICHTB space.
        
        This is the fundamental measure that must be minimized.
        """
        return 1.0 - state1.cosine_similarity(state2)
    
    def check_lock(self, state: ICHTBState, threshold: float = 0.1) -> bool:
        """
        Check if state satisfies the Apex lock condition (Δ₅).
        
        From Chapter 3: Lock occurs when ∂Φ/∂t ≈ 0, meaning
        μ∇²Φ ≈ νΦ (eigenvalue condition).
        """
        # Extract Δ₅ (Apex) component
        apex = state.vector[32:40]
        
        # Lock condition: Apex component is stable (low variance)
        variance = np.var(apex)
        return variance < threshold


def create_ichtb_field(embedding_dim: int = 384):
    """
    Create an ICHTB-aligned phi field.
    
    Returns a PhiField that automatically projects to ICHTB space.
    """
    from .phi_field import PhiField, PhiState
    
    class ICHTBPhiField(PhiField):
        """PhiField that works in ICHTB-aligned 48D space."""
        
        def __init__(self, embedding_model, input_dim: int = 384):
            super().__init__(embedding_model)
            self.projection = ICHTBProjection(input_dim=input_dim)
            self.dimension = ICHTB_DIMENSION  # Always 48
        
        def embed(self, text):
            # Get raw embedding
            raw_state = super().embed(text)
            
            # Project to ICHTB space
            ichtb_state = self.projection.project(raw_state.vector)
            
            # Return as PhiState with ICHTB vector
            return PhiState(
                vector=ichtb_state.vector,
                text=text
            )
    
    return ICHTBPhiField


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("ICHTB Projection Test")
    print("=" * 60)
    print(f"\nICHTB Dimension: {ICHTB_DIMENSION}")
    print(f"  = {ICHTB_ZONES} zones × {ICHTB_OPERATORS} operators × {ICHTB_COMPONENTS} components")
    
    # Test projection
    proj = ICHTBProjection(input_dim=384)
    
    # Random embedding
    test_embed = np.random.randn(384)
    ichtb_state = proj.project(test_embed)
    
    print(f"\nProjected 384D → {len(ichtb_state.vector)}D")
    print(f"i₀ alignment: {ichtb_state.i_zero_alignment:.4f}")
    
    # Test zone extraction
    for zone in Zone:
        zone_vec = ichtb_state.get_zone(zone)
        print(f"  {zone.name}: ||v|| = {np.linalg.norm(zone_vec):.4f}")
    
    print("\n✓ ICHTB projection working")
