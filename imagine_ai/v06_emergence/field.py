"""
field.py - The Semantic Field (Φ)

This is the collapse tension substrate for language.
GloVe embeddings provide the scalar potential at each point.
The field evolves according to the ICHTB master equation.

Φ is not a list of facts. Φ is the semantic manifold itself.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class SemanticField:
    """
    The semantic field Φ.
    
    Not a list. Not a database. A continuous field over meaning space.
    """
    # The GloVe substrate
    glove: any  # gensim KeyedVectors
    
    # Field state (300D for GloVe-300)
    Φ: np.ndarray = None
    
    # Field history (for computing ∂Φ/∂t)
    Φ_history: List[np.ndarray] = None
    
    # The metric tensor M (built from field gradients)
    M: np.ndarray = None
    
    # Dimension
    dim: int = 300
    
    def __post_init__(self):
        if self.Φ is None:
            self.Φ = np.zeros(self.dim)
        if self.Φ_history is None:
            self.Φ_history = []
        if self.M is None:
            self.M = np.eye(self.dim)
    
    def reset(self):
        """Reset field to zero state."""
        self.Φ = np.zeros(self.dim)
        self.Φ_history = []
        self.M = np.eye(self.dim)


def text_to_embedding(text: str, glove) -> np.ndarray:
    """
    Convert text to GloVe embedding (mean of word vectors).
    
    This is the projection of text into the CTS.
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    vectors = []
    
    for word in words:
        if word in glove:
            vectors.append(glove[word])
    
    if not vectors:
        return np.zeros(glove.vector_size)
    
    return np.mean(vectors, axis=0)


def initialize_field_from_question(question: str, glove) -> SemanticField:
    """
    Initialize the field from a question.
    
    The question is a perturbation of the CTS.
    It creates the initial Φ from which emergence proceeds.
    """
    # Project question into semantic space
    Φ_init = text_to_embedding(question, glove)
    
    # Normalize to unit amplitude (the magnitude comes from evolution)
    if np.linalg.norm(Φ_init) > 0:
        Φ_init = Φ_init / np.linalg.norm(Φ_init)
    
    return SemanticField(
        glove=glove,
        Φ=Φ_init,
        Φ_history=[Φ_init.copy()],
        dim=glove.vector_size
    )


def compute_gradient(field: SemanticField, substrate_points: List[np.ndarray]) -> np.ndarray:
    """
    Compute ∇Φ - the gradient of the field.
    
    In discrete semantic space, this is approximated by
    the direction of maximum variation from Φ to nearby points.
    
    This is the Δ₁ (Forward) zone operator.
    """
    if not substrate_points:
        return np.zeros_like(field.Φ)
    
    # Find direction of maximum semantic change
    gradients = []
    for point in substrate_points:
        diff = point - field.Φ
        gradients.append(diff)
    
    # The gradient is the mean direction weighted by magnitude
    grad = np.mean(gradients, axis=0)
    
    return grad


def compute_curl(field: SemanticField, context_vectors: List[np.ndarray]) -> np.ndarray:
    """
    Compute ∇×F - the curl of the field.
    
    In semantic space, curl represents "looping" relationships -
    concepts that refer back to each other.
    
    This is the Δ₂ (Memory) zone operator.
    
    Approximated via the antisymmetric part of the outer product
    of context vectors.
    """
    if len(context_vectors) < 2:
        return np.zeros_like(field.Φ)
    
    # Curl is approximated by rotational component
    # Using cross-correlation of sequential context
    curl = np.zeros_like(field.Φ)
    
    for i in range(len(context_vectors) - 1):
        v1 = context_vectors[i]
        v2 = context_vectors[i + 1]
        
        # Rotational contribution: orthogonal to both
        # In high-D, we use the Gram-Schmidt remainder
        proj = np.dot(v2, v1) * v1 / (np.dot(v1, v1) + 1e-8)
        rotation = v2 - proj
        curl += rotation
    
    return curl / max(len(context_vectors) - 1, 1)


def compute_laplacian(field: SemanticField, substrate_points: List[np.ndarray]) -> np.ndarray:
    """
    Compute ∇²Φ - the Laplacian of the field.
    
    In discrete space, Laplacian ≈ (average of neighbors) - center
    
    This governs both:
    - Δ₃ (Expansion): +∇²Φ spreads the field
    - Δ₄ (Compression): -∇²Φ sharpens the field
    """
    if not substrate_points:
        return np.zeros_like(field.Φ)
    
    # Average of substrate points
    neighbor_avg = np.mean(substrate_points, axis=0)
    
    # Laplacian = average - center
    laplacian = neighbor_avg - field.Φ
    
    return laplacian


def compute_temporal_derivative(field: SemanticField, dt: float = 1.0) -> np.ndarray:
    """
    Compute ∂Φ/∂t - the time derivative of the field.
    
    This is the Δ₅ (Apex) zone operator.
    Lock occurs when ∂Φ/∂t ≈ 0.
    """
    if len(field.Φ_history) < 2:
        return np.zeros_like(field.Φ)
    
    # Finite difference
    dΦ_dt = (field.Φ - field.Φ_history[-1]) / dt
    
    return dΦ_dt


def update_metric_tensor(field: SemanticField, grad_Φ: np.ndarray, curl_F: np.ndarray):
    """
    Update the metric tensor M from field gradients.
    
    From ICHTB master equation:
    M_ij = <∂_i Φ ∂_j Φ> - λ<F_i F_j> + μ δ_ij ∇²Φ
    
    The metric is BUILT from the field, not imposed.
    """
    # Gradient outer product
    grad_outer = np.outer(grad_Φ, grad_Φ)
    
    # Curl correction (reduces metric in rotation directions)
    curl_outer = np.outer(curl_F, curl_F)
    
    # Regularization (keeps metric well-conditioned)
    reg = np.eye(field.dim) * 0.01
    
    # Combined metric
    λ = 0.1  # Curl weight
    field.M = grad_outer - λ * curl_outer + reg
    
    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(field.M)
    if eigvals.min() < 0:
        field.M += np.eye(field.dim) * (abs(eigvals.min()) + 0.01)


# =============================================================================
# The i₀ Anchor (Δ₆)
# =============================================================================

# The imaginary anchor - the recursion seed
# This is not a location in semantic space
# It is the reference point that makes recursion possible
I_0 = 1j  # Pure imaginary

def distance_from_anchor(Φ: np.ndarray) -> float:
    """
    Compute distance from the imaginary anchor i₀.
    
    Since i₀ is imaginary and Φ is real, this is always non-zero.
    The recursion can "approach" i₀ but never reach it.
    
    This is what keeps the system alive.
    """
    # Φ is real, i₀ is imaginary
    # Distance is |Φ - i₀| = sqrt(|Φ|² + 1)
    return np.sqrt(np.dot(Φ, Φ) + 1)
