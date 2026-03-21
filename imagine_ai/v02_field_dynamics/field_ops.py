"""
field_ops.py - ITT Field Operators in 48D ICHTB Space

These operators are the MATH that finds answers.
NOT pattern matching. NOT statistical inference.
Pure field dynamics.

From the Master Equation:
    ∂Φ/∂t = D∇²Φ − Λ|∇Φ|² + γΦ³ − κΦ

Zone Operators:
    Δ₁ (Forward):     ∇Φ  - gradient, direction toward answer
    Δ₂ (Memory):      ∇×F - curl, context preservation
    Δ₃ (Expansion):   +∇²Φ - diffusion outward
    Δ₄ (Compression): -∇²Φ - focus inward
    Δ₅ (Apex):        ∂Φ/∂t - lock test
    Δ₆ (Core):        Φ=i₀ - imaginary anchor
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .ichtb_space import ICHTBPoint, ICHTBSpace, Zone, Operator, DIM, I_ZERO


# =============================================================================
# MASTER EQUATION COEFFICIENTS
# =============================================================================

@dataclass
class MasterCoefficients:
    """
    Coefficients for the Master Equation:
        ∂Φ/∂t = D∇²Φ − Λ|∇Φ|² + γΦ³ − κΦ
    """
    D: float = 0.1      # Diffusivity (spreading rate)
    Lambda: float = 0.2  # Alignment decay rate (gradient braking)
    gamma: float = 0.05  # Nonlinear growth rate
    kappa: float = 0.1   # Linear decay rate
    dt: float = 0.01     # Time step


DEFAULT_COEFFS = MasterCoefficients()


# =============================================================================
# GRADIENT OPERATOR - ∇Φ (Δ₁ Forward Zone)
# =============================================================================

def compute_gradient(
    current: ICHTBPoint,
    target: ICHTBPoint
) -> np.ndarray:
    """
    Compute ∇Φ: gradient from current position toward target.
    
    This is the Δ₁ (Forward) operation.
    The gradient points in the direction of steepest descent toward the answer.
    
    Returns:
        48D gradient vector
    """
    delta = target.vector - current.vector
    norm = np.linalg.norm(delta)
    
    if norm < 1e-10:
        # Already at target
        return np.zeros(DIM)
    
    return delta / norm


def gradient_magnitude(
    current: ICHTBPoint,
    target: ICHTBPoint
) -> float:
    """Magnitude of gradient (how far from target)."""
    return np.linalg.norm(target.vector - current.vector)


# =============================================================================
# CURL OPERATOR - ∇×F (Δ₂ Memory Zone)
# =============================================================================

def compute_curl(
    history: List[ICHTBPoint],
    weights: Optional[List[float]] = None
) -> float:
    """
    Compute ∇×F: curl measuring rotational memory.
    
    This is the Δ₂ (Memory) operation.
    The curl measures how much the path curves/rotates,
    which encodes contextual memory.
    
    Args:
        history: Sequence of previous points (trajectory)
        weights: Optional recency weights
        
    Returns:
        Scalar curl magnitude (rotation amount)
    """
    if len(history) < 3:
        return 0.0
    
    if weights is None:
        # Recency bias: more recent = higher weight
        weights = [1.0 / (1.0 + 0.5 * i) for i in range(len(history))]
    
    total_curl = 0.0
    
    for i in range(len(history) - 2):
        # Three consecutive points
        p1 = history[i].vector
        p2 = history[i + 1].vector
        p3 = history[i + 2].vector
        
        # Vectors between consecutive points
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Curl is measured by how much direction changes
        # (analogous to 2D rotation or 3D curl magnitude)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        
        # Cosine of angle between consecutive displacements
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to rotation: more perpendicular = more curl
        rotation = 1.0 - cos_angle  # 0 for parallel, 2 for antiparallel
        
        # Weight by recency
        w = weights[i] if i < len(weights) else 0.5
        total_curl += w * rotation
    
    return total_curl / max(len(history) - 2, 1)


# =============================================================================
# LAPLACIAN OPERATOR - ∇²Φ (Δ₃/Δ₄ Expansion/Compression)
# =============================================================================

def compute_laplacian(
    center: ICHTBPoint,
    neighbors: List[ICHTBPoint]
) -> float:
    """
    Compute ∇²Φ: Laplacian measuring curvature.
    
    This is the Δ₃/Δ₄ operation.
    Positive Laplacian = center is below neighbors = expanding
    Negative Laplacian = center is above neighbors = compressing
    
    Args:
        center: The point to evaluate at
        neighbors: Surrounding points in the field
        
    Returns:
        Scalar Laplacian (positive = expanding, negative = compressing)
    """
    if not neighbors:
        return 0.0
    
    # Average of neighbors
    neighbor_avg = np.mean([n.vector for n in neighbors], axis=0)
    
    # Laplacian ≈ (neighbor average - center)
    # This is the discrete Laplacian on a graph
    lap = np.linalg.norm(neighbor_avg - center.vector)
    
    # Sign: positive if neighbors are "larger" than center
    center_mag = np.linalg.norm(center.vector)
    neighbor_mag = np.linalg.norm(neighbor_avg)
    
    return lap if neighbor_mag > center_mag else -lap


# =============================================================================
# TEMPORAL DERIVATIVE - ∂Φ/∂t (Δ₅ Apex Zone)
# =============================================================================

def compute_temporal_derivative(
    current: ICHTBPoint,
    previous: ICHTBPoint,
    dt: float = DEFAULT_COEFFS.dt
) -> np.ndarray:
    """
    Compute ∂Φ/∂t: rate of change.
    
    This is the Δ₅ (Apex) operation.
    When ∂Φ/∂t → 0, the field has locked (stable configuration found).
    
    Args:
        current: Current state
        previous: Previous state
        dt: Time step
        
    Returns:
        48D derivative vector
    """
    return (current.vector - previous.vector) / dt


def temporal_magnitude(
    current: ICHTBPoint,
    previous: ICHTBPoint,
    dt: float = DEFAULT_COEFFS.dt
) -> float:
    """Magnitude of temporal change (how fast the field is evolving)."""
    deriv = compute_temporal_derivative(current, previous, dt)
    return np.linalg.norm(deriv)


# =============================================================================
# MASTER EQUATION - The Full Field Evolution
# =============================================================================

def apply_master_equation(
    state: ICHTBPoint,
    target: ICHTBPoint,
    neighbors: List[ICHTBPoint],
    coeffs: MasterCoefficients = DEFAULT_COEFFS
) -> ICHTBPoint:
    """
    Apply one step of the Master Equation:
        ∂Φ/∂t = D∇²Φ − Λ|∇Φ|² + γΦ³ − κΦ
    
    This evolves the field toward the stable configuration.
    
    Args:
        state: Current field state
        target: Target configuration (from constraints)
        neighbors: Neighboring points for Laplacian
        coeffs: Master equation coefficients
        
    Returns:
        New state after one time step
    """
    vec = state.vector.copy()
    
    # Term 1: D∇²Φ (diffusion toward neighbors)
    if neighbors:
        neighbor_avg = np.mean([n.vector for n in neighbors], axis=0)
        term1 = coeffs.D * (neighbor_avg - vec)
    else:
        term1 = np.zeros(DIM)
    
    # Term 2: -Λ|∇Φ|² (gradient braking)
    grad = compute_gradient(state, target)
    grad_mag = np.linalg.norm(grad)
    term2 = -coeffs.Lambda * grad_mag**2 * grad
    
    # Term 3: γΦ³ (nonlinear growth)
    phi_mag = np.linalg.norm(vec)
    term3 = coeffs.gamma * phi_mag**2 * vec
    
    # Term 4: -κΦ (decay toward i₀)
    term4 = -coeffs.kappa * vec
    
    # Apply update: new = old + dt * ∂Φ/∂t
    dPhi_dt = term1 + term2 + term3 + term4
    new_vec = vec + coeffs.dt * dPhi_dt
    
    # Normalize to prevent unbounded growth
    norm = np.linalg.norm(new_vec)
    if norm > 10.0:  # Cap magnitude
        new_vec = new_vec * 10.0 / norm
    
    return ICHTBPoint(new_vec, state.content)


# =============================================================================
# σ RESIDUE - The Measure of Misalignment
# =============================================================================

@dataclass
class SigmaResult:
    """Result of σ (residue) calculation."""
    sigma: float                    # Total σ residue
    distance_component: float       # Distance in 48D space
    constraint_violations: float    # How many constraints violated
    locked: bool                    # Is σ below threshold?


def compute_sigma(
    candidate: ICHTBPoint,
    target: ICHTBPoint,
    constraints: List['Constraint'] = None,
    lock_threshold: float = 0.1
) -> SigmaResult:
    """
    Compute σ: the irreducible residue (misalignment).
    
    σ is the fundamental measure in ITT.
    When σ → 0, the answer has been found.
    
    Args:
        candidate: The candidate answer
        target: The ideal target (from question constraints)
        constraints: Additional boundary conditions (ρ_q)
        lock_threshold: σ below this = locked
        
    Returns:
        SigmaResult with breakdown
    """
    # Distance component: how far in 48D space
    distance = candidate.distance(target)
    
    # Constraint violations
    violations = 0.0
    if constraints:
        for c in constraints:
            if not c.satisfied(candidate):
                violations += c.penalty
    
    # Total σ
    sigma = distance + violations
    
    return SigmaResult(
        sigma=sigma,
        distance_component=distance,
        constraint_violations=violations,
        locked=(sigma < lock_threshold)
    )


# =============================================================================
# CONSTRAINT - ρ_q Boundary Conditions
# =============================================================================

@dataclass
class Constraint:
    """
    A boundary condition (ρ_q) that the answer must satisfy.
    
    These are extracted from the question and define the
    "boundaries" within which the answer must lie.
    """
    name: str           # Description
    check_fn: callable  # Function(ICHTBPoint) -> bool
    penalty: float = 1.0  # Penalty if violated
    
    def satisfied(self, point: ICHTBPoint) -> bool:
        """Check if constraint is satisfied."""
        return self.check_fn(point)


def must_be_similar_to(reference: ICHTBPoint, threshold: float = 0.7) -> Constraint:
    """Constraint: point must be similar to reference."""
    def check(p: ICHTBPoint) -> bool:
        return p.cosine_sim(reference) >= threshold
    return Constraint(f"similar_to({reference.content[:20]})", check)


def must_be_different_from(reference: ICHTBPoint, threshold: float = 0.3) -> Constraint:
    """Constraint: point must be different from reference."""
    def check(p: ICHTBPoint) -> bool:
        return p.cosine_sim(reference) <= threshold
    return Constraint(f"different_from({reference.content[:20]})", check)


def must_be_closer_to_than(
    target: ICHTBPoint,
    reference: ICHTBPoint
) -> Constraint:
    """Constraint: point must be closer to target than to reference."""
    def check(p: ICHTBPoint) -> bool:
        return p.distance(target) < p.distance(reference)
    return Constraint(f"closer_to({target.content[:20]})", check)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    from .ichtb_space import ICHTBProjector
    
    print("=" * 60)
    print("Field Operators Test")
    print("=" * 60)
    
    projector = ICHTBProjector()
    
    # Create test points
    question = projector.project_text("What is the capital of Mississippi?")
    correct = projector.project_text("Jackson is the capital of Mississippi")
    wrong = projector.project_text("The moon is made of cheese")
    
    # Test gradient
    grad_correct = compute_gradient(question, correct)
    grad_wrong = compute_gradient(question, wrong)
    
    print("\nGradient magnitudes (toward answer):")
    print(f"  Q → correct: {np.linalg.norm(grad_correct):.4f}")
    print(f"  Q → wrong:   {np.linalg.norm(grad_wrong):.4f}")
    
    # Test σ
    sigma_correct = compute_sigma(correct, question)
    sigma_wrong = compute_sigma(wrong, question)
    
    print("\nσ residue:")
    print(f"  correct: σ={sigma_correct.sigma:.4f}, locked={sigma_correct.locked}")
    print(f"  wrong:   σ={sigma_wrong.sigma:.4f}, locked={sigma_wrong.locked}")
    
    # Test Master Equation step
    neighbors = [correct, wrong]
    new_state = apply_master_equation(question, correct, neighbors)
    
    print(f"\nAfter Master Equation step:")
    print(f"  Distance to correct: {question.distance(correct):.4f} → {new_state.distance(correct):.4f}")
    
    print("\n✓ Field operators working")
