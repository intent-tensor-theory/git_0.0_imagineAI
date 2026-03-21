"""
operators.py - ITT Field Operators for Semantic Space

The six zones of the ICHTB map to operators on the Φ field:
    Δ₁ (+X) Forward:     ∇Φ    - Gradient (direction to target)
    Δ₂ (−Y) Memory:      ∇×F   - Curl (context loops)
    Δ₃ (+Y) Expansion:   +∇²Φ  - Positive Laplacian (meaning spreads)
    Δ₄ (−X) Compression: −∇²Φ  - Negative Laplacian (meaning focuses)
    Δ₅ (+Z) Apex:        ∂Φ/∂t - Temporal evolution
    Δ₆ (−Z) Core:        Φ=i₀  - Anchor point

In semantic space, these operate on embedding vectors rather than spatial grids.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .phi_field import PhiState, PhiField


@dataclass
class FieldGradient:
    """
    ∇Φ - The gradient of the semantic field.
    
    In spatial fields, gradient points toward increasing values.
    In semantic space, gradient points from current state toward target state.
    """
    direction: np.ndarray  # Unit vector pointing toward target
    magnitude: float       # Distance (strength of gradient)
    source: PhiState       # Where we are
    target: PhiState       # Where gradient points
    
    @property
    def vector(self) -> np.ndarray:
        """Full gradient vector (direction × magnitude)"""
        return self.direction * self.magnitude


@dataclass  
class FieldCurl:
    """
    ∇×F - The curl of the semantic field.
    
    Curl measures "rotation" or loops in the field. In conversation,
    this captures how context circles back - references to earlier topics,
    recurring themes, etc.
    
    High curl = strong contextual coherence
    Zero curl = no contextual memory
    """
    strength: float        # How much "rotation" is present
    loop_vectors: List[np.ndarray]  # The context vectors forming the loop
    
    @property
    def is_closed(self) -> bool:
        """Does the context form a closed loop?"""
        return self.strength > 0.5


@dataclass
class FieldLaplacian:
    """
    ∇²Φ - The Laplacian of the semantic field.
    
    Measures curvature / divergence:
    - Positive Laplacian: meaning is expanding/diffusing (many related concepts)
    - Negative Laplacian: meaning is focusing/condensing (precise definition)
    - Zero Laplacian: stable meaning (equilibrium)
    """
    value: float           # Positive = expansion, Negative = compression
    neighbors: List[PhiState]  # The neighboring states used to compute it
    
    @property
    def is_expanding(self) -> bool:
        return self.value > 0
    
    @property
    def is_compressing(self) -> bool:
        return self.value < 0
    
    @property
    def is_stable(self) -> bool:
        return abs(self.value) < 0.1


class SemanticOperators:
    """
    ITT operators adapted for semantic/embedding space.
    """
    
    def __init__(self, phi_field: PhiField):
        self.field = phi_field
        
    def gradient(self, current: PhiState, target: PhiState) -> FieldGradient:
        """
        Compute ∇Φ: the gradient from current state toward target.
        
        This is the "direction of meaning change" - how to transform
        the current semantic state to reach the target.
        
        Args:
            current: Where we are in semantic space
            target: Where we want to go
            
        Returns:
            FieldGradient with direction and magnitude
        """
        # Direction vector (unnormalized)
        delta = target.vector - current.vector
        
        # Magnitude (semantic distance)
        magnitude = np.linalg.norm(delta)
        
        # Normalized direction
        if magnitude > 1e-8:
            direction = delta / magnitude
        else:
            direction = np.zeros_like(delta)
        
        return FieldGradient(
            direction=direction,
            magnitude=magnitude,
            source=current,
            target=target
        )
    
    def curl(self, states: List[PhiState]) -> FieldCurl:
        """
        Compute ∇×F: the curl measuring contextual loops.
        
        Given a sequence of states (e.g., conversation history),
        measure how much the context "rotates" back on itself.
        
        High curl = references to earlier context
        Low curl = linear progression without backtracking
        
        Args:
            states: Sequence of PhiStates (e.g., conversation turns)
            
        Returns:
            FieldCurl measuring the "rotation" in context
        """
        if len(states) < 3:
            return FieldCurl(strength=0.0, loop_vectors=[])
        
        # Compute displacement vectors between consecutive states
        vectors = [s.vector for s in states]
        displacements = []
        for i in range(len(vectors) - 1):
            displacements.append(vectors[i+1] - vectors[i])
        
        # Measure how much the path curves back
        # High curl = later states similar to earlier states
        total_curl = 0.0
        for i in range(len(displacements) - 1):
            # Cross-product-like measure in high dimensions
            # Use the angle between consecutive displacements
            v1, v2 = displacements[i], displacements[i+1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                # Curl is high when direction changes significantly
                total_curl += (1 - cos_angle)
        
        # Normalize by number of transitions
        avg_curl = total_curl / max(len(displacements) - 1, 1)
        
        # Also check for direct loop-back: is last state similar to first?
        first_last_sim = states[0].similarity_to(states[-1])
        loop_strength = (avg_curl + first_last_sim) / 2
        
        return FieldCurl(
            strength=loop_strength,
            loop_vectors=displacements
        )
    
    def laplacian(self, center: PhiState, neighbors: List[PhiState]) -> FieldLaplacian:
        """
        Compute ∇²Φ: the Laplacian measuring meaning expansion/compression.
        
        Compares the center state to its semantic neighbors:
        - If neighbors are farther (on average) from center than expected: expanding
        - If neighbors are closer (more clustered): compressing
        - If balanced: stable
        
        Args:
            center: The central state to analyze
            neighbors: Nearby states in semantic space
            
        Returns:
            FieldLaplacian with expansion/compression value
        """
        if len(neighbors) == 0:
            return FieldLaplacian(value=0.0, neighbors=[])
        
        # Compute centroid of neighbors
        neighbor_vectors = np.array([n.vector for n in neighbors])
        centroid = np.mean(neighbor_vectors, axis=0)
        
        # Laplacian ≈ difference between center and average of neighbors
        # In discrete form: ∇²Φ ≈ (1/N)Σ(Φ_neighbor - Φ_center)
        laplacian_vec = centroid - center.vector
        
        # Scalar Laplacian: how much the center differs from its surroundings
        # Positive = center is "lower" than surroundings (will expand to fill)
        # Negative = center is "higher" than surroundings (will compress)
        value = np.linalg.norm(laplacian_vec)
        
        # Determine sign by checking if neighbors are diverging or converging
        # Compute average distance from center to neighbors
        avg_dist = np.mean([center.distance_to(n) for n in neighbors])
        
        # Compute average distance between neighbors (spread)
        if len(neighbors) > 1:
            neighbor_spread = 0.0
            count = 0
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    neighbor_spread += neighbors[i].distance_to(neighbors[j])
                    count += 1
            neighbor_spread /= max(count, 1)
        else:
            neighbor_spread = 0.0
        
        # If neighbors are spread out relative to center: positive Laplacian
        # If neighbors are clustered relative to center: negative Laplacian
        if neighbor_spread > avg_dist:
            value = abs(value)  # Positive (expanding)
        else:
            value = -abs(value)  # Negative (compressing)
        
        return FieldLaplacian(value=value, neighbors=neighbors)
    
    def temporal_derivative(
        self, 
        current: PhiState, 
        previous: PhiState, 
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute ∂Φ/∂t: the rate of change of the field.
        
        This is the Apex zone operator - governing temporal evolution.
        
        Args:
            current: Current state
            previous: Previous state
            dt: Time step (conversation turns = 1.0)
            
        Returns:
            Vector representing rate of change
        """
        return (current.vector - previous.vector) / dt
    
    def apply_master_equation(
        self,
        state: PhiState,
        target: Optional[PhiState] = None,
        neighbors: Optional[List[PhiState]] = None,
        history: Optional[List[PhiState]] = None,
        D: float = 0.1,      # Diffusion coefficient
        Lambda: float = 0.05, # Flux coupling
        gamma: float = 0.01,  # Cubic stabilization
        kappa: float = 0.1    # Damping
    ) -> np.ndarray:
        """
        Apply the ITT Master Equation to evolve the field:
        
        ∂Φ/∂t = D∇²Φ − Λ|∇Φ|² + γ|Φ|²Φ − κΦ
        
        This computes what the next state should be based on field dynamics.
        
        Args:
            state: Current state
            target: Target state (for gradient computation)
            neighbors: Neighboring states (for Laplacian)
            history: Past states (for curl/memory)
            D, Lambda, gamma, kappa: Master equation coefficients
            
        Returns:
            dPhi_dt: The rate of change vector
        """
        dPhi_dt = np.zeros_like(state.vector)
        
        # Laplacian term: D∇²Φ (diffusion)
        if neighbors:
            lap = self.laplacian(state, neighbors)
            # Direction from center to centroid of neighbors
            neighbor_vectors = np.array([n.vector for n in neighbors])
            centroid = np.mean(neighbor_vectors, axis=0)
            lap_vec = centroid - state.vector
            dPhi_dt += D * lap_vec
        
        # Gradient term: −Λ|∇Φ|² (flux coupling, nonlinear)
        if target:
            grad = self.gradient(state, target)
            # This term resists strong gradients
            dPhi_dt -= Lambda * (grad.magnitude ** 2) * grad.direction
        
        # Cubic term: +γ|Φ|²Φ (stabilization)
        amplitude_sq = state.amplitude ** 2
        dPhi_dt += gamma * amplitude_sq * state.vector
        
        # Damping term: −κΦ (drives toward equilibrium)
        dPhi_dt -= kappa * state.vector
        
        return dPhi_dt


def compute_sigma(state1: PhiState, state2: PhiState) -> float:
    """
    Compute σ: the residue (semantic distance) between two states.
    
    σ = 0 means perfect alignment
    σ = 1 means completely orthogonal
    σ = 2 means opposite
    
    This is the core measure that must be minimized for resolution.
    """
    return state1.distance_to(state2)


def compute_sigma_multi(states1: List[PhiState], states2: List[PhiState]) -> float:
    """
    Compute total σ across corresponding state pairs.
    
    σ_total = Σᵢ distance(states1[i], states2[i])
    """
    if len(states1) != len(states2):
        raise ValueError("State lists must have same length")
    
    total = 0.0
    for s1, s2 in zip(states1, states2):
        total += compute_sigma(s1, s2)
    return total
