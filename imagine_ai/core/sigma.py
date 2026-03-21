"""
sigma.py - Residue Calculation

σ (sigma) is the irreducible residue - the misalignment between current state
and target state. In imagineAI, this is the measure that must be minimized
for the field to resolve to an answer.

From ITT/ARC-AGI paper:
    σ = irreducible misalignment accumulated in change
    Resolution occurs when σ → 0
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from .phi_field import PhiState


@dataclass
class SigmaResult:
    """
    Result of a sigma (residue) calculation.
    """
    total: float           # Total residue
    components: Dict[str, float]  # Breakdown by component
    is_resolved: bool      # Whether σ < threshold
    
    def __str__(self):
        status = "✓ RESOLVED" if self.is_resolved else "✗ UNRESOLVED"
        return f"σ = {self.total:.4f} [{status}]"


class SigmaCalculator:
    """
    Calculate the residue σ between semantic states.
    
    σ combines multiple distance measures:
    1. Semantic distance: cosine distance in embedding space
    2. Constraint satisfaction: penalty for violating ρ_q boundaries
    3. Coherence: penalty for internal inconsistency
    """
    
    def __init__(self, threshold: float = 0.01):
        """
        Args:
            threshold: σ values below this are considered "resolved"
        """
        self.threshold = threshold
        self.constraint_checkers: List[Callable] = []
        
    def add_constraint(self, checker: Callable[[PhiState], float]):
        """
        Add a constraint checker function.
        
        Args:
            checker: Function that takes a PhiState and returns a penalty
                    (0 = constraint satisfied, >0 = violation)
        """
        self.constraint_checkers.append(checker)
        
    def compute(
        self, 
        current: PhiState, 
        target: Optional[PhiState] = None,
        context: Optional[List[PhiState]] = None
    ) -> SigmaResult:
        """
        Compute the total residue σ.
        
        Args:
            current: Current semantic state
            target: Target state (if known)
            context: Context states (conversation history)
            
        Returns:
            SigmaResult with total and component breakdown
        """
        components = {}
        
        # 1. Semantic distance to target (if known)
        if target is not None:
            semantic_dist = current.distance_to(target)
            components['semantic'] = semantic_dist
        else:
            components['semantic'] = 0.0
        
        # 2. Constraint violations
        constraint_penalty = 0.0
        for i, checker in enumerate(self.constraint_checkers):
            try:
                penalty = checker(current)
                constraint_penalty += penalty
                components[f'constraint_{i}'] = penalty
            except Exception as e:
                # Constraint check failed - add small penalty
                constraint_penalty += 0.1
                components[f'constraint_{i}_error'] = 0.1
        
        components['constraints'] = constraint_penalty
        
        # 3. Coherence with context (if provided)
        if context and len(context) > 0:
            # Measure how well current state fits with context
            # Low coherence = high penalty
            context_distances = [current.distance_to(c) for c in context]
            avg_context_dist = np.mean(context_distances)
            
            # We want some distance (not copying context) but not too much
            # Sweet spot around 0.3-0.7 cosine distance
            if avg_context_dist < 0.1:
                coherence_penalty = 0.2  # Too similar (copying)
            elif avg_context_dist > 0.9:
                coherence_penalty = 0.3  # Too different (non-sequitur)
            else:
                coherence_penalty = 0.0
            
            components['coherence'] = coherence_penalty
        else:
            components['coherence'] = 0.0
        
        # Total σ
        total = sum(components.values())
        
        return SigmaResult(
            total=total,
            components=components,
            is_resolved=(total < self.threshold)
        )
    
    def compute_for_candidates(
        self,
        candidates: List[PhiState],
        target: Optional[PhiState] = None,
        context: Optional[List[PhiState]] = None
    ) -> List[SigmaResult]:
        """
        Compute σ for multiple candidate states.
        
        Args:
            candidates: List of candidate states to evaluate
            target: Target state (if known)
            context: Context for coherence calculation
            
        Returns:
            List of SigmaResults, one per candidate
        """
        return [self.compute(c, target, context) for c in candidates]
    
    def best_candidate(
        self,
        candidates: List[PhiState],
        target: Optional[PhiState] = None,
        context: Optional[List[PhiState]] = None
    ) -> tuple[PhiState, SigmaResult]:
        """
        Find the candidate with minimum σ.
        
        Args:
            candidates: List of candidate states
            target: Target state (if known)
            context: Context for coherence
            
        Returns:
            (best_state, sigma_result) tuple
        """
        if not candidates:
            raise ValueError("No candidates provided")
        
        results = self.compute_for_candidates(candidates, target, context)
        
        # Find minimum σ
        best_idx = np.argmin([r.total for r in results])
        
        return candidates[best_idx], results[best_idx]


def sigma_trace(
    initial: PhiState,
    final: PhiState,
    steps: List[PhiState]
) -> List[float]:
    """
    Trace the σ value over a resolution path.
    
    Args:
        initial: Starting state
        final: Target state
        steps: Intermediate states in the resolution path
        
    Returns:
        List of σ values showing the resolution trajectory
    """
    all_states = [initial] + steps + [final]
    sigmas = []
    
    for state in all_states:
        sigma = state.distance_to(final)
        sigmas.append(sigma)
    
    return sigmas


def sigma_reduction(
    before: PhiState,
    after: PhiState,
    target: PhiState
) -> float:
    """
    Compute how much σ was reduced by a transformation.
    
    Args:
        before: State before transformation
        after: State after transformation
        target: Target state
        
    Returns:
        Δσ = σ_before - σ_after (positive = improvement)
    """
    sigma_before = before.distance_to(target)
    sigma_after = after.distance_to(target)
    return sigma_before - sigma_after
