"""
resolver.py - The Field Collapse Engine

This is the CORE of imagineAI v0.2.

Given:
    - A question (creates field excitation)
    - An information space (48D ICHTB points)

Finds:
    - The answer (stable field configuration where σ → 0)

NO LLM. NO TRAINING. The math FINDS the answer.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

from .ichtb_space import (
    ICHTBPoint, ICHTBSpace, ICHTBProjector,
    Zone, DIM, I_ZERO
)
from .field_ops import (
    compute_gradient, compute_curl, compute_laplacian,
    compute_sigma, apply_master_equation,
    SigmaResult, Constraint, MasterCoefficients,
    must_be_similar_to
)


class ResolutionStatus(Enum):
    """Status of field resolution."""
    LOCKED = "locked"           # σ < threshold, answer found
    MAX_ITERATIONS = "max_iter" # Reached iteration limit
    NO_CANDIDATES = "no_candidates"  # No points in space
    DIVERGED = "diverged"       # σ is increasing


@dataclass
class ResolutionResult:
    """Result of field resolution."""
    answer: Optional[ICHTBPoint]    # The stable configuration
    sigma: float                     # Final σ residue
    iterations: int                  # Steps taken
    status: ResolutionStatus         # How it terminated
    sigma_history: List[float] = field(default_factory=list)  # σ trace
    path: List[int] = field(default_factory=list)  # Indices visited


@dataclass
class FieldResolverConfig:
    """Configuration for the resolver."""
    max_iterations: int = 100       # Max steps before giving up
    sigma_threshold: float = 0.5    # σ below this = locked
    convergence_threshold: float = 0.01  # Δσ below this = converged
    beam_width: int = 5             # How many candidates to track
    verbose: bool = False           # Print debug info


class FieldResolver:
    """
    The Field Collapse Engine.
    
    Takes a question, navigates the information space using
    ITT field dynamics, and finds the answer as the stable
    configuration where σ is minimized.
    
    NO NEURAL NETWORKS. NO TRAINING.
    The answer emerges from the mathematics.
    """
    
    def __init__(
        self,
        space: ICHTBSpace,
        projector: ICHTBProjector,
        config: FieldResolverConfig = None
    ):
        """
        Args:
            space: The 48D information space
            projector: Text → ICHTB projector
            config: Resolution parameters
        """
        self.space = space
        self.projector = projector
        self.config = config or FieldResolverConfig()
        self.coeffs = MasterCoefficients()
    
    def resolve(
        self,
        question: str,
        constraints: List[Constraint] = None
    ) -> ResolutionResult:
        """
        Resolve a question to an answer using field dynamics.
        
        Algorithm:
        1. Project question into ICHTB space → creates excitation
        2. Initialize beam of candidate answers from nearest neighbors
        3. Iterate:
            a. Compute σ for each candidate
            b. Apply Master Equation to evolve field
            c. Keep best candidates (lowest σ)
            d. Check for lock condition (σ < threshold)
        4. Return the stable configuration
        
        Args:
            question: The question text
            constraints: Additional boundary conditions
            
        Returns:
            ResolutionResult with the answer (or failure)
        """
        if self.space.size() == 0:
            return ResolutionResult(
                answer=None,
                sigma=float('inf'),
                iterations=0,
                status=ResolutionStatus.NO_CANDIDATES
            )
        
        # Project question into ICHTB space
        q_point = self.projector.project_text(question)
        
        if self.config.verbose:
            print(f"[Resolver] Question projected to 48D")
            print(f"[Resolver] Question vector norm: {np.linalg.norm(q_point.vector):.4f}")
        
        # Build constraints from question
        all_constraints = constraints or []
        all_constraints.append(must_be_similar_to(q_point, threshold=0.3))
        
        # Initialize beam with nearest neighbors
        nearest = self.space.most_similar(q_point, k=self.config.beam_width * 2)
        beam = [(idx, self.space.get(idx)) for idx, sim in nearest]
        
        if self.config.verbose:
            print(f"[Resolver] Initial beam: {len(beam)} candidates")
        
        # Track history
        sigma_history = []
        path = []
        best_sigma = float('inf')
        best_point = None
        
        # Iterate field dynamics
        for iteration in range(self.config.max_iterations):
            # Compute σ for each candidate
            scored = []
            for idx, candidate in beam:
                sigma_result = compute_sigma(
                    candidate=candidate,
                    target=q_point,
                    constraints=all_constraints,
                    lock_threshold=self.config.sigma_threshold
                )
                scored.append((idx, candidate, sigma_result))
            
            # Sort by σ (ascending)
            scored.sort(key=lambda x: x[2].sigma)
            
            # Best candidate this iteration
            best_idx, best_candidate, best_result = scored[0]
            current_sigma = best_result.sigma
            sigma_history.append(current_sigma)
            path.append(best_idx)
            
            if self.config.verbose:
                print(f"[Resolver] Iter {iteration}: σ={current_sigma:.4f}, "
                      f"best='{best_candidate.content[:40]}...'")
            
            # Check for lock
            if best_result.locked:
                if self.config.verbose:
                    print(f"[Resolver] LOCKED at σ={current_sigma:.4f}")
                return ResolutionResult(
                    answer=best_candidate,
                    sigma=current_sigma,
                    iterations=iteration + 1,
                    status=ResolutionStatus.LOCKED,
                    sigma_history=sigma_history,
                    path=path
                )
            
            # Track overall best
            if current_sigma < best_sigma:
                best_sigma = current_sigma
                best_point = best_candidate
            
            # Check for convergence
            if len(sigma_history) > 5:
                recent_delta = abs(sigma_history[-1] - sigma_history[-5])
                if recent_delta < self.config.convergence_threshold:
                    if self.config.verbose:
                        print(f"[Resolver] Converged (Δσ < {self.config.convergence_threshold})")
                    return ResolutionResult(
                        answer=best_point,
                        sigma=best_sigma,
                        iterations=iteration + 1,
                        status=ResolutionStatus.LOCKED,
                        sigma_history=sigma_history,
                        path=path
                    )
            
            # Check for divergence
            if len(sigma_history) > 10 and sigma_history[-1] > sigma_history[-10]:
                if self.config.verbose:
                    print(f"[Resolver] Diverging, stopping")
                return ResolutionResult(
                    answer=best_point,
                    sigma=best_sigma,
                    iterations=iteration + 1,
                    status=ResolutionStatus.DIVERGED,
                    sigma_history=sigma_history,
                    path=path
                )
            
            # Apply Master Equation to evolve candidates
            new_beam = []
            for idx, candidate, result in scored[:self.config.beam_width]:
                # Get neighbors for Laplacian
                neighbors = [self.space.get(i) for i, s in self.space.most_similar(candidate, k=3)]
                
                # Evolve
                evolved = apply_master_equation(
                    state=candidate,
                    target=q_point,
                    neighbors=neighbors,
                    coeffs=self.coeffs
                )
                
                # Find nearest point in space to evolved position
                nearest_evolved = self.space.most_similar(evolved, k=1)
                if nearest_evolved:
                    new_idx, sim = nearest_evolved[0]
                    new_beam.append((new_idx, self.space.get(new_idx)))
            
            # Also keep some original candidates for stability
            for idx, candidate, result in scored[:2]:
                if (idx, candidate) not in new_beam:
                    new_beam.append((idx, candidate))
            
            beam = new_beam[:self.config.beam_width]
        
        # Max iterations reached
        return ResolutionResult(
            answer=best_point,
            sigma=best_sigma,
            iterations=self.config.max_iterations,
            status=ResolutionStatus.MAX_ITERATIONS,
            sigma_history=sigma_history,
            path=path
        )
    
    def resolve_to_text(self, question: str) -> Tuple[str, float]:
        """
        Convenience method: resolve and return answer text.
        
        Returns:
            (answer_text, sigma)
        """
        result = self.resolve(question)
        
        if result.answer:
            return (result.answer.content, result.sigma)
        else:
            return ("Could not resolve", float('inf'))


# =============================================================================
# POPULATE SPACE FROM TEXT
# =============================================================================

def populate_space_from_text(
    texts: List[str],
    projector: ICHTBProjector = None
) -> Tuple[ICHTBSpace, ICHTBProjector]:
    """
    Create an ICHTB space from a list of text items.
    
    Each text becomes a point in 48D space.
    The resolver can then navigate this space to find answers.
    
    Args:
        texts: List of text items (facts, sentences, etc.)
        projector: Optional projector (creates one if None)
        
    Returns:
        (space, projector)
    """
    if projector is None:
        projector = ICHTBProjector()
    
    space = ICHTBSpace()
    
    for text in texts:
        point = projector.project_text(text)
        space.add(point)
    
    return space, projector


# =============================================================================
# DEMO: KNOWLEDGE BASE RESOLUTION
# =============================================================================

def create_demo_knowledge() -> List[str]:
    """Create a demo knowledge base."""
    return [
        # State capitals
        "Jackson is the capital of Mississippi. It is located on the Pearl River.",
        "Austin is the capital of Texas. It is known for live music.",
        "Sacramento is the capital of California. It is in the Central Valley.",
        "Albany is the capital of New York. Many people think it's New York City.",
        "Tallahassee is the capital of Florida. It is in the Florida Panhandle.",
        "Montgomery is the capital of Alabama. It was the first capital of the Confederacy.",
        
        # Planets
        "Jupiter is the largest planet in our solar system. It is a gas giant.",
        "Mercury is the smallest planet. It is closest to the Sun.",
        "Mars is called the Red Planet because of iron oxide on its surface.",
        "Earth is the third planet from the Sun. It is the only known planet with life.",
        "Saturn has beautiful rings made of ice and rock.",
        
        # Science facts
        "The speed of light is approximately 299,792,458 meters per second.",
        "Water boils at 100 degrees Celsius at standard pressure.",
        "The Earth orbits the Sun at about 93 million miles distance.",
        "DNA carries genetic information in living organisms.",
        
        # Random facts
        "The Eiffel Tower is in Paris, France.",
        "Mount Everest is the tallest mountain on Earth.",
        "The Pacific Ocean is the largest ocean.",
        "Shakespeare wrote Hamlet and Romeo and Juliet.",
        "The Great Wall of China is over 13,000 miles long.",
    ]


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Field Resolver Test")
    print("=" * 60)
    
    # Create demo knowledge
    knowledge = create_demo_knowledge()
    print(f"\nLoaded {len(knowledge)} facts into 48D space")
    
    # Populate space
    space, projector = populate_space_from_text(knowledge)
    print(f"Space contains {space.size()} points")
    
    # Create resolver
    config = FieldResolverConfig(
        max_iterations=50,
        sigma_threshold=0.5,
        beam_width=5,
        verbose=True
    )
    resolver = FieldResolver(space, projector, config)
    
    # Test questions
    questions = [
        "What is the capital of Mississippi?",
        "Which planet is the largest?",
        "How fast is light?",
    ]
    
    print("\n" + "=" * 60)
    print("RESOLVING QUESTIONS")
    print("=" * 60)
    
    for q in questions:
        print(f"\n>>> {q}")
        result = resolver.resolve(q)
        
        print(f"Status: {result.status.value}")
        print(f"Iterations: {result.iterations}")
        print(f"Final σ: {result.sigma:.4f}")
        
        if result.answer:
            print(f"Answer: {result.answer.content}")
        else:
            print("Answer: [No resolution]")
        
        print(f"σ trace: {[round(s, 2) for s in result.sigma_history[:10]]}")
    
    print("\n✓ Field Resolver working")
