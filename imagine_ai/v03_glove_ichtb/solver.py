"""
solver.py - The σ-Minimization Solver

This is the core of imagineAI v0.3.

Given a question, find the answer by minimizing σ (semantic distance)
in ICHTB space.

NO neural network. NO training. Pure math.

σ = 1 - cos(question, candidate)

The answer is whichever fact minimizes σ.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .semantic import GloVeSubstrate
from .ichtb import ICHTBSpace, ICHTBPoint, ICHTBProjection, Zone


class ResolutionStatus(Enum):
    """How the resolution terminated."""
    FOUND = "found"             # σ below threshold
    BEST_AVAILABLE = "best"     # Returned lowest σ even if above threshold
    NO_CANDIDATES = "empty"     # No facts in space


@dataclass 
class ResolutionResult:
    """Result of σ-minimization."""
    answer: Optional[str]           # The answer text
    sigma: float                    # Final σ value
    status: ResolutionStatus        # How it terminated
    rank: int = 1                   # Rank among all candidates
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class SolverConfig:
    """Solver configuration."""
    sigma_threshold: float = 0.7    # σ below this = confident answer
    return_alternatives: int = 3    # How many alternatives to return
    verbose: bool = False           # Debug output


class SigmaSolver:
    """
    The σ-minimization solver.
    
    Takes a question, searches ICHTB space, returns the
    fact with minimum semantic distance.
    
    This is NOT retrieval by keyword matching.
    This is geometric navigation in semantic space.
    """
    
    def __init__(
        self,
        space: ICHTBSpace,
        config: SolverConfig = None
    ):
        """
        Args:
            space: The ICHTB space containing facts
            config: Solver configuration
        """
        self.space = space
        self.config = config or SolverConfig()
    
    def solve(self, question: str) -> ResolutionResult:
        """
        Solve a question via σ-minimization.
        
        Args:
            question: The question text
            
        Returns:
            ResolutionResult with answer and σ
        """
        if self.space.size() == 0:
            return ResolutionResult(
                answer=None,
                sigma=float('inf'),
                status=ResolutionStatus.NO_CANDIDATES
            )
        
        # Embed question
        q_point = self.space.embed(question)
        
        if self.config.verbose:
            print(f"[Solver] Question: {question}")
            print(f"[Solver] Query vector norm: {np.linalg.norm(q_point.vector):.4f}")
        
        # Find minimum σ
        results = self.space.find_minimum_sigma(q_point, n=self.config.return_alternatives + 1)
        
        if not results:
            return ResolutionResult(
                answer=None,
                sigma=float('inf'),
                status=ResolutionStatus.NO_CANDIDATES
            )
        
        # Best result
        best_idx, best_point, best_sigma = results[0]
        
        # Alternatives
        alternatives = [(p.text, s) for _, p, s in results[1:]]
        
        if self.config.verbose:
            print(f"[Solver] Best σ: {best_sigma:.4f}")
            print(f"[Solver] Answer: {best_point.text[:60]}...")
        
        # Determine status
        if best_sigma <= self.config.sigma_threshold:
            status = ResolutionStatus.FOUND
        else:
            status = ResolutionStatus.BEST_AVAILABLE
        
        return ResolutionResult(
            answer=best_point.text,
            sigma=best_sigma,
            status=status,
            rank=1,
            alternatives=alternatives
        )
    
    def solve_with_trace(self, question: str) -> dict:
        """
        Solve with detailed trace of the search.
        
        Returns dict with full debugging info.
        """
        if self.space.size() == 0:
            return {"error": "No facts in space"}
        
        # Embed question
        q_point = self.space.embed(question)
        
        # Get all σ values
        all_results = self.space.find_minimum_sigma(q_point, n=self.space.size())
        
        # Zone analysis of question
        zone_mags = {
            zone.name: q_point.zone_magnitude(zone)
            for zone in Zone
        }
        
        return {
            "question": question,
            "question_vector_norm": float(np.linalg.norm(q_point.vector)),
            "zone_magnitudes": zone_mags,
            "results": [
                {
                    "rank": i + 1,
                    "text": point.text,
                    "sigma": float(sigma),
                    "similarity": float(1 - sigma)
                }
                for i, (_, point, sigma) in enumerate(all_results[:10])
            ],
            "total_candidates": self.space.size()
        }


# =============================================================================
# Create populated solver
# =============================================================================

def create_demo_solver(verbose: bool = False) -> SigmaSolver:
    """
    Create a solver with demo facts.
    
    Returns ready-to-use solver.
    """
    # Create space
    space = ICHTBSpace()
    
    # Demo knowledge base
    facts = [
        # State capitals
        "Jackson is the capital of Mississippi. It is located on the Pearl River.",
        "Austin is the capital of Texas. It is known for live music and technology.",
        "Sacramento is the capital of California. It is in the Central Valley.",
        "Albany is the capital of New York State, not New York City.",
        "Tallahassee is the capital of Florida. It is in the Florida Panhandle.",
        "Montgomery is the capital of Alabama.",
        "Baton Rouge is the capital of Louisiana.",
        "Atlanta is the capital of Georgia.",
        "Nashville is the capital of Tennessee.",
        "Little Rock is the capital of Arkansas.",
        
        # Planets
        "Jupiter is the largest planet in our solar system. It is a gas giant with many moons.",
        "Saturn is the second largest planet. It has beautiful rings made of ice and rock.",
        "Mercury is the smallest planet and closest to the Sun.",
        "Mars is called the Red Planet because of iron oxide on its surface.",
        "Venus is the hottest planet due to its thick atmosphere.",
        "Earth is the only known planet with life.",
        "Neptune is the farthest planet from the Sun.",
        "Uranus rotates on its side.",
        
        # Science
        "The speed of light is approximately 299,792,458 meters per second.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "The Earth is approximately 93 million miles from the Sun.",
        "DNA carries genetic information in all living organisms.",
        "Gravity is the force that attracts objects with mass toward each other.",
        "The human body has 206 bones.",
        
        # Geography
        "Mount Everest is the tallest mountain on Earth at 29,032 feet.",
        "The Pacific Ocean is the largest and deepest ocean.",
        "The Amazon River is the largest river by volume.",
        "The Sahara is the largest hot desert in the world.",
        "Antarctica is the coldest continent.",
        
        # History/Culture
        "Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.",
        "The Great Wall of China is over 13,000 miles long.",
        "The Eiffel Tower is located in Paris, France.",
        "The Pyramids of Giza were built around 2560 BCE.",
    ]
    
    print(f"[Solver] Loading {len(facts)} facts into ICHTB space...")
    space.add_batch(facts)
    print(f"[Solver] Space ready with {space.size()} facts")
    
    config = SolverConfig(
        sigma_threshold=0.7,
        return_alternatives=3,
        verbose=verbose
    )
    
    return SigmaSolver(space, config)


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("σ-Minimization Solver Test")
    print("=" * 60)
    
    solver = create_demo_solver(verbose=True)
    
    questions = [
        "What is the capital of Mississippi?",
        "What is the largest planet?",
        "How fast is light?",
        "What is the tallest mountain?",
        "Who wrote Hamlet?",
    ]
    
    print("\n" + "=" * 60)
    print("SOLVING QUESTIONS")
    print("=" * 60)
    
    for q in questions:
        print(f"\n>>> {q}")
        result = solver.solve(q)
        
        print(f"Status: {result.status.value}")
        print(f"σ: {result.sigma:.4f}")
        print(f"Answer: {result.answer}")
        
        if result.alternatives:
            print("Alternatives:")
            for alt, sigma in result.alternatives[:2]:
                print(f"  σ={sigma:.4f}: {alt[:50]}...")
    
    print("\n✓ Solver working")
