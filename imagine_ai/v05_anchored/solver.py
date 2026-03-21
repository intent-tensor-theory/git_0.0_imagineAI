"""
solver.py - Anchored Filament Solver

imagineAI v0.5 Core

Combines:
- STRUCTURE via gradient DTW (from v0.4)
- SPECIFICITY via anchor word matching (new)

σ_total = σ_dtw + λ * σ_anchor

This should capture what v0.3 got right (word matching)
AND what v0.4 got right (structure matching).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import re

# Import from v04
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from imagine_ai.v04_filament.filament import Filament, FilamentFactory
from imagine_ai.v04_filament.dtw import filament_sigma as dtw_sigma

from .anchors import extract_anchors, anchor_sigma
from .combined import combined_sigma, CombinedResult


class ResolutionStatus(Enum):
    """How the resolution terminated."""
    FOUND = "found"
    BEST_AVAILABLE = "best"
    NO_CANDIDATES = "empty"


@dataclass
class ResolutionResult:
    """Result of anchored resolution."""
    answer: Optional[str]
    sigma_total: float
    sigma_dtw: float
    sigma_anchor: float
    anchors: List[str]
    anchor_matches: int
    status: ResolutionStatus
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class SolverConfig:
    """Solver configuration."""
    sigma_threshold: float = 0.8    # σ below this = confident
    lambda_weight: float = 1.0      # Anchor penalty weight
    return_alternatives: int = 3
    verbose: bool = False


class AnchoredSolver:
    """
    The Anchored Filament Solver.
    
    Combines:
    - Gradient DTW (structural similarity)
    - Anchor matching (specific entities)
    
    σ_total = σ_dtw + λ * σ_anchor
    """
    
    def __init__(
        self,
        glove_model,
        config: SolverConfig = None
    ):
        """
        Args:
            glove_model: GloVe word vectors
            config: Solver configuration
        """
        self.config = config or SolverConfig()
        self.factory = FilamentFactory(glove_model)
        self.glove = glove_model
        
        self.facts: List[str] = []
        self.fact_filaments: List[Filament] = []
        self.fact_words: List[List[str]] = []
    
    def add_fact(self, text: str) -> int:
        """Add a fact to knowledge base."""
        filament = self.factory.create(text)
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        idx = len(self.facts)
        self.facts.append(text)
        self.fact_filaments.append(filament)
        self.fact_words.append(words)
        
        return idx
    
    def add_facts(self, texts: List[str]) -> List[int]:
        """Add multiple facts."""
        return [self.add_fact(t) for t in texts]
    
    def size(self) -> int:
        """Number of facts."""
        return len(self.facts)
    
    def solve(self, question: str) -> ResolutionResult:
        """
        Solve using combined σ.
        """
        if self.size() == 0:
            return ResolutionResult(
                answer=None,
                sigma_total=float('inf'),
                sigma_dtw=float('inf'),
                sigma_anchor=1.0,
                anchors=[],
                anchor_matches=0,
                status=ResolutionStatus.NO_CANDIDATES
            )
        
        # Create question filament
        q_filament = self.factory.create(question)
        
        # Extract anchors
        anchor_result = extract_anchors(question)
        q_anchors = anchor_result.anchors
        
        if self.config.verbose:
            print(f"[Solver] Question: {question}")
            print(f"[Solver] Anchors: {q_anchors}")
            print(f"[Solver] Gradients: {q_filament.num_gradients}")
        
        # Score all facts
        scored = []
        for i in range(self.size()):
            result = combined_sigma(
                query_filament=q_filament,
                query_anchors=q_anchors,
                candidate_filament=self.fact_filaments[i],
                candidate_words=self.fact_words[i],
                lambda_weight=self.config.lambda_weight
            )
            scored.append((i, result))
        
        # Sort by total σ
        scored.sort(key=lambda x: x[1].sigma_total)
        
        # Best result
        best_idx, best_result = scored[0]
        
        # Alternatives
        alternatives = [
            (self.facts[idx], result.sigma_total)
            for idx, result in scored[1:self.config.return_alternatives + 1]
        ]
        
        if self.config.verbose:
            print(f"[Solver] Best σ_total: {best_result.sigma_total:.4f}")
            print(f"[Solver] Best σ_dtw: {best_result.sigma_dtw:.4f}")
            print(f"[Solver] Best σ_anchor: {best_result.sigma_anchor:.4f}")
            print(f"[Solver] Answer: {self.facts[best_idx][:60]}...")
        
        # Status
        if best_result.sigma_total <= self.config.sigma_threshold:
            status = ResolutionStatus.FOUND
        else:
            status = ResolutionStatus.BEST_AVAILABLE
        
        return ResolutionResult(
            answer=self.facts[best_idx],
            sigma_total=best_result.sigma_total,
            sigma_dtw=best_result.sigma_dtw,
            sigma_anchor=best_result.sigma_anchor,
            anchors=q_anchors,
            anchor_matches=best_result.anchor_matches,
            status=status,
            alternatives=alternatives
        )
    
    def solve_with_trace(self, question: str) -> dict:
        """Solve with full trace."""
        q_filament = self.factory.create(question)
        anchor_result = extract_anchors(question)
        q_anchors = anchor_result.anchors
        
        all_results = []
        for i in range(self.size()):
            result = combined_sigma(
                query_filament=q_filament,
                query_anchors=q_anchors,
                candidate_filament=self.fact_filaments[i],
                candidate_words=self.fact_words[i],
                lambda_weight=self.config.lambda_weight
            )
            all_results.append({
                "text": self.facts[i],
                "sigma_total": result.sigma_total,
                "sigma_dtw": result.sigma_dtw,
                "sigma_anchor": result.sigma_anchor,
                "anchor_matches": result.anchor_matches,
            })
        
        # Sort and rank
        all_results.sort(key=lambda x: x["sigma_total"])
        for i, r in enumerate(all_results):
            r["rank"] = i + 1
        
        return {
            "question": question,
            "anchors": q_anchors,
            "num_gradients": q_filament.num_gradients,
            "lambda": self.config.lambda_weight,
            "results": all_results[:10],
            "total_facts": self.size()
        }


# =============================================================================
# Demo Knowledge Base
# =============================================================================

def get_demo_facts() -> List[str]:
    """Demo knowledge base."""
    return [
        # Capitals
        "Jackson is the capital of Mississippi located on the Pearl River.",
        "Austin is the capital of Texas known for live music.",
        "Sacramento is the capital of California.",
        "Albany is the capital of New York State.",
        "Tallahassee is the capital of Florida.",
        "Montgomery is the capital of Alabama.",
        "Baton Rouge is the capital of Louisiana.",
        "Atlanta is the capital of Georgia.",
        
        # Planets
        "Jupiter is the largest planet in our solar system.",
        "Saturn is famous for its beautiful rings made of ice.",
        "Mercury is the smallest planet closest to the Sun.",
        "Mars is called the Red Planet due to iron oxide.",
        "Venus is the hottest planet.",
        "Earth is the only known planet with life.",
        
        # Science
        "The speed of light is approximately 299792458 meters per second.",
        "Light travels at about 300000 kilometers per second.",
        "Water boils at 100 degrees Celsius.",
        "Gravity pulls objects toward each other.",
        
        # Geography
        "Mount Everest is the tallest mountain on Earth.",
        "The Pacific Ocean is the largest ocean.",
        "The Amazon is the largest river by volume.",
        
        # Culture
        "Shakespeare wrote Hamlet Macbeth and Romeo and Juliet.",
        "William Shakespeare was an English playwright.",
        "The Eiffel Tower is located in Paris France.",
    ]


def create_demo_solver(verbose: bool = False) -> AnchoredSolver:
    """Create solver with demo facts."""
    try:
        import gensim.downloader as api
        print("[GloVe] Loading glove-wiki-gigaword-300...")
        glove = api.load("glove-wiki-gigaword-300")
        print(f"[GloVe] Loaded. Vocab: {len(glove)}")
    except Exception as e:
        print(f"[GloVe] Error: {e}")
        glove = None
    
    config = SolverConfig(
        sigma_threshold=0.8,
        lambda_weight=1.0,
        return_alternatives=3,
        verbose=verbose
    )
    
    solver = AnchoredSolver(glove, config)
    
    facts = get_demo_facts()
    print(f"[Solver] Loading {len(facts)} facts...")
    solver.add_facts(facts)
    print(f"[Solver] Ready with {solver.size()} facts")
    
    return solver


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Anchored Solver Test")
    print("=" * 60)
    
    solver = create_demo_solver(verbose=True)
    
    questions = [
        "What is the capital of Mississippi?",
        "What is the largest planet?",
        "How fast does light travel?",
        "What planet has rings?",
        "Who wrote Hamlet?",
    ]
    
    print("\n" + "=" * 60)
    print("SOLVING")
    print("=" * 60)
    
    for q in questions:
        print(f"\n>>> {q}")
        result = solver.solve(q)
        
        print(f"Anchors: {result.anchors}")
        print(f"σ_total: {result.sigma_total:.4f} (dtw={result.sigma_dtw:.4f}, anchor={result.sigma_anchor:.4f})")
        print(f"Matches: {result.anchor_matches}/{len(result.anchors)}")
        print(f"Answer: {result.answer}")
    
    print("\n✓ Anchored solver working")
