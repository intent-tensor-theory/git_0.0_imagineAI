"""
solver.py - Filament σ-Minimization Solver

imagineAI v0.4 Core

Given a question, finds the answer by:
1. Converting question to filament (gradient tensor)
2. DTW comparison against fact filaments
3. Return fact with minimum σ

NO NEURAL NETWORK. Pure gradient flow + dynamic programming.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from .filament import Filament, FilamentFactory
from .dtw import filament_sigma, find_minimum_sigma


class ResolutionStatus(Enum):
    """How the resolution terminated."""
    FOUND = "found"             # σ below threshold
    BEST_AVAILABLE = "best"     # Lowest σ but above threshold
    NO_CANDIDATES = "empty"     # No facts


@dataclass
class ResolutionResult:
    """Result of filament resolution."""
    answer: Optional[str]
    sigma: float
    status: ResolutionStatus
    query_info: dict = field(default_factory=dict)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class SolverConfig:
    """Solver configuration."""
    sigma_threshold: float = 0.5    # σ below this = confident
    return_alternatives: int = 3    # How many alternatives
    verbose: bool = False


class FilamentSolver:
    """
    The Filament σ-Minimization Solver.
    
    Sentences are PATHS through semantic space (filaments).
    σ = DTW distance between gradient tensors.
    The answer is the fact with minimum σ.
    
    This captures STRUCTURAL similarity, not just word overlap.
    """
    
    def __init__(
        self,
        glove_model,
        config: SolverConfig = None
    ):
        """
        Args:
            glove_model: Gensim GloVe model
            config: Solver configuration
        """
        self.config = config or SolverConfig()
        self.factory = FilamentFactory(glove_model)
        self.facts: List[Filament] = []
        self.fact_texts: List[str] = []
    
    def add_fact(self, text: str) -> int:
        """Add a fact to the knowledge base."""
        filament = self.factory.create(text)
        idx = len(self.facts)
        self.facts.append(filament)
        self.fact_texts.append(text)
        return idx
    
    def add_facts(self, texts: List[str]) -> List[int]:
        """Add multiple facts."""
        return [self.add_fact(t) for t in texts]
    
    def size(self) -> int:
        """Number of facts."""
        return len(self.facts)
    
    def solve(self, question: str) -> ResolutionResult:
        """
        Solve a question via filament σ-minimization.
        
        Args:
            question: The question text
            
        Returns:
            ResolutionResult with answer
        """
        if self.size() == 0:
            return ResolutionResult(
                answer=None,
                sigma=float('inf'),
                status=ResolutionStatus.NO_CANDIDATES
            )
        
        # Create question filament
        q_filament = self.factory.create(question)
        
        if self.config.verbose:
            print(f"[Solver] Question: {question}")
            print(f"[Solver] Words: {q_filament.words}")
            print(f"[Solver] Gradients: {q_filament.num_gradients}")
            print(f"[Solver] Curvature: {q_filament.curvature:.4f}")
        
        if q_filament.num_gradients == 0:
            # Can't compare without gradients
            return ResolutionResult(
                answer=None,
                sigma=float('inf'),
                status=ResolutionStatus.NO_CANDIDATES,
                query_info={"error": "Question too short"}
            )
        
        # Find minimum σ
        n_results = self.config.return_alternatives + 1
        results = find_minimum_sigma(q_filament, self.facts, n=n_results)
        
        if not results:
            return ResolutionResult(
                answer=None,
                sigma=float('inf'),
                status=ResolutionStatus.NO_CANDIDATES
            )
        
        # Best result
        best_idx, best_filament, best_sigma = results[0]
        
        # Alternatives
        alternatives = [
            (self.fact_texts[idx], sigma)
            for idx, _, sigma in results[1:]
        ]
        
        if self.config.verbose:
            print(f"[Solver] Best σ: {best_sigma:.4f}")
            print(f"[Solver] Answer: {self.fact_texts[best_idx][:60]}...")
        
        # Status
        if best_sigma <= self.config.sigma_threshold:
            status = ResolutionStatus.FOUND
        else:
            status = ResolutionStatus.BEST_AVAILABLE
        
        # Query info
        query_info = {
            "words": q_filament.words,
            "num_gradients": q_filament.num_gradients,
            "curvature": q_filament.curvature,
            "path_length": q_filament.total_path_length
        }
        
        return ResolutionResult(
            answer=self.fact_texts[best_idx],
            sigma=best_sigma,
            status=status,
            query_info=query_info,
            alternatives=alternatives
        )
    
    def solve_with_trace(self, question: str) -> dict:
        """
        Solve with full debugging trace.
        """
        q_filament = self.factory.create(question)
        
        all_sigmas = []
        for i, fact in enumerate(self.facts):
            sigma = filament_sigma(q_filament, fact)
            all_sigmas.append({
                "rank": None,  # Filled after sort
                "text": self.fact_texts[i],
                "sigma": sigma,
                "fact_gradients": fact.num_gradients,
                "fact_curvature": fact.curvature
            })
        
        # Sort and assign ranks
        all_sigmas.sort(key=lambda x: x["sigma"])
        for i, item in enumerate(all_sigmas):
            item["rank"] = i + 1
        
        return {
            "question": question,
            "question_words": q_filament.words,
            "question_gradients": q_filament.num_gradients,
            "question_curvature": q_filament.curvature,
            "results": all_sigmas[:10],
            "total_facts": self.size()
        }


# =============================================================================
# Demo Knowledge Base
# =============================================================================

def get_demo_facts() -> List[str]:
    """Get demo knowledge base."""
    return [
        # Capitals
        "Jackson is the capital of Mississippi located on the Pearl River.",
        "Austin is the capital of Texas known for live music and technology.",
        "Sacramento is the capital of California in the Central Valley.",
        "Albany is the capital of New York State.",
        "Tallahassee is the capital of Florida.",
        "Montgomery is the capital of Alabama.",
        "Baton Rouge is the capital of Louisiana.",
        "Atlanta is the capital of Georgia.",
        "Nashville is the capital of Tennessee.",
        
        # Planets
        "Jupiter is the largest planet in our solar system.",
        "Saturn is famous for its beautiful rings made of ice.",
        "Mercury is the smallest planet closest to the Sun.",
        "Mars is called the Red Planet due to iron oxide.",
        "Venus is the hottest planet because of its atmosphere.",
        "Earth is the only known planet with life.",
        
        # Science  
        "The speed of light is approximately 299792458 meters per second.",
        "Light travels at about 300000 kilometers per second.",
        "Water boils at 100 degrees Celsius.",
        "The Earth is 93 million miles from the Sun.",
        "Gravity pulls objects toward each other.",
        
        # Geography
        "Mount Everest is the tallest mountain on Earth.",
        "The Pacific Ocean is the largest ocean.",
        "The Amazon is the largest river by volume.",
        "The Sahara is the largest hot desert.",
        
        # Culture
        "Shakespeare wrote Hamlet Macbeth and Romeo and Juliet.",
        "William Shakespeare was an English playwright.",
        "The Great Wall of China is over 13000 miles long.",
        "The Eiffel Tower is located in Paris France.",
        "The Pyramids of Giza were built around 2560 BCE.",
    ]


def create_demo_solver(verbose: bool = False) -> FilamentSolver:
    """Create solver with demo facts."""
    # Load GloVe
    try:
        import gensim.downloader as api
        print("[GloVe] Loading glove-wiki-gigaword-300...")
        glove = api.load("glove-wiki-gigaword-300")
        print(f"[GloVe] Loaded. Vocab: {len(glove)}")
    except Exception as e:
        print(f"[GloVe] Error: {e}")
        print("[GloVe] Using fallback (results will be poor)")
        glove = None
    
    config = SolverConfig(
        sigma_threshold=0.5,
        return_alternatives=3,
        verbose=verbose
    )
    
    solver = FilamentSolver(glove, config)
    
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
    print("Filament Solver Test")
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
        
        print(f"Status: {result.status.value}")
        print(f"σ: {result.sigma:.4f}")
        print(f"Answer: {result.answer}")
        
        if result.alternatives:
            print("Alternatives:")
            for alt, sigma in result.alternatives[:2]:
                print(f"  σ={sigma:.4f}: {alt[:40]}...")
    
    print("\n✓ Filament solver working")
