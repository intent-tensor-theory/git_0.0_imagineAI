"""
solver.py - The Emergence Solver

This is NOT retrieval.
This is field dynamics.

The question perturbs the semantic field.
The field evolves through the six zones.
The answer emerges where S > 1 and closure occurs.

From your ARC paper:
> ARC-AGI solvability emerges from pre-emergent field dynamics,
> providing an executable pathway from axioms to algorithms.

Same principle. Language instead of grids.
"""

import numpy as np
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .field import (
    SemanticField,
    initialize_field_from_question,
    text_to_embedding,
    compute_gradient,
    compute_curl,
    compute_laplacian
)
from .selection import compute_selection_number, SelectionResult
from .evolution import (
    evolve_to_closure,
    find_answer_in_substrate,
    EvolutionParameters,
    ClosureResult
)


# Stop words for anchor extraction
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
    'this', 'that', 'these', 'those', 'am', 'i', 'me', 'my', 'myself',
    'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'whose'
}


@dataclass
class EmergenceResult:
    """Result of the emergence process."""
    answer: Optional[str]           # The emerged answer
    confidence: float               # Similarity to stable config
    selection: SelectionResult      # Final S, R, Ṙ, t_ref
    locked: bool                    # Did closure occur?
    iterations: int                 # Evolution steps
    trace: List[Tuple[float, float]]  # (time, S) history
    
    @property
    def emerged(self) -> bool:
        """Did an answer emerge (S > 1 and locked)?"""
        return self.locked and self.selection.S > 1.0


class EmergenceSolver:
    """
    The Emergence Solver.
    
    Not retrieval. Field dynamics.
    
    The substrate (facts + GloVe) is the CTS for language.
    The question perturbs this substrate.
    The answer emerges where the field stabilizes with S > 1.
    """
    
    def __init__(self, glove, verbose: bool = False):
        """
        Initialize solver.
        
        Args:
            glove: GloVe KeyedVectors (the semantic substrate)
            verbose: Print evolution trace
        """
        self.glove = glove
        self.verbose = verbose
        
        # Knowledge substrate
        self.facts: List[str] = []
        self.fact_embeddings: List[np.ndarray] = []
        
        # Evolution parameters (from ICHTB)
        self.params = EvolutionParameters(
            D=0.1,      # Diffusivity
            Λ=0.05,     # Alignment decay
            γ=0.01,     # Nonlinear growth
            κ=0.02,     # Linear decay
            dt=0.1,     # Time step
            lock_threshold=0.01,
            S_threshold=1.0
        )
    
    def add_fact(self, fact: str):
        """Add a fact to the substrate."""
        self.facts.append(fact)
        emb = text_to_embedding(fact, self.glove)
        self.fact_embeddings.append(emb)
    
    def add_facts(self, facts: List[str]):
        """Add multiple facts to the substrate."""
        for fact in facts:
            self.add_fact(fact)
    
    def extract_anchors(self, question: str) -> Tuple[List[str], List[np.ndarray]]:
        """
        Extract anchors (ρ_q constraints) from question.
        
        Anchors are content words that define the boundary conditions.
        They constrain what configurations can satisfy the question.
        """
        words = re.findall(r'\b[a-z]+\b', question.lower())
        
        anchor_words = []
        anchor_vectors = []
        
        for word in words:
            if word not in STOP_WORDS and word in self.glove:
                anchor_words.append(word)
                anchor_vectors.append(self.glove[word])
        
        return anchor_words, anchor_vectors
    
    def get_substrate_points(self, field: SemanticField, k: int = 20) -> List[np.ndarray]:
        """
        Get nearby substrate points for evolution.
        
        These are the "neighbors" in semantic space that
        the field interacts with during evolution.
        """
        if not self.fact_embeddings:
            return []
        
        # Find k closest facts to current field state
        distances = []
        for emb in self.fact_embeddings:
            if np.linalg.norm(emb) > 0 and np.linalg.norm(field.Φ) > 0:
                sim = np.dot(field.Φ, emb) / (np.linalg.norm(field.Φ) * np.linalg.norm(emb))
                distances.append((emb, sim))
            else:
                distances.append((emb, 0.0))
        
        # Sort by similarity and take top k
        distances.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in distances[:k]]
    
    def solve(self, question: str, max_iterations: int = 100) -> EmergenceResult:
        """
        Solve via emergence.
        
        The question perturbs the field.
        The field evolves.
        The answer emerges.
        
        This is NOT retrieval.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EMERGENCE: {question}")
            print(f"{'='*60}")
        
        # 1. Extract anchors (ρ_q boundary conditions)
        anchor_words, anchors = self.extract_anchors(question)
        
        if self.verbose:
            print(f"Anchors (ρ_q): {anchor_words}")
        
        # 2. Initialize field from question
        field = initialize_field_from_question(question, self.glove)
        
        if self.verbose:
            print(f"Field initialized: |Φ₀| = {np.linalg.norm(field.Φ):.4f}")
        
        # 3. Get substrate points for evolution
        substrate_points = self.get_substrate_points(field)
        
        # 4. Context vectors (from anchors for curl computation)
        context_vectors = anchors if anchors else [field.Φ]
        
        # 5. Evolve to closure
        closure = evolve_to_closure(
            field=field,
            substrate_points=substrate_points,
            context_vectors=context_vectors,
            anchors=anchors,
            anchor_words=anchor_words,
            question=question,
            params=self.params,
            max_iterations=max_iterations,
            verbose=self.verbose
        )
        
        # 6. Find answer in substrate (with anchor constraints)
        answer, confidence = find_answer_in_substrate(
            Φ_final=closure.Φ_final,
            facts=self.facts,
            fact_embeddings=self.fact_embeddings,
            glove=self.glove,
            anchor_words=anchor_words
        )
        
        if self.verbose:
            print(f"\nRESULT:")
            print(f"  Locked: {closure.locked}")
            print(f"  S: {closure.S_final.S:.4f}")
            print(f"  Regime: {closure.S_final.regime}")
            print(f"  Answer: {answer}")
            print(f"  Confidence: {confidence:.4f}")
        
        return EmergenceResult(
            answer=answer,
            confidence=confidence,
            selection=closure.S_final,
            locked=closure.locked,
            iterations=closure.iterations,
            trace=closure.trace
        )


# =============================================================================
# Demo Knowledge Base (same as v0.5 for comparison)
# =============================================================================

DEMO_FACTS = [
    # US State Capitals
    "Jackson is the capital of Mississippi",
    "Montgomery is the capital of Alabama",
    "Austin is the capital of Texas",
    "Denver is the capital of Colorado",
    "Phoenix is the capital of Arizona",
    
    # World Capitals
    "Paris is the capital of France",
    "Tokyo is the capital of Japan",
    "Berlin is the capital of Germany",
    
    # Solar System
    "Saturn is famous for its beautiful rings",
    "Jupiter is the largest planet in our solar system",
    "Mars is known as the red planet",
    "Earth is the only planet known to support life",
    "Venus is the hottest planet in our solar system",
    "Mercury is the smallest planet and closest to the sun",
    
    # Geography
    "The Nile is the longest river in Africa",
    "Mount Everest is the tallest mountain on Earth",
    "The Pacific is the largest ocean on Earth",
    "The Amazon rainforest is in South America",
    
    # Science
    "Water freezes at zero degrees Celsius",
    "The speed of light is approximately 300000 kilometers per second",
    "DNA contains genetic information",
    "Photosynthesis converts sunlight into energy",
]


def create_demo_solver(glove, verbose: bool = False) -> EmergenceSolver:
    """Create solver with demo knowledge base."""
    solver = EmergenceSolver(glove, verbose=verbose)
    solver.add_facts(DEMO_FACTS)
    return solver
