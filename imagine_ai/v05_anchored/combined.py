"""
combined.py - Combined σ Calculation

σ_total = σ_dtw + λ * σ_anchor

Where:
- σ_dtw: Structural similarity via gradient DTW
- σ_anchor: Penalty for missing anchor words
- λ: Balancing weight

This combines:
- STRUCTURE (how meaning flows) 
- SPECIFICITY (which entities)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import re

# Import from v04 (reuse filament and DTW)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from imagine_ai.v04_filament.filament import Filament, FilamentFactory
from imagine_ai.v04_filament.dtw import filament_sigma as dtw_sigma

from .anchors import extract_anchors, anchor_sigma, AnchorResult


@dataclass
class CombinedResult:
    """Result of combined σ calculation."""
    sigma_total: float      # Combined σ
    sigma_dtw: float        # Structure component
    sigma_anchor: float     # Specificity component
    anchors: List[str]      # Query anchors
    anchor_matches: int     # How many anchors matched
    
    @property
    def structure_contribution(self) -> float:
        """How much of σ comes from structure."""
        if self.sigma_total == 0:
            return 0.0
        return self.sigma_dtw / self.sigma_total
    
    @property
    def specificity_contribution(self) -> float:
        """How much of σ comes from specificity."""
        if self.sigma_total == 0:
            return 0.0
        return (self.sigma_total - self.sigma_dtw) / self.sigma_total


def combined_sigma(
    query_filament: Filament,
    query_anchors: List[str],
    candidate_filament: Filament,
    candidate_words: List[str],
    lambda_weight: float = 1.0
) -> CombinedResult:
    """
    Calculate combined σ = σ_dtw + λ * σ_anchor.
    
    Args:
        query_filament: Question as filament (for DTW)
        query_anchors: Anchor words from question
        candidate_filament: Candidate answer as filament
        candidate_words: Words in candidate answer
        lambda_weight: Weight for anchor penalty
        
    Returns:
        CombinedResult with breakdown
    """
    # Structure: DTW distance
    sigma_d = dtw_sigma(query_filament, candidate_filament)
    
    # Handle infinite DTW (no gradients)
    if np.isinf(sigma_d):
        sigma_d = 2.0  # Max cosine distance
    
    # Specificity: Anchor presence
    sigma_a = anchor_sigma(query_anchors, candidate_words)
    
    # Count matches for reporting
    matches = sum(1 for a in query_anchors if a.lower() in set(w.lower() for w in candidate_words))
    
    # Combined
    sigma_total = sigma_d + lambda_weight * sigma_a
    
    return CombinedResult(
        sigma_total=sigma_total,
        sigma_dtw=sigma_d,
        sigma_anchor=sigma_a,
        anchors=query_anchors,
        anchor_matches=matches
    )


class CombinedMatcher:
    """
    Combines filament matching with anchor constraints.
    """
    
    def __init__(self, glove_model, lambda_weight: float = 1.0):
        """
        Args:
            glove_model: GloVe word vectors
            lambda_weight: Weight for anchor penalty
        """
        self.factory = FilamentFactory(glove_model)
        self.lambda_weight = lambda_weight
    
    def match(
        self,
        question: str,
        candidate: str
    ) -> CombinedResult:
        """
        Calculate combined σ between question and candidate.
        """
        # Create filaments
        q_filament = self.factory.create(question)
        c_filament = self.factory.create(candidate)
        
        # Extract anchors from question
        q_anchor_result = extract_anchors(question)
        
        # Get candidate words
        c_words = re.findall(r'\b[a-z]+\b', candidate.lower())
        
        return combined_sigma(
            query_filament=q_filament,
            query_anchors=q_anchor_result.anchors,
            candidate_filament=c_filament,
            candidate_words=c_words,
            lambda_weight=self.lambda_weight
        )
    
    def rank_candidates(
        self,
        question: str,
        candidates: List[str],
        n: int = 5
    ) -> List[Tuple[str, CombinedResult]]:
        """
        Rank candidates by combined σ.
        
        Args:
            question: The question
            candidates: List of candidate answers
            n: How many to return
            
        Returns:
            List of (candidate, result) sorted by σ_total
        """
        results = []
        
        for cand in candidates:
            result = self.match(question, cand)
            results.append((cand, result))
        
        # Sort by total σ
        results.sort(key=lambda x: x[1].sigma_total)
        
        return results[:n]


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Combined σ Test")
    print("=" * 60)
    
    # Load GloVe
    try:
        import gensim.downloader as api
        print("[GloVe] Loading...")
        glove = api.load("glove-wiki-gigaword-300")
        print("[GloVe] Loaded")
    except Exception as e:
        print(f"[GloVe] Error: {e}")
        glove = None
    
    matcher = CombinedMatcher(glove, lambda_weight=1.0)
    
    # Test
    question = "What is the capital of Mississippi?"
    candidates = [
        "Jackson is the capital of Mississippi.",
        "Montgomery is the capital of Alabama.",
        "Jupiter is the largest planet.",
    ]
    
    print(f"\nQuestion: {question}")
    print(f"Anchors: {extract_anchors(question).anchors}")
    
    results = matcher.rank_candidates(question, candidates)
    
    print("\nResults (sorted by σ_total):")
    for cand, result in results:
        print(f"\n  '{cand}'")
        print(f"    σ_total:  {result.sigma_total:.4f}")
        print(f"    σ_dtw:    {result.sigma_dtw:.4f}")
        print(f"    σ_anchor: {result.sigma_anchor:.4f}")
        print(f"    Anchor matches: {result.anchor_matches}/{len(result.anchors)}")
    
    print("\n✓ Combined σ working")
