"""
imagineAI v0.4 - Semantic Filaments

THE BREAKTHROUGH:
    A sentence is not a POINT. It's a PATH through semantic space.
    The GRADIENTS between words encode how meaning FLOWS.
    
Averaging word vectors throws away structure.
Gradient tensors PRESERVE structure.

Architecture:
    Text → Words → GloVe vectors → Gradient tensor → DTW comparison → σ
    
What's new:
    - Filament = gradient sequence (not word average)
    - DTW aligns sequences of different lengths
    - σ = structural similarity, not just word overlap

What's NOT trained:
    - Gradients = vector subtraction
    - DTW = dynamic programming
    - Everything except GloVe (which we inherit)

Usage:
    python -m imagine_ai.v04_filament.demo
    python -m imagine_ai.v04_filament.demo --test
"""

from .filament import Filament, FilamentFactory, gradient_overlap, filament_similarity_simple
from .dtw import DTWResult, cosine_distance, dtw_gradients, filament_sigma, find_minimum_sigma
from .solver import (
    FilamentSolver, SolverConfig, ResolutionResult, ResolutionStatus,
    create_demo_solver, get_demo_facts
)

__all__ = [
    # Filaments
    'Filament', 'FilamentFactory', 'gradient_overlap', 'filament_similarity_simple',
    
    # DTW
    'DTWResult', 'cosine_distance', 'dtw_gradients', 'filament_sigma', 'find_minimum_sigma',
    
    # Solver
    'FilamentSolver', 'SolverConfig', 'ResolutionResult', 'ResolutionStatus',
    'create_demo_solver', 'get_demo_facts',
]
