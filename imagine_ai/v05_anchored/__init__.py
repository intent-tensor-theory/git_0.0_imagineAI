"""
imagineAI v0.5 - Anchored Filaments

THE EVOLUTION:
    v0.3: Specificity (word presence) → 37%
    v0.4: Structure (gradient flow) → 12%
    v0.5: BOTH → ?%

σ_total = σ_dtw + λ * σ_anchor

Where:
- σ_dtw: Gradient DTW distance (structure)
- σ_anchor: Missing anchor penalty (specificity)
- λ: Balancing weight (default 1.0)

This is the ρ_q (boundary condition) concept from ITT:
- Gradient DTW = field dynamics
- Anchors = boundary constraints

Usage:
    python -m imagine_ai.v05_anchored.demo
    python -m imagine_ai.v05_anchored.demo --test
"""

from .anchors import extract_anchors, anchor_sigma, anchor_overlap, AnchorResult
from .combined import combined_sigma, CombinedResult, CombinedMatcher
from .solver import (
    AnchoredSolver, SolverConfig, ResolutionResult, ResolutionStatus,
    create_demo_solver, get_demo_facts
)

__all__ = [
    # Anchors
    'extract_anchors', 'anchor_sigma', 'anchor_overlap', 'AnchorResult',
    
    # Combined
    'combined_sigma', 'CombinedResult', 'CombinedMatcher',
    
    # Solver
    'AnchoredSolver', 'SolverConfig', 'ResolutionResult', 'ResolutionStatus',
    'create_demo_solver', 'get_demo_facts',
]
