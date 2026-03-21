"""
imagineAI v0.3 - GloVe + ICHTB + σ-Minimization

Architecture:
    Layer 3: Wikipedia/Facts (raw text)
    Layer 2: GloVe 300D (pre-trained semantic geometry)
    Layer 1: ICHTB 48D (structured ITT projection)
    Layer 0: σ-minimization (pure math solver)

What requires training: GloVe (but we inherit it)
What we build: ICHTB projection + σ-solver (no training)

Usage:
    python -m imagine_ai.v03_glove_ichtb.demo
    python -m imagine_ai.v03_glove_ichtb.demo --test
"""

from .semantic import GloVeSubstrate, SemanticPoint
from .ichtb import ICHTBSpace, ICHTBPoint, ICHTBProjection, Zone
from .solver import SigmaSolver, SolverConfig, ResolutionResult, ResolutionStatus, create_demo_solver

__all__ = [
    # Semantic layer
    'GloVeSubstrate', 'SemanticPoint',
    
    # ICHTB layer
    'ICHTBSpace', 'ICHTBPoint', 'ICHTBProjection', 'Zone',
    
    # Solver
    'SigmaSolver', 'SolverConfig', 'ResolutionResult', 'ResolutionStatus',
    'create_demo_solver',
]
