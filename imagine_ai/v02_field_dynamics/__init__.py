"""
imagineAI v0.2 - Pure Field Dynamics

This is the REAL imagineAI.
Answers emerge from ITT field collapse, not from training.

Usage:
    python -m imagine_ai.v02_field_dynamics.demo
"""

from .ichtb_space import (
    ICHTBPoint, ICHTBSpace, ICHTBProjector,
    Zone, Operator, DIM, I_ZERO
)

from .field_ops import (
    compute_gradient, compute_curl, compute_laplacian,
    compute_sigma, apply_master_equation,
    SigmaResult, Constraint, MasterCoefficients
)

from .resolver import (
    FieldResolver, FieldResolverConfig, ResolutionResult, ResolutionStatus,
    populate_space_from_text, create_demo_knowledge
)

__all__ = [
    # Space
    'ICHTBPoint', 'ICHTBSpace', 'ICHTBProjector',
    'Zone', 'Operator', 'DIM', 'I_ZERO',
    
    # Operators
    'compute_gradient', 'compute_curl', 'compute_laplacian',
    'compute_sigma', 'apply_master_equation',
    'SigmaResult', 'Constraint', 'MasterCoefficients',
    
    # Resolver
    'FieldResolver', 'FieldResolverConfig', 'ResolutionResult', 'ResolutionStatus',
    'populate_space_from_text', 'create_demo_knowledge',
]
