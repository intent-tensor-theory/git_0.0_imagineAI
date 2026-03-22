"""
imagineAI v0.6 - Emergence Architecture

Not retrieval. Field dynamics.

The question perturbs the semantic field.
The field evolves through the six zones.
The answer emerges where S > 1 and closure occurs.

From Intent Tensor Theory:
- CTS provides the substrate
- ICHTB provides the zone dynamics
- Selection Number S determines persistence

The math finds the answer.
"""

from .field import (
    SemanticField,
    initialize_field_from_question,
    text_to_embedding,
    compute_gradient,
    compute_curl,
    compute_laplacian,
    compute_temporal_derivative,
    I_0
)

from .selection import (
    compute_selection_number,
    compute_retained_structure,
    compute_loss_rate,
    compute_reference_timescale,
    SelectionResult,
    SelectionLandscape,
    build_selection_landscape
)

from .evolution import (
    master_equation_step,
    evolve_to_closure,
    find_answer_in_substrate,
    EvolutionParameters,
    ClosureResult
)

from .solver import (
    EmergenceSolver,
    EmergenceResult,
    create_demo_solver,
    DEMO_FACTS
)

__version__ = "0.6.0"
__all__ = [
    # Field
    "SemanticField",
    "initialize_field_from_question",
    "text_to_embedding",
    "compute_gradient",
    "compute_curl",
    "compute_laplacian",
    "I_0",
    
    # Selection
    "compute_selection_number",
    "SelectionResult",
    "SelectionLandscape",
    
    # Evolution
    "evolve_to_closure",
    "EvolutionParameters",
    "ClosureResult",
    
    # Solver
    "EmergenceSolver",
    "EmergenceResult",
    "create_demo_solver",
    "DEMO_FACTS"
]
