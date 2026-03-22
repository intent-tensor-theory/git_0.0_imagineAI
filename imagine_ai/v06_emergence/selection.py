"""
selection.py - The Selection Number

S = R / (Ṙ · t_ref)

Where:
- R = retained structure (semantic coherence with constraints)
- Ṙ = loss rate (how fast coherence is degrading)  
- t_ref = reference timescale (question complexity)

S > 1: Supercritical → Answer persists
S < 1: Subcritical → Dissolves
S = 1: Critical → Marginal

From "The Matter of Emergence":
> Structure emerges when retention mechanisms dominate loss mechanisms
> over the timescale that matters.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SelectionResult:
    """Result of selection calculation."""
    S: float                    # Selection number
    R: float                    # Retained structure
    R_dot: float               # Loss rate
    t_ref: float               # Reference timescale
    regime: str                # "supercritical", "critical", "subcritical"
    
    @property
    def persists(self) -> bool:
        """Does this configuration survive?"""
        return self.S > 1.0


def compute_retained_structure(
    Φ: np.ndarray,
    anchors: List[np.ndarray],
    glove,
    anchor_words: List[str]
) -> float:
    """
    Compute R - the retained structure.
    
    R measures how much "organized structure" the field configuration
    has relative to the constraints (anchors).
    
    Higher R = more coherence with what the question requires.
    """
    if not anchors or len(anchors) == 0:
        return 1.0  # No constraints = everything retained
    
    # R is the alignment between Φ and anchor constraints
    alignments = []
    
    for anchor in anchors:
        if np.linalg.norm(anchor) > 0 and np.linalg.norm(Φ) > 0:
            # Cosine similarity
            cos_sim = np.dot(Φ, anchor) / (np.linalg.norm(Φ) * np.linalg.norm(anchor))
            # Convert to [0, 1] range
            alignment = (cos_sim + 1) / 2
            alignments.append(alignment)
    
    if not alignments:
        return 0.5  # Neutral
    
    # R is the geometric mean of alignments (all must be satisfied)
    R = np.exp(np.mean(np.log(np.array(alignments) + 1e-8)))
    
    return float(R)


def compute_loss_rate(
    Φ_current: np.ndarray,
    Φ_previous: np.ndarray,
    anchors: List[np.ndarray],
    dt: float = 1.0
) -> float:
    """
    Compute Ṙ - the loss rate.
    
    How fast is the field losing coherence with constraints?
    
    Positive Ṙ = losing structure
    Negative Ṙ = gaining structure (this decreases S, which is good!)
    """
    # Compute R at both timesteps
    # Simplified: use magnitude of change as proxy for loss
    
    dΦ = Φ_current - Φ_previous
    
    # Loss rate is magnitude of change per unit time
    # This is a simplification - full calculation requires R at both times
    R_dot = np.linalg.norm(dΦ) / dt
    
    # Normalize to [0, 1] range
    R_dot = R_dot / (np.linalg.norm(Φ_current) + 1e-8)
    
    return float(R_dot)


def compute_reference_timescale(
    question: str,
    anchor_count: int,
    complexity_factor: float = 1.0
) -> float:
    """
    Compute t_ref - the reference timescale.
    
    This is the externally imposed timescale of the "question process."
    More complex questions have longer reference times.
    
    From ICHTB: t_ref is the time available for the system to
    complete its organizational task before the collapse window closes.
    """
    # Base timescale
    t_base = 1.0
    
    # Scale by question complexity
    word_count = len(question.split())
    t_ref = t_base * (1 + 0.1 * word_count) * (1 + 0.2 * anchor_count) * complexity_factor
    
    return max(t_ref, 0.1)  # Minimum timescale


def compute_selection_number(
    Φ: np.ndarray,
    Φ_previous: np.ndarray,
    anchors: List[np.ndarray],
    anchor_words: List[str],
    glove,
    question: str,
    dt: float = 1.0
) -> SelectionResult:
    """
    Compute S = R / (Ṙ · t_ref)
    
    The central quantity determining emergence.
    
    S > 1: The configuration persists - it's an answer
    S < 1: The configuration dissolves - not an answer
    S = 1: Critical boundary
    """
    # Compute components
    R = compute_retained_structure(Φ, anchors, glove, anchor_words)
    R_dot = compute_loss_rate(Φ, Φ_previous, anchors, dt)
    t_ref = compute_reference_timescale(question, len(anchors))
    
    # Selection number
    if R_dot * t_ref > 1e-8:
        S = R / (R_dot * t_ref)
    else:
        # Zero loss rate = infinite persistence
        S = float('inf') if R > 0 else 0.0
    
    # Determine regime
    if S > 1.0:
        regime = "supercritical"
    elif S < 1.0:
        regime = "subcritical"
    else:
        regime = "critical"
    
    return SelectionResult(
        S=S,
        R=R,
        R_dot=R_dot,
        t_ref=t_ref,
        regime=regime
    )


# =============================================================================
# Selection Landscape
# =============================================================================

@dataclass
class SelectionLandscape:
    """
    The S-landscape over configuration space.
    
    - Valley (S < 1): Subcritical, funnels to vacuum
    - Ridge (S = 1): Critical boundary
    - Plateau (S > 1): Supercritical, funnels to lock
    """
    configurations: List[np.ndarray]
    selection_numbers: List[float]
    
    @property
    def supercritical_configs(self) -> List[Tuple[int, np.ndarray]]:
        """Configurations that persist."""
        return [
            (i, cfg) for i, (cfg, s) in enumerate(
                zip(self.configurations, self.selection_numbers)
            ) if s > 1.0
        ]
    
    @property
    def best_configuration(self) -> Optional[np.ndarray]:
        """Configuration with highest S."""
        if not self.selection_numbers:
            return None
        
        max_idx = np.argmax(self.selection_numbers)
        return self.configurations[max_idx]
    
    @property
    def max_S(self) -> float:
        """Maximum selection number."""
        return max(self.selection_numbers) if self.selection_numbers else 0.0


def build_selection_landscape(
    field_configs: List[np.ndarray],
    Φ_previous: np.ndarray,
    anchors: List[np.ndarray],
    anchor_words: List[str],
    glove,
    question: str
) -> SelectionLandscape:
    """
    Build the selection landscape over a set of configurations.
    
    This shows which configurations persist (S > 1) and which dissolve.
    """
    selection_numbers = []
    
    for cfg in field_configs:
        result = compute_selection_number(
            Φ=cfg,
            Φ_previous=Φ_previous,
            anchors=anchors,
            anchor_words=anchor_words,
            glove=glove,
            question=question
        )
        selection_numbers.append(result.S)
    
    return SelectionLandscape(
        configurations=field_configs,
        selection_numbers=selection_numbers
    )
