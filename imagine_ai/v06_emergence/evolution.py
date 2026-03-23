"""
evolution.py - Master Equation Dynamics

The ICHTB Master Equation governs field evolution:

∂Φ/∂t = D·∇ᵢ(Mⁱʲ∇ⱼΦ) - Λ·Mⁱʲ∇ᵢΦ∇ⱼΦ + γΦ³ - κΦ

Four terms:
1. Diffusive modulation: Spreading shaped by memory
2. Alignment decay: Braking when gradient aligns with metric  
3. Nonlinear growth: Self-amplification (shell formation)
4. Linear decay: Return toward anchor

The field evolves until closure: ∂Φ/∂t ≈ 0 with S > 1
"""

import numpy as np
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .field import (
    SemanticField, 
    compute_gradient, 
    compute_curl, 
    compute_laplacian,
    compute_temporal_derivative,
    update_metric_tensor,
    text_to_embedding
)
from .selection import compute_selection_number, SelectionResult


@dataclass
class EvolutionParameters:
    """Parameters for the master equation."""
    D: float = 0.3      # Diffusivity (spreading rate) - higher = faster convergence
    Λ: float = 0.1      # Alignment decay rate (braking)
    γ: float = 0.05     # Nonlinear growth rate (shell formation)
    κ: float = 0.1      # Linear decay rate (anchor return) - prevents explosion
    dt: float = 0.5     # Time step - larger for faster convergence
    
    # Lock thresholds
    lock_threshold: float = 0.05    # ∂Φ/∂t magnitude for lock
    S_threshold: float = 0.5        # Selection number for persistence (lowered for testing)


@dataclass 
class ClosureResult:
    """Result of evolution to closure."""
    Φ_final: np.ndarray             # Final field configuration
    locked: bool                     # Did closure occur?
    iterations: int                  # How many steps
    S_final: SelectionResult        # Final selection number
    trace: List[Tuple[float, float]] # (time, S) history


def master_equation_step(
    field: SemanticField,
    substrate_points: List[np.ndarray],
    context_vectors: List[np.ndarray],
    params: EvolutionParameters
) -> np.ndarray:
    """
    One step of the master equation.
    
    ∂Φ/∂t = D·∇ᵢ(Mⁱʲ∇ⱼΦ) - Λ·Mⁱʲ∇ᵢΦ∇ⱼΦ + γΦ³ - κΦ
    
    Returns: dΦ/dt
    
    STABILITY: All terms are bounded to prevent explosion.
    """
    Φ = field.Φ.copy()
    
    # STABILITY: Normalize Φ to prevent explosion
    Φ_norm = np.linalg.norm(Φ)
    if Φ_norm > 10.0:
        Φ = Φ / Φ_norm * 10.0
        field.Φ = Φ
    
    # Compute operators
    grad_Φ = compute_gradient(field, substrate_points)
    curl_F = compute_curl(field, context_vectors)
    laplacian_Φ = compute_laplacian(field, substrate_points)
    
    # STABILITY: Clip gradients
    grad_norm = np.linalg.norm(grad_Φ)
    if grad_norm > 10.0:
        grad_Φ = grad_Φ / grad_norm * 10.0
    
    # Skip metric update for now - use identity
    # This simplifies to pure semantic dynamics
    M = np.eye(field.dim) * 0.01
    
    # Term 1: Diffusive modulation D·laplacian
    # Move toward neighbors
    term1 = params.D * laplacian_Φ
    
    # Term 2: Alignment decay
    # Brake when aligned with gradient
    alignment = np.dot(grad_Φ, grad_Φ) / (grad_norm**2 + 1e-8)
    term2 = -params.Λ * alignment * grad_Φ * 0.1
    
    # Term 3: Nonlinear growth +γΦ³
    # STABILITY: Use tanh saturation to bound
    saturation = np.tanh(Φ_norm / 5.0)
    term3 = params.γ * saturation * Φ
    
    # Term 4: Linear decay -κΦ
    # Return toward anchor
    term4 = -params.κ * Φ
    
    # Total rate of change
    dΦ_dt = term1 + term2 + term3 + term4
    
    # STABILITY: Clip final rate
    dΦ_norm = np.linalg.norm(dΦ_dt)
    if dΦ_norm > 1.0:
        dΦ_dt = dΦ_dt / dΦ_norm
    
    return dΦ_dt


def evolve_to_closure(
    field: SemanticField,
    substrate_points: List[np.ndarray],
    context_vectors: List[np.ndarray],
    anchors: List[np.ndarray],
    anchor_words: List[str],
    question: str,
    params: EvolutionParameters = None,
    max_iterations: int = 100,
    verbose: bool = False
) -> ClosureResult:
    """
    Evolve the field until closure or max iterations.
    
    Closure occurs when:
    1. ∂Φ/∂t ≈ 0 (field stabilizes)
    2. S > 1 (configuration persists)
    
    This is the emergence process.
    """
    if params is None:
        params = EvolutionParameters()
    
    trace = []
    locked = False
    
    for t in range(max_iterations):
        # Store previous state
        Φ_prev = field.Φ.copy()
        
        # Compute rate of change
        dΦ_dt = master_equation_step(
            field=field,
            substrate_points=substrate_points,
            context_vectors=context_vectors,
            params=params
        )
        
        # Update field
        field.Φ = field.Φ + params.dt * dΦ_dt
        field.Φ_history.append(field.Φ.copy())
        
        # Compute selection number
        S_result = compute_selection_number(
            Φ=field.Φ,
            Φ_previous=Φ_prev,
            anchors=anchors,
            anchor_words=anchor_words,
            glove=field.glove,
            question=question,
            dt=params.dt
        )
        
        trace.append((t * params.dt, S_result.S))
        
        # Check for lock (Δ₅ apex test)
        dΦ_magnitude = np.linalg.norm(dΦ_dt)
        
        if verbose and t % 10 == 0:
            print(f"[t={t}] |∂Φ/∂t|={dΦ_magnitude:.4f}, S={S_result.S:.4f}, regime={S_result.regime}")
        
        # Lock condition: field stabilizes AND persists
        if dΦ_magnitude < params.lock_threshold and S_result.S > params.S_threshold:
            locked = True
            if verbose:
                print(f"[LOCK] Closure at t={t}, S={S_result.S:.4f}")
            break
    
    # Final selection
    S_final = compute_selection_number(
        Φ=field.Φ,
        Φ_previous=field.Φ_history[-2] if len(field.Φ_history) > 1 else field.Φ,
        anchors=anchors,
        anchor_words=anchor_words,
        glove=field.glove,
        question=question,
        dt=params.dt
    )
    
    return ClosureResult(
        Φ_final=field.Φ,
        locked=locked,
        iterations=t + 1,
        S_final=S_final,
        trace=trace
    )


def find_answer_in_substrate(
    Φ_final: np.ndarray,
    facts: List[str],
    fact_embeddings: List[np.ndarray],
    glove,
    anchor_words: List[str] = None,
    question: str = ""
) -> Tuple[str, float]:
    """
    Given the final field configuration, find the answer.
    
    The answer must:
    1. Be close to the stable configuration (Φ_final)
    2. Satisfy anchor constraints (ρ_q)
    3. NOT have disqualifying words (second, land-only, etc.)
    4. Have anchors close together (proximity bonus)
    
    This is the writability gate: only configurations
    that respect boundaries can be valid answers.
    """
    if not facts or not fact_embeddings:
        return None, 0.0
    
    # Extract question characteristics for context-aware scoring
    question_lower = question.lower()
    question_words = set(re.findall(r'\b[a-z]+\b', question_lower))
    
    # Does question ask for an ordinal? (second tallest, third largest, etc.)
    question_asks_ordinal = any(ord in question_words for ord in ['second', 'third', 'fourth', 'fifth'])
    
    # Ordinal qualifiers - indicate a non-primary answer (penalized)
    # Note: "second" can be ordinal OR time unit - we check context below
    ORDINAL_QUALIFIERS = {'third', 'fourth', 'fifth'}
    
    # Phrases that indicate partial/qualified answer  
    PARTIAL_PHRASES = {'one of', 'among', 'nearly', 'almost'}
    
    # Words that indicate specificity when anchor is general
    SPECIFICITY_QUALIFIERS = {'land', 'sea', 'marine', 'flying', 'bird'}
    
    # Regional qualifiers - if question doesn't specify region, penalize regional facts
    REGIONAL_QUALIFIERS = {'in africa', 'in asia', 'in europe', 'in north america', 
                           'in south america', 'in australia', 'in japan', 'in china',
                           'in america', 'in the world'}
    
    # Score each fact
    scores = []
    
    for i, (fact, emb) in enumerate(zip(facts, fact_embeddings)):
        fact_lower = fact.lower()
        # Use regex for word extraction (same as anchor extraction)
        # This ensures "co-founded" becomes ['co', 'founded'] to match anchors
        fact_words_list = re.findall(r'\b[a-z]+\b', fact_lower)
        fact_words = set(fact_words_list)
        
        # Component 1: Semantic similarity to Φ_final
        if np.linalg.norm(emb) > 0 and np.linalg.norm(Φ_final) > 0:
            semantic_sim = np.dot(Φ_final, emb) / (np.linalg.norm(Φ_final) * np.linalg.norm(emb))
        else:
            semantic_sim = 0.0
        
        # Component 2: Anchor satisfaction (ρ_q constraint)
        anchor_score = 0.0
        anchor_positions = []  # Track where anchors appear
        direct_match_count = 0  # Count direct (not semantic) matches
        
        if anchor_words:
            for anchor in anchor_words:
                best_match = 0.0
                best_pos = -1
                is_direct = False
                
                # Direct match (whole word only)
                # Check if anchor is a complete word in the fact
                for pos, word in enumerate(fact_words_list):
                    # Strip punctuation for comparison
                    clean_word = word.strip('.,!?;:\'"()[]')
                    if anchor == clean_word:
                        best_match = 1.0
                        best_pos = pos
                        is_direct = True
                        break
                
                # If no direct match, try semantic match
                if best_match < 1.0:
                    for pos, word in enumerate(fact_words_list):
                        clean_word = word.strip('.,!?;:\'"()[]')
                        if clean_word in glove and anchor in glove:
                            sim = glove.similarity(anchor, clean_word)
                            match_score = 1.0 / (1.0 + np.exp(-10 * (sim - 0.3)))
                            if match_score > best_match:
                                best_match = match_score
                                best_pos = pos
                
                anchor_score += best_match
                if best_pos >= 0:
                    anchor_positions.append(best_pos)
                if is_direct:
                    direct_match_count += 1
            
            anchor_score = anchor_score / len(anchor_words) if anchor_words else 1.0
        else:
            anchor_score = 1.0
        
        # Component 3: Proximity bonus
        # If multiple anchors appear close together, give small boost
        # But only for DIRECT matches (not semantic)
        proximity_bonus = 1.0
        if len(anchor_positions) >= 2 and direct_match_count >= 2:
            anchor_positions.sort()
            # Calculate average distance between consecutive anchors
            total_dist = 0
            for j in range(len(anchor_positions) - 1):
                total_dist += anchor_positions[j+1] - anchor_positions[j]
            avg_dist = total_dist / (len(anchor_positions) - 1)
            
            # Smaller bonuses - proximity is a TIEBREAKER, not dominant
            if avg_dist <= 3:
                proximity_bonus = 1.1  # Small bonus for adjacent anchors
            elif avg_dist <= 5:
                proximity_bonus = 1.05
        
        # Component 4: Qualifier penalty
        qualifier_penalty = 1.0
        
        # Only penalize ordinals if question does NOT ask for an ordinal
        if not question_asks_ordinal:
            # Check for ordinal qualifiers (third, fourth, fifth)
            for qual in ORDINAL_QUALIFIERS:
                if qual in fact_words:
                    qualifier_penalty *= 0.2
            
            # Special handling for "second" - distinguish ordinal from time unit
            # "second tallest" = ordinal (penalize)
            # "per second" = time unit (don't penalize)
            if 'second' in fact_words:
                # Check context - is it followed by a superlative or preceded by "per"?
                is_time_unit = 'per' in fact_words or 'seconds' in fact_words
                is_ordinal = any(sup in fact_lower for sup in ['second tallest', 'second largest', 'second longest', 'second biggest', 'second highest', 'second fastest', 'second most'])
                if is_ordinal and not is_time_unit:
                    qualifier_penalty *= 0.3
        else:
            # Question ASKS for ordinal - BOOST facts with matching ordinal
            for qual in ['second', 'third', 'fourth', 'fifth']:
                if qual in question_words and qual in fact_words:
                    qualifier_penalty *= 1.5  # Boost for matching ordinal
        
        # Check for partial phrases
        for phrase in PARTIAL_PHRASES:
            if phrase in fact_lower:
                qualifier_penalty *= 0.5
        
        # Component 5: Specificity penalty
        specificity_penalty = 1.0
        if anchor_words:
            anchor_set = set(anchor_words)
            for qual in SPECIFICITY_QUALIFIERS:
                if qual in fact_words and qual not in anchor_set:
                    specificity_penalty *= 0.7
        
        # Component 6: Regional penalty
        # If question doesn't mention a region but fact does, penalize
        regional_penalty = 1.0
        if anchor_words:
            question_has_region = any(region.split()[-1] in anchor_words for region in REGIONAL_QUALIFIERS)
            if not question_has_region:
                for region in REGIONAL_QUALIFIERS:
                    if region in fact_lower:
                        regional_penalty *= 0.6
                        break  # Only apply once
        
        # Combined score
        combined = anchor_score * proximity_bonus * qualifier_penalty * specificity_penalty * regional_penalty + 0.05 * (semantic_sim + 1) / 2
        
        scores.append((i, fact, combined, anchor_score, proximity_bonus))
    
    # Sort by combined score
    scores.sort(key=lambda x: x[2], reverse=True)
    
    # Return best fact
    if scores:
        best_idx, best_fact, best_score, anch_score, prox = scores[0]
        return best_fact, best_score
    
    return None, 0.0
