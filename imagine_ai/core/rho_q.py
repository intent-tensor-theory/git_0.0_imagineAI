"""
rho_q.py - Boundary Conditions

ρ_q (rho_q) represents boundary charge - the frozen values at termination
surfaces. In imagineAI, these are the constraints that valid answers must satisfy.

From ITT:
    ρ_q = boundary charge at termination surfaces
    Value freezes at boundaries where ∇Φ = 0
    
In language:
    ρ_q = constraints like "must be a city", "must be factually correct"
    Answer is valid only if all ρ_q conditions are satisfied
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .phi_field import PhiState


class ConstraintType(Enum):
    """Types of boundary constraints"""
    SEMANTIC = "semantic"       # Must be similar to certain concepts
    FACTUAL = "factual"         # Must match known facts
    LOGICAL = "logical"         # Must be internally consistent
    FORMAT = "format"           # Must have certain structure
    EXCLUSION = "exclusion"     # Must NOT be certain things


@dataclass
class Constraint:
    """
    A single boundary constraint (ρ_q element).
    """
    name: str
    constraint_type: ConstraintType
    checker: Callable[[PhiState], float]  # Returns penalty (0 = satisfied)
    description: str = ""
    weight: float = 1.0
    
    def check(self, state: PhiState) -> float:
        """Check constraint and return weighted penalty"""
        return self.weight * self.checker(state)


@dataclass
class BoundaryResult:
    """
    Result of checking all boundary conditions.
    """
    satisfied: bool
    total_penalty: float
    violations: List[str]
    details: Dict[str, float]


class BoundaryConditions:
    """
    Manager for ρ_q boundary conditions.
    
    In ITT terms, boundaries are where the field cannot change - 
    the "walls" that constrain valid solutions.
    """
    
    def __init__(self, phi_field=None):
        """
        Args:
            phi_field: Optional PhiField for semantic constraint checking
        """
        self.phi_field = phi_field
        self.constraints: List[Constraint] = []
        
    def add_semantic_constraint(
        self,
        name: str,
        target_concepts: List[str],
        min_similarity: float = 0.3,
        weight: float = 1.0
    ):
        """
        Add constraint that answer must be semantically similar to concepts.
        
        Args:
            name: Name of this constraint
            target_concepts: Words/phrases the answer should be similar to
            min_similarity: Minimum cosine similarity required
            weight: How much to penalize violations
        """
        if self.phi_field is None:
            raise ValueError("PhiField required for semantic constraints")
        
        # Embed target concepts
        target_states = [self.phi_field.embed(c) for c in target_concepts]
        
        def checker(state: PhiState) -> float:
            # Check similarity to each target concept
            similarities = [state.similarity_to(t) for t in target_states]
            max_sim = max(similarities)
            
            if max_sim >= min_similarity:
                return 0.0  # Constraint satisfied
            else:
                # Penalty proportional to how far below threshold
                return (min_similarity - max_sim)
        
        self.constraints.append(Constraint(
            name=name,
            constraint_type=ConstraintType.SEMANTIC,
            checker=checker,
            description=f"Must be similar to: {target_concepts}",
            weight=weight
        ))
        
    def add_exclusion_constraint(
        self,
        name: str,
        excluded_concepts: List[str],
        max_similarity: float = 0.7,
        weight: float = 1.0
    ):
        """
        Add constraint that answer must NOT be similar to certain concepts.
        
        Args:
            name: Name of this constraint
            excluded_concepts: Words/phrases to avoid
            max_similarity: Maximum allowed similarity
            weight: How much to penalize violations
        """
        if self.phi_field is None:
            raise ValueError("PhiField required for exclusion constraints")
        
        # Embed excluded concepts
        excluded_states = [self.phi_field.embed(c) for c in excluded_concepts]
        
        def checker(state: PhiState) -> float:
            similarities = [state.similarity_to(e) for e in excluded_states]
            max_sim = max(similarities)
            
            if max_sim <= max_similarity:
                return 0.0
            else:
                return (max_sim - max_similarity)
        
        self.constraints.append(Constraint(
            name=name,
            constraint_type=ConstraintType.EXCLUSION,
            checker=checker,
            description=f"Must NOT be similar to: {excluded_concepts}",
            weight=weight
        ))
        
    def add_custom_constraint(
        self,
        name: str,
        checker: Callable[[PhiState], float],
        constraint_type: ConstraintType = ConstraintType.LOGICAL,
        description: str = "",
        weight: float = 1.0
    ):
        """
        Add a custom constraint with arbitrary logic.
        
        Args:
            name: Name of this constraint
            checker: Function returning penalty (0 = satisfied)
            constraint_type: Type of constraint
            description: Human-readable description
            weight: Penalty weight
        """
        self.constraints.append(Constraint(
            name=name,
            constraint_type=constraint_type,
            checker=checker,
            description=description,
            weight=weight
        ))
        
    def add_length_constraint(
        self,
        min_length: int = 1,
        max_length: int = 1000,
        weight: float = 0.5
    ):
        """
        Add constraint on response length (in characters).
        """
        def checker(state: PhiState) -> float:
            length = len(state.text)
            if length < min_length:
                return (min_length - length) / min_length
            elif length > max_length:
                return (length - max_length) / max_length
            return 0.0
        
        self.add_custom_constraint(
            name="length",
            checker=checker,
            constraint_type=ConstraintType.FORMAT,
            description=f"Length must be between {min_length} and {max_length}",
            weight=weight
        )
        
    def check(self, state: PhiState) -> BoundaryResult:
        """
        Check all boundary conditions for a state.
        
        Args:
            state: The semantic state to check
            
        Returns:
            BoundaryResult with satisfaction status and penalties
        """
        violations = []
        details = {}
        total_penalty = 0.0
        
        for constraint in self.constraints:
            try:
                penalty = constraint.check(state)
                details[constraint.name] = penalty
                total_penalty += penalty
                
                if penalty > 0:
                    violations.append(f"{constraint.name}: {penalty:.3f}")
            except Exception as e:
                # Constraint check failed
                penalty = 0.5
                details[constraint.name] = penalty
                total_penalty += penalty
                violations.append(f"{constraint.name}: ERROR ({e})")
        
        return BoundaryResult(
            satisfied=(len(violations) == 0),
            total_penalty=total_penalty,
            violations=violations,
            details=details
        )
    
    def is_valid(self, state: PhiState) -> bool:
        """Quick check if state satisfies all constraints"""
        return self.check(state).satisfied
    
    def filter_valid(self, candidates: List[PhiState]) -> List[PhiState]:
        """Filter list to only valid candidates"""
        return [c for c in candidates if self.is_valid(c)]
    
    def rank_by_validity(self, candidates: List[PhiState]) -> List[tuple[PhiState, float]]:
        """
        Rank candidates by how well they satisfy constraints.
        
        Returns:
            List of (state, penalty) tuples, sorted by penalty (lowest first)
        """
        results = [(c, self.check(c).total_penalty) for c in candidates]
        return sorted(results, key=lambda x: x[1])


def extract_constraints_from_question(
    question: str,
    phi_field
) -> BoundaryConditions:
    """
    Extract implicit boundary conditions from a question.
    
    For example, "What is the capital of Mississippi?" implies:
    - Answer should be similar to "city", "capital"
    - Answer should be related to "Mississippi"
    
    This is a basic implementation - can be enhanced with NLP.
    
    Args:
        question: The question text
        phi_field: PhiField for embedding
        
    Returns:
        BoundaryConditions extracted from question
    """
    boundaries = BoundaryConditions(phi_field)
    
    # Simple keyword extraction
    question_lower = question.lower()
    
    # Question type constraints
    if "what is" in question_lower or "what's" in question_lower:
        # Factual question - answer should be noun-like
        boundaries.add_length_constraint(min_length=1, max_length=200)
    
    if "capital" in question_lower:
        boundaries.add_semantic_constraint(
            name="is_capital",
            target_concepts=["capital", "city", "seat of government"],
            min_similarity=0.2
        )
    
    if "when" in question_lower:
        boundaries.add_semantic_constraint(
            name="is_time",
            target_concepts=["year", "date", "time", "century"],
            min_similarity=0.2
        )
    
    if "who" in question_lower:
        boundaries.add_semantic_constraint(
            name="is_person",
            target_concepts=["person", "name", "individual"],
            min_similarity=0.2
        )
    
    if "where" in question_lower:
        boundaries.add_semantic_constraint(
            name="is_place",
            target_concepts=["place", "location", "city", "country"],
            min_similarity=0.2
        )
    
    # Extract potential topic words (simple heuristic)
    # Words after "of", "in", "about" are often topics
    words = question.split()
    for i, word in enumerate(words):
        if word.lower() in ["of", "in", "about"] and i + 1 < len(words):
            topic = words[i + 1].strip("?.,!")
            if len(topic) > 2:
                boundaries.add_semantic_constraint(
                    name=f"about_{topic}",
                    target_concepts=[topic],
                    min_similarity=0.15,
                    weight=0.5
                )
    
    return boundaries
