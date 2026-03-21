"""
resolver.py - The Field Collapse Engine

This is the heart of imagineAI. It takes a question (excitation) and
resolves the semantic field to find the answer (stable configuration).

The resolution loop:
    1. Question excites the Φ field
    2. LLM generates candidate response (field generator)
    3. Apply ITT operators to evaluate candidate
    4. Check σ (residue) and ρ_q (boundaries)
    5. If σ ≈ 0 and boundaries satisfied → answer found
    6. Otherwise → iterate with feedback

From ITT: Resolution occurs when the field reaches a stable configuration
where σ → 0 and all boundary conditions are satisfied.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .phi_field import PhiField, PhiState
from .operators import SemanticOperators, compute_sigma
from .sigma import SigmaCalculator, SigmaResult
from .rho_q import BoundaryConditions, extract_constraints_from_question


class ResolutionStatus(Enum):
    """Status of the resolution process"""
    PENDING = "pending"
    RESOLVED = "resolved"
    MAX_ITERATIONS = "max_iterations"
    NO_CANDIDATES = "no_candidates"
    FAILED = "failed"


@dataclass
class ResolutionResult:
    """
    Result of the resolution process.
    """
    status: ResolutionStatus
    answer: Optional[str]
    answer_state: Optional[PhiState]
    sigma: float
    iterations: int
    history: List[Dict[str, Any]]
    
    @property
    def success(self) -> bool:
        return self.status == ResolutionStatus.RESOLVED
    
    def __str__(self):
        if self.success:
            return f"✓ RESOLVED in {self.iterations} iterations (σ={self.sigma:.4f})\nAnswer: {self.answer}"
        else:
            return f"✗ {self.status.value} after {self.iterations} iterations (σ={self.sigma:.4f})"


class ImagineAIResolver:
    """
    The main resolution engine for imagineAI.
    
    Takes a question and resolves it to an answer through field collapse.
    """
    
    def __init__(
        self,
        phi_field: PhiField,
        response_generator: Optional[Callable] = None,
        knowledge_retriever: Optional[Callable] = None,
        max_iterations: int = 10,
        sigma_threshold: float = 0.1,
        verbose: bool = False
    ):
        """
        Args:
            phi_field: The embedding substrate
            response_generator: Function that generates candidate responses
                               (question: str, context: List[str]) -> str
            knowledge_retriever: Function that retrieves relevant knowledge
                                (query: str) -> List[str]
            max_iterations: Maximum resolution iterations
            sigma_threshold: σ below this = resolved
            verbose: Print debug info
        """
        self.phi_field = phi_field
        self.operators = SemanticOperators(phi_field)
        self.sigma_calc = SigmaCalculator(threshold=sigma_threshold)
        
        self.response_generator = response_generator
        self.knowledge_retriever = knowledge_retriever
        
        self.max_iterations = max_iterations
        self.sigma_threshold = sigma_threshold
        self.verbose = verbose
        
        # Conversation history
        self.conversation: List[PhiState] = []
        
    def set_response_generator(self, generator: Callable):
        """Set the response generator (LLM or other)"""
        self.response_generator = generator
        
    def set_knowledge_retriever(self, retriever: Callable):
        """Set the knowledge retrieval function"""
        self.knowledge_retriever = retriever
        
    def _log(self, msg: str):
        """Log message if verbose"""
        if self.verbose:
            print(f"[RESOLVER] {msg}")
    
    def resolve(
        self,
        question: str,
        additional_context: Optional[List[str]] = None
    ) -> ResolutionResult:
        """
        Resolve a question to an answer.
        
        This is the main entry point. It:
        1. Embeds the question
        2. Extracts boundary conditions
        3. Retrieves relevant knowledge
        4. Generates candidate response
        5. Checks resolution (σ and ρ_q)
        6. Iterates if needed
        
        Args:
            question: The question to resolve
            additional_context: Extra context to consider
            
        Returns:
            ResolutionResult with answer and metadata
        """
        history = []
        
        # 1. Embed the question (excitation)
        self._log(f"Question: {question}")
        question_state = self.phi_field.embed(question)
        self.conversation.append(question_state)
        
        # 2. Extract boundary conditions from question
        boundaries = extract_constraints_from_question(question, self.phi_field)
        self._log(f"Extracted {len(boundaries.constraints)} constraints")
        
        # 3. Retrieve relevant knowledge
        knowledge_context = []
        if self.knowledge_retriever:
            try:
                knowledge = self.knowledge_retriever(question)
                if knowledge:
                    knowledge_context = knowledge if isinstance(knowledge, list) else [knowledge]
                    self._log(f"Retrieved {len(knowledge_context)} knowledge items")
            except Exception as e:
                self._log(f"Knowledge retrieval failed: {e}")
        
        # Combine all context
        context = []
        if additional_context:
            context.extend(additional_context)
        context.extend(knowledge_context)
        
        # Add conversation history
        conversation_context = [s.text for s in self.conversation[-5:]]  # Last 5 turns
        context.extend(conversation_context)
        
        # 4. Resolution loop
        current_sigma = float('inf')
        best_answer = None
        best_state = None
        best_sigma = float('inf')
        
        for iteration in range(self.max_iterations):
            self._log(f"\n--- Iteration {iteration + 1} ---")
            
            # Generate candidate response
            if self.response_generator is None:
                self._log("No response generator - cannot continue")
                return ResolutionResult(
                    status=ResolutionStatus.FAILED,
                    answer=None,
                    answer_state=None,
                    sigma=float('inf'),
                    iterations=iteration,
                    history=history
                )
            
            try:
                # Build context string
                context_str = "\n".join(context) if context else ""
                
                # Generate response
                candidate_text = self.response_generator(question, context_str)
                self._log(f"Candidate: {candidate_text[:100]}...")
                
                # Embed the candidate
                candidate_state = self.phi_field.embed(candidate_text)
                
            except Exception as e:
                self._log(f"Generation failed: {e}")
                history.append({"iteration": iteration, "error": str(e)})
                continue
            
            # 5. Check resolution
            
            # Compute σ (semantic distance from question to answer should be appropriate)
            # For Q&A, we want some distance (it's an answer, not echo) but coherence
            sigma_result = self.sigma_calc.compute(
                current=candidate_state,
                context=[question_state] + [self.phi_field.embed(c) for c in context[:3]]
            )
            current_sigma = sigma_result.total
            self._log(f"σ = {current_sigma:.4f}")
            
            # Check boundary conditions
            boundary_result = boundaries.check(candidate_state)
            self._log(f"Boundaries: {'✓' if boundary_result.satisfied else '✗'} (penalty: {boundary_result.total_penalty:.3f})")
            
            # Combined score
            total_score = current_sigma + boundary_result.total_penalty
            
            # Track best so far
            if total_score < best_sigma:
                best_sigma = total_score
                best_answer = candidate_text
                best_state = candidate_state
            
            # Record iteration
            history.append({
                "iteration": iteration,
                "candidate": candidate_text,
                "sigma": current_sigma,
                "boundary_penalty": boundary_result.total_penalty,
                "total_score": total_score,
                "violations": boundary_result.violations
            })
            
            # Check if resolved
            if sigma_result.is_resolved and boundary_result.satisfied:
                self._log("✓ RESOLVED")
                
                # Add answer to conversation
                self.conversation.append(candidate_state)
                
                return ResolutionResult(
                    status=ResolutionStatus.RESOLVED,
                    answer=candidate_text,
                    answer_state=candidate_state,
                    sigma=current_sigma,
                    iterations=iteration + 1,
                    history=history
                )
            
            # 6. If not resolved, add feedback for next iteration
            if boundary_result.violations:
                # Add constraint violations as context
                context.append(f"Previous attempt violated: {', '.join(boundary_result.violations)}")
        
        # Max iterations reached
        self._log(f"Max iterations reached. Best σ = {best_sigma:.4f}")
        
        # Return best answer even if not fully resolved
        if best_answer:
            self.conversation.append(best_state)
            
        return ResolutionResult(
            status=ResolutionStatus.MAX_ITERATIONS,
            answer=best_answer,
            answer_state=best_state,
            sigma=best_sigma,
            iterations=self.max_iterations,
            history=history
        )
    
    def chat(self, message: str) -> str:
        """
        Simple chat interface - resolve and return just the answer.
        
        Args:
            message: User message
            
        Returns:
            Response string
        """
        result = self.resolve(message)
        if result.success:
            return result.answer
        elif result.answer:
            return f"(uncertain) {result.answer}"
        else:
            return "I could not resolve an answer."
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation = []
    
    def get_conversation_curl(self) -> float:
        """
        Compute the curl (∇×F) of the conversation.
        
        High curl = conversation loops back on itself (coherent themes)
        Low curl = linear conversation with no backtracking
        """
        if len(self.conversation) < 3:
            return 0.0
        
        curl = self.operators.curl(self.conversation)
        return curl.strength


# =============================================================================
# Simple Generator Implementations (for testing without LLM)
# =============================================================================

def create_simple_generator(knowledge_base: Dict[str, str]) -> Callable:
    """
    Create a simple lookup-based response generator.
    
    Args:
        knowledge_base: Dict mapping keywords to answers
        
    Returns:
        Generator function
    """
    def generator(question: str, context: str) -> str:
        question_lower = question.lower()
        
        # Look for keywords in knowledge base
        for keyword, answer in knowledge_base.items():
            if keyword.lower() in question_lower:
                return answer
        
        return "I don't have information about that."
    
    return generator


def create_echo_generator() -> Callable:
    """Create a generator that just echoes (for testing)"""
    def generator(question: str, context: str) -> str:
        return f"You asked: {question}"
    return generator


# =============================================================================
# Quick Demo Function
# =============================================================================

def demo_resolver():
    """
    Quick demo of the resolver with simple components.
    """
    print("=" * 60)
    print("imagineAI Resolver Demo")
    print("=" * 60)
    
    # Create simple word embedding field
    from .phi_field import create_simple_field
    
    try:
        field = create_simple_field()
    except ImportError:
        print("Run: pip install gensim")
        return
    
    # Simple knowledge base
    knowledge = {
        "capital of mississippi": "Jackson",
        "capital of texas": "Austin",
        "capital of california": "Sacramento",
        "largest planet": "Jupiter",
        "smallest planet": "Mercury"
    }
    
    # Create resolver
    resolver = ImagineAIResolver(
        phi_field=field,
        response_generator=create_simple_generator(knowledge),
        max_iterations=3,
        sigma_threshold=0.3,
        verbose=True
    )
    
    # Test questions
    questions = [
        "What is the capital of Mississippi?",
        "What is the capital of Texas?",
        "What is the largest planet?"
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        result = resolver.resolve(q)
        print(result)
        print()


if __name__ == "__main__":
    demo_resolver()
