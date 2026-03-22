"""
solver.py - Wikipedia-Scale Emergence Solver

Scales v0.6 emergence to millions of facts via:
1. Inverted index pre-filter (anchors → candidates)
2. Emergence dynamics on candidates only
3. Selection Number determines final answer

Computational complexity:
- v0.5: O(N) where N = all facts
- v0.7: O(k) where k = anchor-matched candidates (~100)

The substrate is Wikipedia. The math finds the answer.
"""

import numpy as np
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .wiki_index import WikipediaIndex, build_index_from_sentences

# Import v0.6 emergence components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from imagine_ai.v06_emergence.field import (
    SemanticField,
    initialize_field_from_question,
    text_to_embedding
)
from imagine_ai.v06_emergence.selection import compute_selection_number, SelectionResult
from imagine_ai.v06_emergence.evolution import (
    evolve_to_closure,
    find_answer_in_substrate,
    EvolutionParameters,
    ClosureResult
)


# Stop words for anchor extraction
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
    'this', 'that', 'these', 'those', 'am', 'i', 'me', 'my', 'myself',
    'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'whose', 'many', 'much', 'any'
}


@dataclass
class WikiEmergenceResult:
    """Result of Wikipedia-scale emergence."""
    answer: Optional[str]
    confidence: float
    selection: SelectionResult
    candidates_found: int
    candidates_checked: int
    locked: bool


class WikipediaSolver:
    """
    Wikipedia-scale emergence solver.
    
    Uses inverted index for O(k) candidate retrieval,
    then applies emergence dynamics to find the answer.
    """
    
    def __init__(self, glove, verbose: bool = False):
        self.glove = glove
        self.verbose = verbose
        
        # Wikipedia index (loaded or built)
        self.index: WikipediaIndex = None
        
        # Evolution parameters
        self.params = EvolutionParameters(
            D=0.3,
            Λ=0.1,
            γ=0.05,
            κ=0.1,
            dt=0.5,
            lock_threshold=0.05,
            S_threshold=0.5
        )
    
    def load_index(self, path: str):
        """Load pre-built index."""
        if self.verbose:
            print(f"Loading index from {path}...")
        self.index = WikipediaIndex.load(path)
        if self.verbose:
            print(f"Loaded {self.index.total_facts} facts")
    
    def build_index(self, sentences: List[str]):
        """Build index from sentences."""
        if self.verbose:
            print(f"Building index from {len(sentences)} sentences...")
        self.index = build_index_from_sentences(
            sentences, 
            glove=self.glove,
            show_progress=self.verbose
        )
        if self.verbose:
            print(f"Index built: {self.index.total_facts} facts")
    
    def extract_anchors(self, question: str) -> Tuple[List[str], List[np.ndarray]]:
        """Extract anchors from question."""
        words = re.findall(r'\b[a-z]+\b', question.lower())
        
        anchor_words = []
        anchor_vectors = []
        
        for word in words:
            if word not in STOP_WORDS and word in self.glove:
                anchor_words.append(word)
                anchor_vectors.append(self.glove[word])
        
        return anchor_words, anchor_vectors
    
    def solve(
        self, 
        question: str, 
        max_candidates: int = 100,
        max_iterations: int = 50
    ) -> WikiEmergenceResult:
        """
        Solve via Wikipedia-scale emergence.
        
        1. Extract anchors (ρ_q)
        2. Find candidates via inverted index
        3. Apply emergence dynamics
        4. Return answer where S > 1
        """
        if self.index is None:
            raise ValueError("No index loaded. Call load_index() or build_index() first.")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QUESTION: {question}")
            print(f"{'='*60}")
        
        # 1. Extract anchors
        anchor_words, anchors = self.extract_anchors(question)
        
        if self.verbose:
            print(f"Anchors (ρ_q): {anchor_words}")
        
        # 2. Find candidates via inverted index
        candidate_indices = self.index.find_candidates(
            anchor_words,
            glove=self.glove,
            semantic_expansion=True,
            max_candidates=max_candidates
        )
        
        if self.verbose:
            print(f"Candidates found: {len(candidate_indices)}")
        
        if not candidate_indices:
            return WikiEmergenceResult(
                answer=None,
                confidence=0.0,
                selection=SelectionResult(S=0, R=0, R_dot=1, t_ref=1, regime="subcritical"),
                candidates_found=0,
                candidates_checked=0,
                locked=False
            )
        
        # 3. Get candidate facts and embeddings
        candidate_facts = self.index.get_facts(candidate_indices)
        candidate_embeddings = self.index.get_embeddings(candidate_indices)
        
        # If no pre-computed embeddings, compute now
        if not candidate_embeddings:
            candidate_embeddings = [
                text_to_embedding(fact, self.glove)
                for fact in candidate_facts
            ]
        
        # 4. Initialize field from question
        field = initialize_field_from_question(question, self.glove)
        
        # 5. Get substrate points (candidate embeddings)
        substrate_points = candidate_embeddings[:20]  # Top 20 for evolution
        context_vectors = anchors if anchors else [field.Φ]
        
        # 6. Evolve to closure
        closure = evolve_to_closure(
            field=field,
            substrate_points=substrate_points,
            context_vectors=context_vectors,
            anchors=anchors,
            anchor_words=anchor_words,
            question=question,
            params=self.params,
            max_iterations=max_iterations,
            verbose=self.verbose
        )
        
        # 7. Find answer in candidates (not full index!)
        answer, confidence = find_answer_in_substrate(
            Φ_final=closure.Φ_final,
            facts=candidate_facts,
            fact_embeddings=candidate_embeddings,
            glove=self.glove,
            anchor_words=anchor_words
        )
        
        if self.verbose:
            print(f"\nRESULT:")
            print(f"  Answer: {answer}")
            print(f"  S: {closure.S_final.S:.4f}")
            print(f"  Confidence: {confidence:.4f}")
        
        return WikiEmergenceResult(
            answer=answer,
            confidence=confidence,
            selection=closure.S_final,
            candidates_found=len(candidate_indices),
            candidates_checked=len(candidate_facts),
            locked=closure.locked
        )


# =============================================================================
# Demo with sample data
# =============================================================================

SAMPLE_FACTS = [
    # Capitals
    "Washington D.C. is the capital of the United States.",
    "London is the capital of the United Kingdom.",
    "Tokyo is the capital of Japan.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Rome is the capital of Italy.",
    "Madrid is the capital of Spain.",
    "Beijing is the capital of China.",
    "Moscow is the capital of Russia.",
    "Ottawa is the capital of Canada.",
    "Canberra is the capital of Australia.",
    "New Delhi is the capital of India.",
    "Brasilia is the capital of Brazil.",
    "Cairo is the capital of Egypt.",
    "Nairobi is the capital of Kenya.",
    
    # US State Capitals
    "Jackson is the capital of Mississippi.",
    "Montgomery is the capital of Alabama.",
    "Austin is the capital of Texas.",
    "Denver is the capital of Colorado.",
    "Phoenix is the capital of Arizona.",
    "Sacramento is the capital of California.",
    "Tallahassee is the capital of Florida.",
    "Atlanta is the capital of Georgia.",
    "Springfield is the capital of Illinois.",
    "Indianapolis is the capital of Indiana.",
    
    # Solar System
    "Mercury is the smallest planet and closest to the Sun.",
    "Venus is the hottest planet in our solar system.",
    "Earth is the only planet known to support life.",
    "Mars is known as the Red Planet.",
    "Jupiter is the largest planet in our solar system.",
    "Saturn is famous for its beautiful rings.",
    "Uranus rotates on its side.",
    "Neptune is the windiest planet.",
    "Pluto was reclassified as a dwarf planet in 2006.",
    
    # Geography
    "Mount Everest is the tallest mountain on Earth.",
    "The Nile is the longest river in Africa.",
    "The Amazon is the largest river by volume.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "The Sahara is the largest hot desert in the world.",
    "Lake Baikal is the deepest lake in the world.",
    "The Dead Sea is the lowest point on land.",
    
    # Science
    "Water freezes at zero degrees Celsius.",
    "Light travels at approximately 300,000 kilometers per second.",
    "DNA contains the genetic instructions for life.",
    "The human body has 206 bones.",
    "Oxygen makes up about 21% of Earth's atmosphere.",
    
    # History
    "World War II ended in 1945.",
    "The Declaration of Independence was signed in 1776.",
    "The Berlin Wall fell in 1989.",
    "Neil Armstrong walked on the Moon in 1969.",
    
    # Animals
    "The blue whale is the largest animal ever known.",
    "Cheetahs are the fastest land animals.",
    "Elephants are the largest land animals.",
    "Hummingbirds are the smallest birds.",
]


def create_demo_solver(glove, verbose: bool = False) -> WikipediaSolver:
    """Create solver with sample facts."""
    solver = WikipediaSolver(glove, verbose=verbose)
    solver.build_index(SAMPLE_FACTS)
    return solver
