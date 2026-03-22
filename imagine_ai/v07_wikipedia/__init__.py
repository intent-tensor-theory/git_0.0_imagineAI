"""
imagineAI v0.7 - Wikipedia Scale Emergence

Scales v0.6 emergence to millions of facts via:
1. Inverted index pre-filter (anchors → candidates)
2. Emergence dynamics on candidates only
3. Selection Number determines final answer

Computational complexity:
- v0.5: O(N) where N = all facts
- v0.6: O(N) with emergence dynamics
- v0.7: O(k) where k = anchor-matched candidates (~100)

The substrate is Wikipedia. The math finds the answer.
"""

from .wiki_index import (
    WikipediaIndex,
    build_index_from_sentences,
    load_wikipedia_sentences
)

from .solver import (
    WikipediaSolver,
    WikiEmergenceResult,
    create_demo_solver,
    SAMPLE_FACTS
)

__version__ = "0.7.0"
__all__ = [
    "WikipediaIndex",
    "WikipediaSolver",
    "WikiEmergenceResult",
    "create_demo_solver",
    "SAMPLE_FACTS"
]
