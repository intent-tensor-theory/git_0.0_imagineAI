"""
imagineAI v0.8 - Scale Testing

Testing emergence at scale:
- 762 facts (world capitals, US states, science, history, etc.)
- 70 comprehensive test questions
- 88.6% accuracy achieved

Key improvements over v0.7:
- Ordinal context-awareness (second tallest vs per second)
- Regional qualifier penalties
- Improved word tokenization (regex-based)
- Reduced proximity bonus (tiebreaker only)
"""

from .corpus import (
    generate_all_facts,
    TEST_SUITE,
    WORLD_CAPITALS,
    US_STATE_CAPITALS,
)

__version__ = "0.8.0"
__all__ = [
    "generate_all_facts",
    "TEST_SUITE",
    "WORLD_CAPITALS", 
    "US_STATE_CAPITALS",
]
