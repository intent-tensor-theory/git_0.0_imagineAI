"""
anchors.py - Anchor Word Extraction

Anchors are the specific entities/concepts that must appear in the answer.

From "What is the capital of Mississippi?":
    Anchors: ["capital", "mississippi"]
    
The answer must contain these to be valid.
This provides SPECIFICITY that gradient-only matching loses.
"""

import re
from typing import List, Set, Tuple
from dataclasses import dataclass


# =============================================================================
# Stop Words and Question Words
# =============================================================================

STOP_WORDS = {
    # Articles
    'a', 'an', 'the',
    
    # Pronouns
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your',
    'he', 'him', 'his', 'she', 'her', 'it', 'its',
    'they', 'them', 'their',
    
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'about', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'over',
    
    # Conjunctions
    'and', 'or', 'but', 'so', 'yet', 'nor',
    
    # Common verbs (as stop words)
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'can', 'could', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must',
    
    # Other common words
    'that', 'this', 'these', 'those',
    'there', 'here', 'where',
    'some', 'any', 'all', 'each', 'every',
    'very', 'just', 'also', 'only',
}

QUESTION_WORDS = {
    'what', 'who', 'whom', 'whose', 'which',
    'where', 'when', 'why', 'how',
    'tell', 'describe', 'explain', 'give',
    'me', 'about',
}

# Words that indicate a question but aren't anchors
QUESTION_CONTEXT = {
    'please', 'know', 'find', 'look', 'search',
    'information', 'details', 'facts',
}


# =============================================================================
# Anchor Extraction
# =============================================================================

@dataclass
class AnchorResult:
    """Result of anchor extraction."""
    anchors: List[str]          # Extracted anchor words
    all_words: List[str]        # All words in input
    filtered_words: List[str]   # Words after stop word removal


def extract_anchors(text: str, min_length: int = 3) -> AnchorResult:
    """
    Extract anchor words from text.
    
    Anchors are words that:
    - Are not stop words
    - Are not question words
    - Have minimum length (filters out small words)
    
    Args:
        text: Input text
        min_length: Minimum word length to be anchor
        
    Returns:
        AnchorResult with extracted anchors
    """
    # Lowercase and tokenize
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    
    # Filter
    anchors = []
    filtered = []
    
    all_stop = STOP_WORDS | QUESTION_WORDS | QUESTION_CONTEXT
    
    for word in words:
        if word in all_stop:
            continue
        if len(word) < min_length:
            continue
        
        filtered.append(word)
        anchors.append(word)
    
    return AnchorResult(
        anchors=anchors,
        all_words=words,
        filtered_words=filtered
    )


def anchor_overlap(query_anchors: List[str], target_words: List[str]) -> Tuple[int, int, float]:
    """
    Count how many query anchors appear in target.
    
    Args:
        query_anchors: Anchor words from question
        target_words: Words in candidate answer
        
    Returns:
        (matches, total, ratio)
    """
    if not query_anchors:
        return (0, 0, 1.0)  # No anchors = no constraint
    
    target_set = set(w.lower() for w in target_words)
    
    matches = sum(1 for a in query_anchors if a.lower() in target_set)
    total = len(query_anchors)
    ratio = matches / total if total > 0 else 1.0
    
    return (matches, total, ratio)


def anchor_sigma(query_anchors: List[str], target_words: List[str]) -> float:
    """
    Calculate σ_anchor: penalty for missing anchors.
    
    σ_anchor = 1 - (matches / total_anchors)
    
    - 0.0 = all anchors present (no penalty)
    - 1.0 = no anchors present (full penalty)
    
    Args:
        query_anchors: Anchor words from question
        target_words: Words in candidate answer
        
    Returns:
        σ_anchor value
    """
    matches, total, ratio = anchor_overlap(query_anchors, target_words)
    
    return 1.0 - ratio  # 0 if all present, 1 if none present


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Anchor Extraction Test")
    print("=" * 60)
    
    test_questions = [
        "What is the capital of Mississippi?",
        "What is the largest planet?",
        "How fast does light travel?",
        "Who wrote Hamlet?",
        "What planet has rings?",
        "Tell me about Mars",
    ]
    
    for q in test_questions:
        result = extract_anchors(q)
        print(f"\n'{q}'")
        print(f"  Anchors: {result.anchors}")
    
    print("\n" + "=" * 60)
    print("Anchor Overlap Test")
    print("=" * 60)
    
    q = "What is the capital of Mississippi?"
    q_anchors = extract_anchors(q).anchors
    
    candidates = [
        "Jackson is the capital of Mississippi",
        "Montgomery is the capital of Alabama",
        "Jupiter is the largest planet",
    ]
    
    print(f"\nQuestion anchors: {q_anchors}")
    
    for cand in candidates:
        cand_words = re.findall(r'\b[a-z]+\b', cand.lower())
        matches, total, ratio = anchor_overlap(q_anchors, cand_words)
        sigma = anchor_sigma(q_anchors, cand_words)
        print(f"\n'{cand}'")
        print(f"  Matches: {matches}/{total} = {ratio:.2f}")
        print(f"  σ_anchor: {sigma:.2f}")
    
    print("\n✓ Anchor extraction working")
