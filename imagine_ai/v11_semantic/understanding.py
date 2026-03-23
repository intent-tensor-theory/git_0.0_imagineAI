"""
v1.1 Query Understanding Module

The problem with v1.0: It assumes clean input.
Real users type: "what is teh captial of Frace?"

This module resolves input BEFORE the ITT solver:
1. Spelling Correction - Levenshtein distance against vocabulary
2. Grammar Analysis - POS tagging to understand structure
3. Semantic Expansion - Synonyms via GloVe similarity
4. Variant Resolution - singular/plural, verb forms

The philosophy: Use REAL linguistic tools (dictionary, thesaurus, grammar)
not hardcoded contextual hacks. The system should UNDERSTAND the input,
not just pattern match against it.

Mathematical framing:
- Input text = noisy observation
- True query = latent intent
- Resolution = σ-minimization over linguistic space
- Tension = edit distance (Levenshtein)
- Resonance = semantic similarity (GloVe cosine)
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field


# =============================================================================
# TENSION FUNCTION (Levenshtein Edit Distance)
# =============================================================================

def levenshtein_distance(s: str, t: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    
    This IS the tension function - how "far" is the observed word
    from a known vocabulary word?
    
    Tension = 0: exact match (stable)
    Tension = 1: one edit (minor perturbation)
    Tension = 2: two edits (still resolvable)
    Tension > 2: too far (leave as-is or flag)
    """
    if not s: return len(t)
    if not t: return len(s)
    
    m, n = len(s), len(t)
    
    # Use two rows for space efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i-1] == t[j-1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j-1] + 1,    # insertion
                prev[j-1] + cost  # substitution
            )
        prev, curr = curr, prev
    
    return prev[n]


# =============================================================================
# GRAMMAR TENSOR - Transition Probabilities
# =============================================================================

# POS transition probabilities: P(next_POS | current_POS)
# This encodes basic English grammar structure
GRAMMAR_TENSOR = {
    "START": {"DET": 0.25, "NOUN": 0.20, "PRON": 0.20, "VERB": 0.15, "ADV": 0.10, "ADJ": 0.05, "PREP": 0.05},
    "DET": {"NOUN": 0.60, "ADJ": 0.35, "VERB": 0.03, "ADV": 0.02},
    "NOUN": {"VERB": 0.35, "PREP": 0.25, "NOUN": 0.15, "ADJ": 0.10, "DET": 0.05, "CONJ": 0.10},
    "VERB": {"DET": 0.25, "NOUN": 0.20, "PREP": 0.20, "ADV": 0.15, "ADJ": 0.10, "PRON": 0.10},
    "ADJ": {"NOUN": 0.70, "ADJ": 0.15, "VERB": 0.10, "PREP": 0.05},
    "ADV": {"VERB": 0.40, "ADJ": 0.30, "ADV": 0.15, "NOUN": 0.10, "PREP": 0.05},
    "PREP": {"DET": 0.40, "NOUN": 0.35, "PRON": 0.15, "ADJ": 0.10},
    "PRON": {"VERB": 0.50, "NOUN": 0.20, "ADV": 0.15, "PREP": 0.15},
    "CONJ": {"DET": 0.30, "NOUN": 0.25, "PRON": 0.20, "VERB": 0.15, "ADJ": 0.10},
    "WH": {"VERB": 0.40, "NOUN": 0.30, "ADJ": 0.15, "ADV": 0.15},
    "UNKNOWN": {"NOUN": 0.40, "VERB": 0.30, "ADJ": 0.15, "ADV": 0.15}
}


# =============================================================================
# STATIC DICTIONARY - Known Words with POS
# =============================================================================

# Common words with their part-of-speech
# This is the "dictionary" the system can look up
STATIC_DICTIONARY = {
    # Question words (WH)
    "what": {"pos": "WH", "polarity": "question"},
    "which": {"pos": "WH", "polarity": "question"},
    "who": {"pos": "WH", "polarity": "question"},
    "where": {"pos": "WH", "polarity": "question"},
    "when": {"pos": "WH", "polarity": "question"},
    "why": {"pos": "WH", "polarity": "question"},
    "how": {"pos": "WH", "polarity": "question"},
    
    # Determiners
    "the": {"pos": "DET"},
    "a": {"pos": "DET"},
    "an": {"pos": "DET"},
    "this": {"pos": "DET"},
    "that": {"pos": "DET"},
    "these": {"pos": "DET"},
    "those": {"pos": "DET"},
    
    # Pronouns
    "i": {"pos": "PRON"},
    "you": {"pos": "PRON"},
    "he": {"pos": "PRON"},
    "she": {"pos": "PRON"},
    "it": {"pos": "PRON"},
    "we": {"pos": "PRON"},
    "they": {"pos": "PRON"},
    "me": {"pos": "PRON"},
    "him": {"pos": "PRON"},
    "her": {"pos": "PRON"},
    "us": {"pos": "PRON"},
    "them": {"pos": "PRON"},
    
    # Common verbs
    "is": {"pos": "VERB"},
    "are": {"pos": "VERB"},
    "was": {"pos": "VERB"},
    "were": {"pos": "VERB"},
    "be": {"pos": "VERB"},
    "been": {"pos": "VERB"},
    "being": {"pos": "VERB"},
    "have": {"pos": "VERB"},
    "has": {"pos": "VERB"},
    "had": {"pos": "VERB"},
    "do": {"pos": "VERB"},
    "does": {"pos": "VERB"},
    "did": {"pos": "VERB"},
    "can": {"pos": "VERB", "polarity": "question"},
    "could": {"pos": "VERB"},
    "will": {"pos": "VERB"},
    "would": {"pos": "VERB"},
    "should": {"pos": "VERB"},
    "may": {"pos": "VERB"},
    "might": {"pos": "VERB"},
    "must": {"pos": "VERB"},
    
    # Prepositions
    "of": {"pos": "PREP"},
    "in": {"pos": "PREP"},
    "on": {"pos": "PREP"},
    "at": {"pos": "PREP"},
    "by": {"pos": "PREP"},
    "for": {"pos": "PREP"},
    "with": {"pos": "PREP"},
    "from": {"pos": "PREP"},
    "to": {"pos": "PREP"},
    "into": {"pos": "PREP"},
    "through": {"pos": "PREP"},
    "between": {"pos": "PREP"},
    "about": {"pos": "PREP"},
    
    # Conjunctions
    "and": {"pos": "CONJ"},
    "or": {"pos": "CONJ"},
    "but": {"pos": "CONJ"},
    
    # Common adjectives/adverbs
    "many": {"pos": "ADJ"},
    "much": {"pos": "ADJ"},
    "most": {"pos": "ADJ", "superlative": True},
    "more": {"pos": "ADJ"},
    "largest": {"pos": "ADJ", "superlative": True},
    "biggest": {"pos": "ADJ", "superlative": True},
    "smallest": {"pos": "ADJ", "superlative": True},
    "tallest": {"pos": "ADJ", "superlative": True},
    "longest": {"pos": "ADJ", "superlative": True},
    "deepest": {"pos": "ADJ", "superlative": True},
    "fastest": {"pos": "ADJ", "superlative": True},
    "first": {"pos": "ADJ", "ordinal": True},
    "second": {"pos": "ADJ", "ordinal": True},
    "third": {"pos": "ADJ", "ordinal": True},
    
    # Common nouns (domain-specific)
    "capital": {"pos": "NOUN"},
    "country": {"pos": "NOUN"},
    "city": {"pos": "NOUN"},
    "state": {"pos": "NOUN"},
    "planet": {"pos": "NOUN"},
    "mountain": {"pos": "NOUN"},
    "river": {"pos": "NOUN"},
    "ocean": {"pos": "NOUN"},
    "lake": {"pos": "NOUN"},
    "desert": {"pos": "NOUN"},
    "animal": {"pos": "NOUN"},
    "element": {"pos": "NOUN"},
    "symbol": {"pos": "NOUN"},
}

# Common misspellings / text shortcuts
ALIASES = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "teh": "the",
    "thier": "their",
    "recieve": "receive",
    "seperate": "separate",
    "occured": "occurred",
    "definately": "definitely",
    "goverment": "government",
    "enviroment": "environment",
    "occassion": "occasion",
    "accomodate": "accommodate",
    "untill": "until",
    "wich": "which",
    "wut": "what",
    "wat": "what",
    "whos": "whose",
    "whats": "what's",
    "dont": "don't",
    "cant": "can't",
    "wont": "won't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
}


# =============================================================================
# RESOLVED TOKEN
# =============================================================================

@dataclass
class ResolvedToken:
    """A token after resolution through the semantic pipeline."""
    original: str           # What user typed
    resolved: str           # What we think they meant
    pos: str                # Part of speech
    confidence: float       # How confident are we in resolution
    tension: int            # Edit distance to resolved form
    status: str             # 'exact', 'alias', 'corrected', 'inferred', 'unknown'
    synonyms: List[str] = field(default_factory=list)  # Related words
    features: Dict = field(default_factory=dict)  # Additional features


# =============================================================================
# QUERY UNDERSTANDING MODULE
# =============================================================================

class QueryUnderstanding:
    """
    Resolves raw user input into clean semantic tokens.
    
    Pipeline:
    1. Tokenize
    2. Alias resolution (u → you, teh → the)
    3. Spelling correction via Levenshtein
    4. POS tagging using grammar tensor
    5. Synonym expansion via GloVe
    
    This is the "front door" - it ensures the ITT solver
    receives clean, understood input.
    """
    
    def __init__(self, glove=None, vocabulary: Set[str] = None):
        self.glove = glove
        
        # Build vocabulary from dictionary + corpus words
        self.vocabulary = vocabulary or set()
        self.vocabulary.update(STATIC_DICTIONARY.keys())
        self.vocabulary.update(ALIASES.values())
        
        # Cache for spelling corrections
        self._spell_cache = {}
    
    def add_vocabulary(self, words: List[str]):
        """Add words to the known vocabulary."""
        self.vocabulary.update(w.lower() for w in words)
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        # Lowercase and extract words
        text = text.lower()
        # Keep contractions together initially
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
        return tokens
    
    def resolve_alias(self, word: str) -> Tuple[str, str]:
        """
        Check if word is a known alias/shorthand.
        Returns (resolved_word, status).
        """
        if word in ALIASES:
            return ALIASES[word], "alias"
        return word, "none"
    
    def correct_spelling(self, word: str, max_distance: int = 2) -> Tuple[str, int, str]:
        """
        Find closest vocabulary word by Levenshtein distance.
        Returns (corrected_word, distance, status).
        
        This is σ-minimization over the vocabulary space.
        """
        # Check cache
        if word in self._spell_cache:
            return self._spell_cache[word]
        
        # Exact match
        if word in self.vocabulary:
            result = (word, 0, "exact")
            self._spell_cache[word] = result
            return result
        
        # Find closest by edit distance
        best_word = word
        best_distance = max_distance + 1
        
        for vocab_word in self.vocabulary:
            # Quick length check to avoid unnecessary computation
            if abs(len(word) - len(vocab_word)) > max_distance:
                continue
            
            dist = levenshtein_distance(word, vocab_word)
            if dist < best_distance:
                best_distance = dist
                best_word = vocab_word
        
        if best_distance <= max_distance:
            status = "corrected"
        else:
            best_word = word
            best_distance = 0
            status = "unknown"
        
        result = (best_word, best_distance, status)
        self._spell_cache[word] = result
        return result
    
    def get_pos(self, word: str, prev_pos: str = "START") -> Tuple[str, Dict]:
        """
        Determine part of speech using dictionary lookup and grammar tensor.
        Returns (pos, features).
        """
        features = {}
        
        # Direct dictionary lookup
        if word in STATIC_DICTIONARY:
            entry = STATIC_DICTIONARY[word]
            features = {k: v for k, v in entry.items() if k != "pos"}
            return entry["pos"], features
        
        # Use grammar tensor to predict most likely POS
        if prev_pos in GRAMMAR_TENSOR:
            probs = GRAMMAR_TENSOR[prev_pos]
            # Default to most likely given previous POS
            predicted = max(probs.items(), key=lambda x: x[1])[0]
            return predicted, features
        
        return "NOUN", features  # Default assumption
    
    def get_synonyms(self, word: str, top_k: int = 5, threshold: float = 0.5) -> List[str]:
        """
        Find semantically similar words using GloVe.
        This is the thesaurus function.
        """
        if self.glove is None or word not in self.glove:
            return []
        
        try:
            similar = self.glove.most_similar(word, topn=top_k)
            return [w for w, sim in similar if sim >= threshold]
        except:
            return []
    
    def resolve(self, text: str) -> List[ResolvedToken]:
        """
        Full resolution pipeline.
        
        Input: "what is teh captial of Frace?"
        Output: List of ResolvedTokens with corrections and understanding
        """
        tokens = self.tokenize(text)
        resolved = []
        prev_pos = "START"
        
        for token in tokens:
            # Step 1: Alias resolution
            word, alias_status = self.resolve_alias(token)
            
            # Step 2: Spelling correction
            corrected, tension, spell_status = self.correct_spelling(word)
            
            # Determine final status
            if alias_status == "alias":
                status = "alias"
            elif spell_status == "corrected":
                status = "corrected"
            elif spell_status == "exact":
                status = "exact"
            else:
                status = "unknown"
            
            # Step 3: POS tagging
            pos, features = self.get_pos(corrected, prev_pos)
            
            # Step 4: Confidence based on resolution path
            if status == "exact":
                confidence = 1.0
            elif status == "alias":
                confidence = 0.95
            elif status == "corrected":
                confidence = max(0.5, 1.0 - (tension * 0.2))
            else:
                confidence = 0.3
            
            # Step 5: Synonym expansion (for content words)
            synonyms = []
            if pos in ("NOUN", "VERB", "ADJ", "ADV") and status != "unknown":
                synonyms = self.get_synonyms(corrected)
            
            resolved.append(ResolvedToken(
                original=token,
                resolved=corrected,
                pos=pos,
                confidence=confidence,
                tension=tension,
                status=status,
                synonyms=synonyms,
                features=features
            ))
            
            prev_pos = pos
        
        return resolved
    
    def extract_anchors(self, resolved_tokens: List[ResolvedToken]) -> Tuple[List[str], List[str]]:
        """
        Extract anchor words from resolved tokens.
        
        Returns:
        - primary_anchors: High-confidence content words
        - expanded_anchors: Primary + synonyms
        """
        # Content word POS tags
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        
        # Skip words (even if tagged as content)
        skip_words = {"is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did",
                      "can", "could", "will", "would", "should"}
        
        primary = []
        expanded = set()
        
        for token in resolved_tokens:
            if token.pos in content_pos and token.resolved not in skip_words:
                if token.confidence >= 0.5:
                    primary.append(token.resolved)
                    expanded.add(token.resolved)
                    expanded.update(token.synonyms[:3])  # Top 3 synonyms
        
        return primary, list(expanded)
    
    def detect_question_type(self, resolved_tokens: List[ResolvedToken]) -> Dict:
        """
        Analyze question structure to understand intent.
        """
        result = {
            "is_question": False,
            "question_word": None,
            "asks_superlative": False,
            "asks_ordinal": False,
            "asks_count": False,
            "asks_location": False,
            "asks_identity": False,
        }
        
        for token in resolved_tokens:
            # Question detection
            if token.features.get("polarity") == "question":
                result["is_question"] = True
                result["question_word"] = token.resolved
            
            # Feature detection
            if token.features.get("superlative"):
                result["asks_superlative"] = True
            if token.features.get("ordinal"):
                result["asks_ordinal"] = True
            
            # Question word specific
            if token.resolved == "how" and any(t.resolved == "many" for t in resolved_tokens):
                result["asks_count"] = True
            if token.resolved == "where":
                result["asks_location"] = True
            if token.resolved in ("who", "what"):
                result["asks_identity"] = True
        
        return result


# =============================================================================
# INTEGRATION WITH v1.0 SOLVER
# =============================================================================

def preprocess_query(query: str, glove=None, vocabulary: Set[str] = None) -> Dict:
    """
    Preprocess a query for the ITT solver.
    
    Returns a dictionary with:
    - resolved_text: Cleaned query string
    - anchors: Primary anchor words
    - expanded_anchors: Anchors + synonyms
    - tokens: Full resolved token list
    - question_type: Question analysis
    - corrections: Any spelling/alias corrections made
    """
    processor = QueryUnderstanding(glove=glove, vocabulary=vocabulary)
    
    # Add vocabulary if provided
    if vocabulary:
        processor.add_vocabulary(list(vocabulary))
    
    # Resolve
    tokens = processor.resolve(query)
    
    # Extract anchors
    primary_anchors, expanded_anchors = processor.extract_anchors(tokens)
    
    # Question analysis
    question_type = processor.detect_question_type(tokens)
    
    # Build resolved text
    resolved_text = " ".join(t.resolved for t in tokens)
    
    # Track corrections
    corrections = []
    for t in tokens:
        if t.original != t.resolved:
            corrections.append({
                "original": t.original,
                "resolved": t.resolved,
                "status": t.status,
                "confidence": t.confidence
            })
    
    return {
        "original": query,
        "resolved_text": resolved_text,
        "anchors": primary_anchors,
        "expanded_anchors": expanded_anchors,
        "tokens": tokens,
        "question_type": question_type,
        "corrections": corrections
    }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Test the query understanding module
    test_queries = [
        "what is teh captial of Frace?",
        "wut is the bigest planit?",
        "who paintd the mona lisa",
        "how many moons does earth hav",
        "whats the deepst lake in teh world",
    ]
    
    processor = QueryUnderstanding()
    
    for query in test_queries:
        print(f"\nInput: {query}")
        tokens = processor.resolve(query)
        
        resolved = " ".join(t.resolved for t in tokens)
        print(f"Resolved: {resolved}")
        
        corrections = [(t.original, t.resolved, t.status) 
                       for t in tokens if t.original != t.resolved]
        if corrections:
            print(f"Corrections: {corrections}")
        
        anchors, _ = processor.extract_anchors(tokens)
        print(f"Anchors: {anchors}")
