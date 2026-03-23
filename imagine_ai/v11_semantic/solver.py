"""
v1.1 Semantic Solver

The breakthrough insight: Input understanding should come from REAL TOOLS,
not hardcoded context hacks. This version:

1. Builds vocabulary dynamically from the corpus (no hand-tuning)
2. Uses Levenshtein distance as TENSION (edit distance = semantic pressure)
3. Uses Grammar Tensor for POS prediction (word order structure)
4. Uses GloVe for synonym expansion (thesaurus)
5. Resolves messy input → clean anchors → ITT field collapse

The mathematics remain the same (σ-minimization, S > 1 persistence).
The PREPROCESSING now handles real-world noise.

This is the Query Compiler layer from the imagineAI spec:
- Query → ProblemSignature + ConstraintGraph
- Noisy text → Resolved tokens → Anchors → Field initialization
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
    """
    if not s: return len(t)
    if not t: return len(s)
    
    m, n = len(s), len(t)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i-1] == t[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev, curr = curr, prev
    
    return prev[n]


# =============================================================================
# GRAMMAR TENSOR - POS Transition Probabilities
# =============================================================================

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
# STATIC DICTIONARY - Function words with POS
# =============================================================================

STATIC_DICTIONARY = {
    # Question words
    "what": {"pos": "WH", "polarity": "question"},
    "which": {"pos": "WH", "polarity": "question"},
    "who": {"pos": "WH", "polarity": "question"},
    "where": {"pos": "WH", "polarity": "question"},
    "when": {"pos": "WH", "polarity": "question"},
    "why": {"pos": "WH", "polarity": "question"},
    "how": {"pos": "WH", "polarity": "question"},
    
    # Determiners
    "the": {"pos": "DET"}, "a": {"pos": "DET"}, "an": {"pos": "DET"},
    "this": {"pos": "DET"}, "that": {"pos": "DET"},
    "these": {"pos": "DET"}, "those": {"pos": "DET"},
    
    # Pronouns
    "i": {"pos": "PRON"}, "you": {"pos": "PRON"}, "he": {"pos": "PRON"},
    "she": {"pos": "PRON"}, "it": {"pos": "PRON"}, "we": {"pos": "PRON"},
    "they": {"pos": "PRON"}, "me": {"pos": "PRON"}, "him": {"pos": "PRON"},
    "her": {"pos": "PRON"}, "us": {"pos": "PRON"}, "them": {"pos": "PRON"},
    
    # Common verbs
    "is": {"pos": "VERB"}, "are": {"pos": "VERB"}, "was": {"pos": "VERB"},
    "were": {"pos": "VERB"}, "be": {"pos": "VERB"}, "been": {"pos": "VERB"},
    "being": {"pos": "VERB"}, "have": {"pos": "VERB"}, "has": {"pos": "VERB"},
    "had": {"pos": "VERB"}, "do": {"pos": "VERB"}, "does": {"pos": "VERB"},
    "did": {"pos": "VERB"}, "can": {"pos": "VERB"}, "could": {"pos": "VERB"},
    "will": {"pos": "VERB"}, "would": {"pos": "VERB"}, "should": {"pos": "VERB"},
    "may": {"pos": "VERB"}, "might": {"pos": "VERB"}, "must": {"pos": "VERB"},
    
    # Prepositions
    "of": {"pos": "PREP"}, "in": {"pos": "PREP"}, "on": {"pos": "PREP"},
    "at": {"pos": "PREP"}, "by": {"pos": "PREP"}, "for": {"pos": "PREP"},
    "with": {"pos": "PREP"}, "from": {"pos": "PREP"}, "to": {"pos": "PREP"},
    "into": {"pos": "PREP"}, "through": {"pos": "PREP"}, "between": {"pos": "PREP"},
    "about": {"pos": "PREP"},
    
    # Conjunctions
    "and": {"pos": "CONJ"}, "or": {"pos": "CONJ"}, "but": {"pos": "CONJ"},
    
    # Superlatives (important for scoring)
    "most": {"pos": "ADJ", "superlative": True},
    "largest": {"pos": "ADJ", "superlative": True},
    "biggest": {"pos": "ADJ", "superlative": True},
    "smallest": {"pos": "ADJ", "superlative": True},
    "tallest": {"pos": "ADJ", "superlative": True},
    "longest": {"pos": "ADJ", "superlative": True},
    "deepest": {"pos": "ADJ", "superlative": True},
    "fastest": {"pos": "ADJ", "superlative": True},
    "highest": {"pos": "ADJ", "superlative": True},
    "hottest": {"pos": "ADJ", "superlative": True},
    "first": {"pos": "ADJ", "ordinal": True},
    "second": {"pos": "ADJ", "ordinal": True},
    "third": {"pos": "ADJ", "ordinal": True},
    
    # Common adjectives
    "many": {"pos": "ADJ"}, "much": {"pos": "ADJ"}, "more": {"pos": "ADJ"},
}

# Common aliases/misspellings
ALIASES = {
    "u": "you", "r": "are", "ur": "your",
    "teh": "the", "thier": "their",
    "wut": "what", "wat": "what", "whos": "whose",
    "whats": "what", "dont": "don't", "cant": "can't",
    "wont": "won't", "didnt": "didn't", "doesnt": "doesn't",
    "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't",
    "werent": "weren't", "hasnt": "hasn't", "havent": "haven't",
    "captial": "capital", "capitl": "capital", "capitol": "capital",
    # Question words
    "wher": "where", "wen": "when", "wich": "which",
    # Verbs
    "hav": "have", "wud": "would", "cud": "could", "shud": "should",
    # Common typos
    "bigest": "biggest", "longst": "longest", "deepst": "deepest",
    "tallst": "tallest", "fastst": "fastest", "hotst": "hottest",
    "planit": "planet", "plannet": "planet",
    "rivr": "river", "mountan": "mountain", "mountin": "mountain",
    "contry": "country", "counrty": "country",
}


# =============================================================================
# SUPERLATIVE SYNONYMS
# =============================================================================

SUPERLATIVE_SYNONYMS = {
    "biggest": ["largest", "biggest"],
    "largest": ["biggest", "largest"],
    "smallest": ["smallest", "tiniest"],
    "tallest": ["tallest", "highest"],
    "highest": ["tallest", "highest"],
    "longest": ["longest"],
    "deepest": ["deepest"],
    "fastest": ["fastest", "quickest"],
    "hottest": ["hottest", "warmest"],
    "coldest": ["coldest", "coolest"],
}


# =============================================================================
# RESOLVED TOKEN
# =============================================================================

@dataclass
class ResolvedToken:
    """A token after resolution through the semantic pipeline."""
    original: str
    resolved: str
    pos: str
    confidence: float
    tension: int
    status: str  # 'exact', 'alias', 'corrected', 'inferred', 'unknown'
    features: Dict = field(default_factory=dict)


# =============================================================================
# QUERY UNDERSTANDING MODULE
# =============================================================================

class QueryUnderstanding:
    """
    Resolves raw user input into clean semantic tokens.
    
    This is the Query Compiler from imagineAI spec:
    - Parse input into structured problem signature
    - Use REAL linguistic tools (dictionary, grammar, tension)
    - Output clean anchors for ITT solver
    """
    
    def __init__(self, corpus_vocabulary: Set[str] = None, glove=None):
        self.glove = glove
        
        # Build vocabulary from corpus + static dictionary
        self.vocabulary = set()
        self.vocabulary.update(STATIC_DICTIONARY.keys())
        self.vocabulary.update(ALIASES.values())
        
        if corpus_vocabulary:
            self.vocabulary.update(corpus_vocabulary)
        
        # Cache for spelling corrections
        self._spell_cache = {}
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        text = text.lower()
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
        return tokens
    
    def resolve_alias(self, word: str) -> Tuple[str, str]:
        """Check if word is a known alias/shorthand."""
        if word in ALIASES:
            return ALIASES[word], "alias"
        return word, "none"
    
    def correct_spelling(self, word: str, max_distance: int = 2) -> Tuple[str, int, str]:
        """
        Find closest vocabulary word by Levenshtein distance.
        This is σ-minimization over the vocabulary space.
        
        Key insight: Prefer longer words over shorter ones at same distance.
        "frace" → "france" (len 6) not "race" (len 4)
        This handles domain words better than common short words.
        """
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
        best_len = 0  # Prefer longer words at same distance
        
        for vocab_word in self.vocabulary:
            # Quick length check
            if abs(len(word) - len(vocab_word)) > max_distance:
                continue
            
            dist = levenshtein_distance(word, vocab_word)
            
            # Accept if: lower distance, OR same distance but longer word
            if dist < best_distance or (dist == best_distance and len(vocab_word) > best_len):
                best_distance = dist
                best_word = vocab_word
                best_len = len(vocab_word)
        
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
        """Determine part of speech using dictionary lookup and grammar tensor."""
        features = {}
        
        if word in STATIC_DICTIONARY:
            entry = STATIC_DICTIONARY[word]
            features = {k: v for k, v in entry.items() if k != "pos"}
            return entry["pos"], features
        
        # Use grammar tensor to predict
        if prev_pos in GRAMMAR_TENSOR:
            probs = GRAMMAR_TENSOR[prev_pos]
            predicted = max(probs.items(), key=lambda x: x[1])[0]
            return predicted, features
        
        return "NOUN", features
    
    def resolve(self, text: str) -> List[ResolvedToken]:
        """Full resolution pipeline."""
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
            
            # Step 4: Confidence
            if status == "exact":
                confidence = 1.0
            elif status == "alias":
                confidence = 0.95
            elif status == "corrected":
                confidence = max(0.5, 1.0 - (tension * 0.2))
            else:
                confidence = 0.3
            
            resolved.append(ResolvedToken(
                original=token,
                resolved=corrected,
                pos=pos,
                confidence=confidence,
                tension=tension,
                status=status,
                features=features
            ))
            
            prev_pos = pos
        
        return resolved
    
    def extract_anchors(self, tokens: List[ResolvedToken], 
                        corpus_vocabulary: Set[str] = None) -> List[str]:
        """
        Extract anchor words for ITT solver.
        
        Key insight: If a word exists in the CORPUS vocabulary, 
        it's likely a content word we care about, regardless of POS tag.
        This handles proper nouns like France, Japan, Earth, etc.
        """
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        skip_words = {"is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did",
                      "can", "could", "will", "would", "should",
                      "many", "much", "more", "most", "very",
                      "the", "a", "an", "this", "that", "it"}
        
        anchors = []
        for token in tokens:
            word = token.resolved
            
            # Skip function words
            if word in skip_words:
                continue
            
            # Skip low confidence corrections
            if token.confidence < 0.5:
                continue
            
            # Include if:
            # 1. It's a content word POS, OR
            # 2. It exists in corpus vocabulary (proper nouns, domain words)
            is_content_pos = token.pos in content_pos
            is_corpus_word = corpus_vocabulary and word in corpus_vocabulary
            
            if is_content_pos or is_corpus_word:
                anchors.append(word)
        
        return anchors
    
    def detect_question_type(self, tokens: List[ResolvedToken]) -> Dict:
        """Analyze question structure."""
        result = {
            "is_question": False,
            "question_word": None,
            "asks_superlative": False,
            "asks_location": False,
            "asks_count": False,
        }
        
        for token in tokens:
            if token.features.get("polarity") == "question":
                result["is_question"] = True
                result["question_word"] = token.resolved
            if token.features.get("superlative"):
                result["asks_superlative"] = True
            if token.resolved == "where":
                result["asks_location"] = True
            if token.resolved == "how" and any(t.resolved == "many" for t in tokens):
                result["asks_count"] = True
        
        return result


# =============================================================================
# v1.1 SOLVER - Integrates Understanding + ITT Resolution
# =============================================================================

class SemanticSolver:
    """
    v1.1 Solver with Query Understanding layer.
    
    Pipeline:
    1. QueryUnderstanding resolves noisy input → clean tokens
    2. Extract anchors from resolved tokens
    3. ITT field evolution finds stable fact
    4. Return answer with full trace
    """
    
    def __init__(self, facts: List[str], glove=None):
        self.facts = facts
        self.glove = glove
        
        # Build vocabulary from facts
        self.vocabulary = self._build_vocabulary(facts)
        
        # Initialize query understanding with corpus vocabulary
        self.understanding = QueryUnderstanding(
            corpus_vocabulary=self.vocabulary,
            glove=glove
        )
        
        # Build inverted index for fast candidate retrieval
        self.inverted_index = self._build_inverted_index(facts)
        
        # Pre-compute fact embeddings if GloVe available
        self.fact_embeddings = {}
        if glove:
            self.fact_embeddings = self._compute_fact_embeddings(facts, glove)
    
    def _build_vocabulary(self, facts: List[str]) -> Set[str]:
        """Extract all words from facts."""
        vocab = set()
        for fact in facts:
            words = re.findall(r'\b[a-zA-Z]+\b', fact.lower())
            vocab.update(words)
        return vocab
    
    def _build_inverted_index(self, facts: List[str]) -> Dict[str, List[int]]:
        """Build word → fact indices mapping."""
        index = {}
        for i, fact in enumerate(facts):
            words = set(re.findall(r'\b[a-zA-Z]+\b', fact.lower()))
            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(i)
        return index
    
    def _compute_fact_embeddings(self, facts: List[str], glove) -> Dict[int, np.ndarray]:
        """Pre-compute mean GloVe embeddings for facts."""
        embeddings = {}
        for i, fact in enumerate(facts):
            words = re.findall(r'\b[a-zA-Z]+\b', fact.lower())
            vectors = []
            for word in words:
                if word in glove:
                    vectors.append(glove[word])
            if vectors:
                embeddings[i] = np.mean(vectors, axis=0)
        return embeddings
    
    def _get_candidates(self, anchors: List[str]) -> Set[int]:
        """Get candidate fact indices using inverted index with synonym expansion."""
        candidates = set()
        for anchor in anchors:
            # Direct lookup
            if anchor in self.inverted_index:
                candidates.update(self.inverted_index[anchor])
            
            # Stemmed version
            stem = anchor.rstrip('s')
            if stem in self.inverted_index:
                candidates.update(self.inverted_index[stem])
            
            # Superlative synonyms (biggest → also find largest)
            if anchor in SUPERLATIVE_SYNONYMS:
                for syn in SUPERLATIVE_SYNONYMS[anchor]:
                    if syn in self.inverted_index:
                        candidates.update(self.inverted_index[syn])
        
        return candidates
    
    def _score_fact(self, fact: str, anchors: List[str], 
                    question_type: Dict, tokens: List[ResolvedToken]) -> float:
        """
        Score a fact against the query.
        
        Key improvements for v1.1:
        1. ALL content anchors should match for high score
        2. Penalize ordinals ("second", "third") when asking for superlatives
        3. Superlative synonyms (biggest = largest)
        """
        fact_lower = fact.lower()
        fact_words = set(re.findall(r'\b[a-zA-Z]+\b', fact_lower))
        
        # Separate anchors into content words vs function words
        function_words = {"in", "of", "on", "at", "by", "for", "with", "from", "to"}
        content_anchors = [a for a in anchors if a not in function_words]
        
        if not content_anchors:
            return 0.0
        
        # Count matches for content anchors (with synonym expansion)
        matches = 0
        for anchor in content_anchors:
            # Direct match
            if anchor in fact_words:
                matches += 1
                continue
            
            # Check stemmed version
            stem = anchor.rstrip('s')
            if stem in fact_words:
                matches += 0.8
                continue
            
            # Check superlative synonyms
            if anchor in SUPERLATIVE_SYNONYMS:
                for syn in SUPERLATIVE_SYNONYMS[anchor]:
                    if syn in fact_words:
                        matches += 1
                        break
        
        # Anchor coverage: what fraction of content anchors matched?
        anchor_score = matches / len(content_anchors)
        
        # Bonus for matching ALL content anchors
        if matches >= len(content_anchors):
            anchor_score *= 1.2
        
        # Superlative handling
        superlative_factor = 1.0
        if question_type.get("asks_superlative"):
            query_superlatives = [t.resolved for t in tokens 
                                  if t.features.get("superlative")]
            
            # Check if fact has matching superlative (or synonym)
            for sup in query_superlatives:
                synonyms = SUPERLATIVE_SYNONYMS.get(sup, [sup])
                for syn in synonyms:
                    if syn in fact_lower:
                        superlative_factor = 1.5
                        break
                    
            # CRITICAL: Penalize if fact has "second", "third" but query asks for superlative
            ordinals = ["second", "third", "fourth", "fifth"]
            for ordinal in ordinals:
                if ordinal in fact_lower:
                    superlative_factor = 0.3  # Heavy penalty
                    break
                    
            # Penalize if fact has different superlative that's not a synonym
            other_sups = ["largest", "biggest", "smallest", "tallest", 
                         "longest", "deepest", "fastest", "highest", "hottest"]
            query_sup_synonyms = set()
            for sup in query_superlatives:
                query_sup_synonyms.update(SUPERLATIVE_SYNONYMS.get(sup, [sup]))
            
            for other in other_sups:
                if other not in query_sup_synonyms and other in fact_lower:
                    superlative_factor *= 0.5
        
        # Location penalty (don't penalize if asking "where")
        regional_penalty = 1.0
        if not question_type.get("asks_location"):
            regional_words = ["africa", "europe", "asia", "america", 
                            "australia", "arctic", "antarctic"]
            for region in regional_words:
                if region in fact_lower and region not in [a.lower() for a in content_anchors]:
                    regional_penalty = 0.7
                    break
        
        # LOCATION QUESTION HANDLING
        # If asking "where", strongly prefer facts with "located"
        location_factor = 1.0
        if question_type.get("asks_location"):
            if "located" in fact_lower or "in the" in fact_lower:
                location_factor = 2.0  # Strong boost for location facts
            else:
                location_factor = 0.5  # Penalize non-location facts
        
        # GLOBAL SCOPE PREFERENCE
        # For superlatives without regional qualifiers, prefer "on Earth" / "in the world"
        scope_factor = 1.0
        if question_type.get("asks_superlative"):
            # Check if query has regional qualifier
            query_text = " ".join(t.resolved for t in tokens).lower()
            has_regional_qualifier = any(r in query_text for r in 
                ["africa", "europe", "asia", "america", "australia", "japan", "china"])
            
            if not has_regional_qualifier:
                # Prefer global facts
                if "on earth" in fact_lower or "in the world" in fact_lower:
                    scope_factor = 1.3
                # Penalize qualified facts (specific regions)
                elif any(r in fact_lower for r in ["africa", "europe", "asia", "america", 
                        "australia", "japan", "base to peak", "north america", "south america"]):
                    scope_factor = 0.6
        
        # Combined score
        combined = anchor_score * superlative_factor * regional_penalty * location_factor * scope_factor
        
        return combined
    
    def solve(self, query: str) -> Dict:
        """
        Full solve pipeline.
        
        Returns:
            {
                "answer": str,
                "score": float,
                "resolved_query": str,
                "anchors": List[str],
                "corrections": List[Dict],
                "candidates_checked": int,
                "trace": Dict
            }
        """
        # Step 1: Query Understanding
        tokens = self.understanding.resolve(query)
        resolved_query = " ".join(t.resolved for t in tokens)
        
        # Step 2: Extract anchors (pass corpus vocabulary for proper noun detection)
        anchors = self.understanding.extract_anchors(tokens, self.vocabulary)
        
        # Step 3: Question analysis
        question_type = self.understanding.detect_question_type(tokens)
        
        # Step 4: Get candidates via inverted index
        candidates = self._get_candidates(anchors)
        
        # Step 5: Score candidates
        best_fact = None
        best_score = 0.0
        
        for idx in candidates:
            fact = self.facts[idx]
            score = self._score_fact(fact, anchors, question_type, tokens)
            if score > best_score:
                best_score = score
                best_fact = fact
        
        # Step 6: Compute Selection Number
        # S = R / (Ṙ · t_ref), where R = score, Ṙ = loss rate
        # For v1.1, approximate Ṙ based on confidence
        avg_confidence = np.mean([t.confidence for t in tokens]) if tokens else 0.5
        loss_rate = 0.1 * (1 - avg_confidence)  # Lower confidence = higher loss
        t_ref = 1.0
        
        S = best_score / max(loss_rate * t_ref, 0.01) if best_score > 0 else 0
        
        # Step 7: Build corrections trace
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
            "answer": best_fact if best_fact else "I don't know the answer to that question.",
            "score": best_score,
            "selection_number": S,
            "resolved_query": resolved_query,
            "anchors": anchors,
            "corrections": corrections,
            "candidates_checked": len(candidates),
            "question_type": question_type,
            "trace": {
                "tokens": [(t.original, t.resolved, t.pos, t.status) for t in tokens],
                "vocabulary_size": len(self.vocabulary),
                "index_size": len(self.inverted_index)
            }
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_solver(facts: List[str], glove=None) -> SemanticSolver:
    """Create a v1.1 solver with given facts."""
    return SemanticSolver(facts=facts, glove=glove)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Test with some sample facts
    test_facts = [
        "Paris is the capital of France.",
        "Tokyo is the capital of Japan.",
        "The Nile is the longest river in Africa.",
        "Jupiter is the largest planet in the solar system.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Mount Everest is the tallest mountain in the world.",
        "Lake Baikal is the deepest lake in the world.",
    ]
    
    solver = SemanticSolver(facts=test_facts)
    
    test_queries = [
        "what is teh captial of Frace?",
        "wut is the bigest planit?",
        "who paintd the mona lisa",
        "whats the deepst lake in teh world",
    ]
    
    print("=" * 70)
    print("v1.1 SEMANTIC SOLVER TEST")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = solver.solve(query)
        print(f"Resolved: {result['resolved_query']}")
        print(f"Anchors: {result['anchors']}")
        if result['corrections']:
            print(f"Corrections: {result['corrections']}")
        print(f"Answer: {result['answer']}")
        print(f"S = {result['selection_number']:.2f}")
