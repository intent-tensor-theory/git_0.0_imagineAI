"""
wiki_index.py - Wikipedia Inverted Index

For emergence to scale to millions of facts, we need:
1. Inverted index: word → [fact_ids that contain it]
2. Anchor pre-filter: only evolve on facts matching ρ_q
3. Semantic expansion: also include similar words

This reduces O(N) to O(k) where k = facts matching anchors.

The index IS the substrate. The emergence happens on candidates.
"""

import numpy as np
import re
import pickle
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import gzip


@dataclass
class WikipediaIndex:
    """
    Inverted index over Wikipedia sentences.
    
    Maps: word → set of fact indices
    Allows: fast lookup of candidate facts for any anchor set
    """
    # The facts themselves
    facts: List[str] = field(default_factory=list)
    
    # Pre-computed embeddings (lazy loaded)
    embeddings: List[np.ndarray] = field(default_factory=list)
    
    # Inverted index: word → set of fact indices
    word_to_facts: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    
    # Word frequencies (for IDF weighting)
    word_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Total facts
    total_facts: int = 0
    
    def add_fact(self, fact: str, embedding: np.ndarray = None):
        """Add a fact to the index."""
        fact_idx = len(self.facts)
        self.facts.append(fact)
        
        if embedding is not None:
            self.embeddings.append(embedding)
        
        # Index words
        words = set(re.findall(r'\b[a-z]+\b', fact.lower()))
        for word in words:
            self.word_to_facts[word].add(fact_idx)
            self.word_counts[word] += 1
        
        self.total_facts += 1
    
    def find_candidates(
        self, 
        anchor_words: List[str],
        glove = None,
        semantic_expansion: bool = True,
        max_candidates: int = 100
    ) -> List[int]:
        """
        Find candidate fact indices that match anchors.
        
        Uses inverted index for O(1) lookup per anchor.
        Optionally expands anchors semantically via GloVe.
        """
        if not anchor_words:
            return list(range(min(max_candidates, self.total_facts)))
        
        # Expand anchors semantically
        expanded_anchors = set(anchor_words)
        
        if semantic_expansion and glove is not None:
            for anchor in anchor_words:
                if anchor in glove:
                    # Add similar words
                    try:
                        similar = glove.most_similar(anchor, topn=5)
                        for word, sim in similar:
                            if sim > 0.5:
                                expanded_anchors.add(word.lower())
                    except:
                        pass
        
        # Find facts matching any anchor
        candidate_scores: Dict[int, float] = defaultdict(float)
        
        for anchor in expanded_anchors:
            if anchor in self.word_to_facts:
                fact_ids = self.word_to_facts[anchor]
                
                # IDF weight: rare words count more
                idf = np.log(self.total_facts / (self.word_counts[anchor] + 1))
                
                for fact_id in fact_ids:
                    # Original anchor gets full weight, expanded gets partial
                    weight = 1.0 if anchor in anchor_words else 0.5
                    candidate_scores[fact_id] += weight * idf
        
        # Sort by score and return top candidates
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [idx for idx, score in sorted_candidates[:max_candidates]]
    
    def get_facts(self, indices: List[int]) -> List[str]:
        """Get facts by indices."""
        return [self.facts[i] for i in indices if i < len(self.facts)]
    
    def get_embeddings(self, indices: List[int]) -> List[np.ndarray]:
        """Get embeddings by indices."""
        if not self.embeddings:
            return []
        return [self.embeddings[i] for i in indices if i < len(self.embeddings)]
    
    def save(self, path: str):
        """Save index to disk."""
        data = {
            'facts': self.facts,
            'word_to_facts': {k: list(v) for k, v in self.word_to_facts.items()},
            'word_counts': dict(self.word_counts),
            'total_facts': self.total_facts
        }
        
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save embeddings separately (large)
        if self.embeddings:
            emb_path = path.replace('.pkl.gz', '_embeddings.npy')
            np.save(emb_path, np.array(self.embeddings))
    
    @classmethod
    def load(cls, path: str) -> 'WikipediaIndex':
        """Load index from disk."""
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        
        index = cls()
        index.facts = data['facts']
        index.word_to_facts = defaultdict(set, {
            k: set(v) for k, v in data['word_to_facts'].items()
        })
        index.word_counts = defaultdict(int, data['word_counts'])
        index.total_facts = data['total_facts']
        
        # Load embeddings if available
        emb_path = path.replace('.pkl.gz', '_embeddings.npy')
        if Path(emb_path).exists():
            index.embeddings = list(np.load(emb_path))
        
        return index


def build_index_from_sentences(
    sentences: List[str],
    glove = None,
    show_progress: bool = True
) -> WikipediaIndex:
    """
    Build index from a list of sentences.
    
    If glove is provided, also computes embeddings.
    """
    index = WikipediaIndex()
    
    total = len(sentences)
    
    for i, sentence in enumerate(sentences):
        # Compute embedding if glove available
        embedding = None
        if glove is not None:
            words = re.findall(r'\b[a-z]+\b', sentence.lower())
            vectors = [glove[w] for w in words if w in glove]
            if vectors:
                embedding = np.mean(vectors, axis=0)
            else:
                embedding = np.zeros(glove.vector_size)
        
        index.add_fact(sentence, embedding)
        
        if show_progress and (i + 1) % 10000 == 0:
            print(f"  Indexed {i+1}/{total} sentences...")
    
    return index


# =============================================================================
# Wikipedia Loading Utilities
# =============================================================================

def load_wikipedia_sentences(
    path: str,
    max_sentences: int = None,
    min_length: int = 20,
    max_length: int = 200
) -> List[str]:
    """
    Load sentences from a Wikipedia dump file.
    
    Expected format: one sentence per line, or standard Wikipedia JSON.
    """
    sentences = []
    
    path = Path(path)
    
    if path.suffix == '.gz':
        opener = gzip.open
    else:
        opener = open
    
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            # Filter by length
            if len(line) < min_length or len(line) > max_length:
                continue
            
            # Skip lines that don't look like sentences
            if not line[0].isupper():
                continue
            if not line[-1] in '.!?':
                continue
            
            sentences.append(line)
            
            if max_sentences and len(sentences) >= max_sentences:
                break
    
    return sentences


def download_wikipedia_sample(output_dir: str = "data") -> str:
    """
    Download a sample of Wikipedia sentences.
    
    Uses the Simple Wikipedia dump for smaller size.
    """
    import urllib.request
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Simple English Wikipedia - smaller and cleaner
    url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
    
    output_path = output_dir / "simplewiki.xml.bz2"
    
    if not output_path.exists():
        print(f"Downloading Simple Wikipedia...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
    
    return str(output_path)
