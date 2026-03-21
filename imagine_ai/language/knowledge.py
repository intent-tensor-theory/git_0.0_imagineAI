"""
knowledge.py - Knowledge Retrieval

Retrieves relevant knowledge to ground the resolver's answers.

Currently supports:
- Wikipedia (free, easy to scrape, vast information)
- Simple in-memory knowledge base

The knowledge provides context that helps the LLM generate
better candidates and helps the resolver check factual accuracy.
"""

import os
import re
from typing import List, Optional, Dict
from abc import ABC, abstractmethod


class KnowledgeRetriever(ABC):
    """Abstract base for knowledge retrieval"""
    
    @abstractmethod
    def retrieve(self, query: str, max_results: int = 3) -> List[str]:
        """Retrieve relevant knowledge for a query"""
        pass


class WikipediaRetriever(KnowledgeRetriever):
    """
    Retrieve knowledge from Wikipedia.
    
    Free, easy to access, vast information.
    We can improve later with better sources.
    """
    
    def __init__(self):
        try:
            import wikipedia
            self.wiki = wikipedia
            # Set language
            self.wiki.set_lang("en")
        except ImportError:
            raise ImportError("Install wikipedia: pip install wikipedia-api")
    
    def retrieve(self, query: str, max_results: int = 3) -> List[str]:
        """
        Retrieve Wikipedia summaries relevant to query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of summary texts
        """
        results = []
        
        try:
            # Search Wikipedia
            search_results = self.wiki.search(query, results=max_results)
            
            for title in search_results:
                try:
                    # Get page summary
                    page = self.wiki.page(title, auto_suggest=False)
                    summary = page.summary
                    
                    # Truncate to reasonable length
                    if len(summary) > 1000:
                        summary = summary[:1000] + "..."
                    
                    results.append(f"[Wikipedia: {title}]\n{summary}")
                    
                except self.wiki.exceptions.DisambiguationError as e:
                    # Try first option from disambiguation
                    if e.options:
                        try:
                            page = self.wiki.page(e.options[0], auto_suggest=False)
                            summary = page.summary[:1000]
                            results.append(f"[Wikipedia: {e.options[0]}]\n{summary}")
                        except:
                            pass
                except self.wiki.exceptions.PageError:
                    # Page doesn't exist, skip
                    pass
                except Exception as e:
                    print(f"Wikipedia error for '{title}': {e}")
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
        
        return results
    
    def get_page(self, title: str) -> Optional[str]:
        """
        Get full Wikipedia page content.
        
        Args:
            title: Exact page title
            
        Returns:
            Page content or None
        """
        try:
            page = self.wiki.page(title, auto_suggest=False)
            return page.content
        except:
            return None


class SimpleKnowledgeBase(KnowledgeRetriever):
    """
    Simple in-memory knowledge base.
    
    Good for testing and domain-specific knowledge.
    """
    
    def __init__(self):
        self.facts: Dict[str, str] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def add_fact(self, keywords: str, fact: str, category: str = "general"):
        """
        Add a fact to the knowledge base.
        
        Args:
            keywords: Keywords that should trigger this fact
            fact: The fact text
            category: Category for organization
        """
        self.facts[keywords.lower()] = fact
        
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(keywords.lower())
    
    def retrieve(self, query: str, max_results: int = 3) -> List[str]:
        """Search for relevant facts"""
        query_lower = query.lower()
        results = []
        
        for keywords, fact in self.facts.items():
            # Check if any keyword appears in query
            keyword_list = keywords.split()
            matches = sum(1 for k in keyword_list if k in query_lower)
            
            if matches > 0:
                results.append((matches, fact))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in results[:max_results]]
    
    def load_from_dict(self, data: Dict[str, str], category: str = "general"):
        """Load facts from a dictionary"""
        for keywords, fact in data.items():
            self.add_fact(keywords, fact, category)


class HybridRetriever(KnowledgeRetriever):
    """
    Combines multiple knowledge sources.
    
    Checks local knowledge base first, falls back to Wikipedia.
    """
    
    def __init__(self, use_wikipedia: bool = True):
        self.local_kb = SimpleKnowledgeBase()
        self.wikipedia = WikipediaRetriever() if use_wikipedia else None
    
    def add_fact(self, keywords: str, fact: str, category: str = "general"):
        """Add to local knowledge base"""
        self.local_kb.add_fact(keywords, fact, category)
    
    def retrieve(self, query: str, max_results: int = 3) -> List[str]:
        """
        Retrieve from local first, then Wikipedia.
        """
        results = []
        
        # Check local knowledge base
        local_results = self.local_kb.retrieve(query, max_results=max_results)
        results.extend(local_results)
        
        # If we have enough local results, return
        if len(results) >= max_results:
            return results[:max_results]
        
        # Fill remaining slots with Wikipedia
        if self.wikipedia:
            remaining = max_results - len(results)
            wiki_results = self.wikipedia.retrieve(query, max_results=remaining)
            results.extend(wiki_results)
        
        return results[:max_results]


def create_knowledge_retriever(
    retriever_type: str = "hybrid",
    use_wikipedia: bool = True
) -> KnowledgeRetriever:
    """
    Create a knowledge retriever.
    
    Args:
        retriever_type: "wikipedia", "simple", or "hybrid"
        use_wikipedia: Whether to include Wikipedia (for hybrid)
        
    Returns:
        KnowledgeRetriever instance
    """
    if retriever_type == "wikipedia":
        return WikipediaRetriever()
    elif retriever_type == "simple":
        return SimpleKnowledgeBase()
    elif retriever_type == "hybrid":
        return HybridRetriever(use_wikipedia=use_wikipedia)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


# =============================================================================
# Pre-built knowledge for common questions (for testing)
# =============================================================================

def create_test_knowledge() -> SimpleKnowledgeBase:
    """Create a knowledge base with common facts for testing"""
    kb = SimpleKnowledgeBase()
    
    # US State Capitals
    capitals = {
        "mississippi jackson": "Jackson is the capital city of Mississippi. It is the most populous city in the state.",
        "texas austin": "Austin is the capital of Texas. It is located in Central Texas.",
        "california sacramento": "Sacramento is the capital of California.",
        "new york albany": "Albany is the capital of New York State (not New York City).",
        "florida tallahassee": "Tallahassee is the capital of Florida.",
    }
    
    # Planets
    planets = {
        "largest planet jupiter": "Jupiter is the largest planet in our solar system. It is a gas giant.",
        "smallest planet mercury": "Mercury is the smallest planet in our solar system.",
        "red planet mars": "Mars is known as the Red Planet due to its reddish appearance.",
    }
    
    # General facts
    general = {
        "water boiling point": "Water boils at 100°C (212°F) at standard atmospheric pressure.",
        "speed of light": "The speed of light is approximately 299,792,458 meters per second.",
        "earth sun distance": "Earth is approximately 93 million miles (150 million km) from the Sun.",
    }
    
    for keywords, fact in capitals.items():
        kb.add_fact(keywords, fact, "geography")
    
    for keywords, fact in planets.items():
        kb.add_fact(keywords, fact, "astronomy")
    
    for keywords, fact in general.items():
        kb.add_fact(keywords, fact, "science")
    
    return kb
