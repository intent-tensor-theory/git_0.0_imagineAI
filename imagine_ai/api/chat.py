"""
chat.py - Conversational Interface for imagineAI

This is the main entry point for talking to imagineAI.
It ties together all the components:
- Φ field (embeddings)
- LLM (field generator)  
- Knowledge retrieval
- ITT operators
- σ-minimization resolver

Usage:
    from imagine_ai.api.chat import ImagineAI
    
    ai = ImagineAI()
    response = ai.chat("What is the capital of Mississippi?")
    print(response)
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """A single message in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ImagineAI:
    """
    Main conversational interface for imagineAI.
    
    This is what you talk to. Ask it anything.
    
    Under the hood:
    1. Your question excites the Φ field
    2. LLM generates a candidate response
    3. ITT operators evaluate the candidate
    4. σ-minimization finds the stable answer
    5. That's what you get back
    """
    
    def __init__(
        self,
        embedding_backend: str = "auto",
        generator_type: str = "auto",
        use_wikipedia: bool = True,
        hf_token: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize imagineAI.
        
        Args:
            embedding_backend: "sentence-transformers", "glove", or "auto"
            generator_type: "huggingface", "ollama", "simple", or "auto"
            use_wikipedia: Whether to use Wikipedia for knowledge
            hf_token: HuggingFace API token
            verbose: Print debug info
        """
        self.verbose = verbose
        self.conversation: List[ChatMessage] = []
        
        # Get HuggingFace token
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        self._log("Initializing imagineAI...")
        
        # Initialize components
        self._init_phi_field(embedding_backend)
        self._init_generator(generator_type)
        self._init_knowledge(use_wikipedia)
        self._init_resolver()
        
        self._log("imagineAI ready.")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[imagineAI] {msg}")
    
    def _init_phi_field(self, backend: str):
        """Initialize the Φ field (embedding substrate)"""
        self._log(f"Initializing Φ field with {backend} backend...")
        
        from ..language.embeddings import create_phi_field
        
        try:
            self.phi_field = create_phi_field(backend_type=backend)
            self._log(f"Φ field ready (dim={self.phi_field.dimension})")
        except Exception as e:
            self._log(f"Φ field init failed: {e}")
            self.phi_field = None
    
    def _init_generator(self, generator_type: str):
        """Initialize the response generator (LLM)"""
        self._log(f"Initializing generator ({generator_type})...")
        
        from ..language.generator import get_generator
        
        try:
            self.generator = get_generator(
                generator_type=generator_type,
                hf_token=self.hf_token
            )
            self._log("Generator ready")
        except Exception as e:
            self._log(f"Generator init failed: {e}")
            # Fall back to simple generator
            from ..language.generator import SimpleGenerator
            self.generator = SimpleGenerator()
    
    def _init_knowledge(self, use_wikipedia: bool):
        """Initialize knowledge retrieval"""
        self._log("Initializing knowledge retrieval...")
        
        from ..language.knowledge import create_knowledge_retriever, create_test_knowledge
        
        try:
            self.knowledge = create_knowledge_retriever(
                retriever_type="hybrid",
                use_wikipedia=use_wikipedia
            )
            
            # Add test knowledge
            test_kb = create_test_knowledge()
            for keywords, fact in test_kb.facts.items():
                self.knowledge.add_fact(keywords, fact)
            
            self._log("Knowledge retrieval ready")
        except Exception as e:
            self._log(f"Knowledge init failed: {e}")
            self.knowledge = None
    
    def _init_resolver(self):
        """Initialize the ITT resolver"""
        self._log("Initializing resolver...")
        
        from ..core.resolver import ImagineAIResolver
        
        if self.phi_field is None:
            self._log("No Φ field - resolver will be limited")
            self.resolver = None
            return
        
        # Create generator wrapper
        def generate_fn(question: str, context: str) -> str:
            return self.generator.generate(question, context)
        
        # Create knowledge wrapper
        def knowledge_fn(query: str) -> List[str]:
            if self.knowledge:
                return self.knowledge.retrieve(query)
            return []
        
        self.resolver = ImagineAIResolver(
            phi_field=self.phi_field,
            response_generator=generate_fn,
            knowledge_retriever=knowledge_fn,
            max_iterations=5,
            sigma_threshold=0.2,
            verbose=self.verbose
        )
        
        self._log("Resolver ready")
    
    def chat(self, message: str) -> str:
        """
        Chat with imagineAI.
        
        Args:
            message: Your message/question
            
        Returns:
            Response string
        """
        # Record user message
        self.conversation.append(ChatMessage(role="user", content=message))
        
        # Get conversation context
        context = self._get_conversation_context()
        
        # Resolve
        if self.resolver:
            result = self.resolver.resolve(message, additional_context=context)
            response = result.answer or "I could not resolve an answer."
            
            # Add metadata
            metadata = {
                "sigma": result.sigma,
                "iterations": result.iterations,
                "status": result.status.value
            }
        else:
            # Fallback to direct generation
            response = self.generator.generate(message, "\n".join(context))
            metadata = {"fallback": True}
        
        # Record response
        self.conversation.append(ChatMessage(
            role="assistant",
            content=response,
            metadata=metadata
        ))
        
        return response
    
    def _get_conversation_context(self, max_turns: int = 5) -> List[str]:
        """Get recent conversation as context"""
        context = []
        recent = self.conversation[-max_turns*2:]  # Last N exchanges
        
        for msg in recent:
            if msg.role == "user":
                context.append(f"User: {msg.content}")
            else:
                context.append(f"Assistant: {msg.content}")
        
        return context
    
    def reset(self):
        """Reset conversation history"""
        self.conversation = []
        if self.resolver:
            self.resolver.reset_conversation()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        stats = {
            "turns": len(self.conversation) // 2,
            "has_phi_field": self.phi_field is not None,
            "has_resolver": self.resolver is not None,
        }
        
        if self.resolver and self.resolver.conversation:
            stats["conversation_curl"] = self.resolver.get_conversation_curl()
        
        return stats


def demo():
    """Quick demo of imagineAI"""
    print("=" * 60)
    print("imagineAI Demo")
    print("=" * 60)
    print()
    
    # Initialize with verbose mode
    ai = ImagineAI(verbose=True)
    
    # Test questions
    questions = [
        "What is the capital of Mississippi?",
        "Tell me about Jupiter.",
        "What is the speed of light?"
    ]
    
    for q in questions:
        print(f"\nYou: {q}")
        response = ai.chat(q)
        print(f"imagineAI: {response}")
    
    # Show stats
    print(f"\n{'-'*40}")
    print(f"Session stats: {ai.get_stats()}")


def interactive():
    """Interactive chat session"""
    print("=" * 60)
    print("imagineAI Interactive")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)
    print()
    
    ai = ImagineAI(verbose=False)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            if user_input.lower() == "reset":
                ai.reset()
                print("Conversation reset.")
                continue
            
            if user_input.lower() == "stats":
                print(f"Stats: {ai.get_stats()}")
                continue
            
            response = ai.chat(user_input)
            print(f"imagineAI: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive()
    else:
        demo()
