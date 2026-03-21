"""
generator.py - Field Generator (LLM Integration)

The LLM acts as a "field generator" - it produces candidate responses
that the resolver then evaluates via σ-minimization.

This is NOT using the LLM as the "brain" - the ITT operators are the brain.
The LLM is just generating the field of possibilities.

Supports:
- HuggingFace Transformers (local)
- HuggingFace Inference API (cloud)
- Ollama (local)
"""

import os
from typing import Optional, List, Callable
from abc import ABC, abstractmethod


class ResponseGenerator(ABC):
    """Abstract base for response generators"""
    
    @abstractmethod
    def generate(
        self, 
        question: str, 
        context: str = "",
        max_tokens: int = 256
    ) -> str:
        """Generate a response to the question"""
        pass


class HuggingFaceGenerator(ResponseGenerator):
    """
    Generate responses using HuggingFace Transformers.
    
    For local GPU inference with models like Llama 3.2.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "auto",
        hf_token: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: "auto", "cuda", "cpu"
            hf_token: HuggingFace API token (for gated models)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Get token from env if not provided
            token = hf_token or os.environ.get("HF_TOKEN")
            
            print(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=token
            )
            
            # Determine device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            # Load model with appropriate settings
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=token,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=token,
                    torch_dtype=torch.float32
                )
                self.model.to(self.device)
            
            self.model.eval()
            print(f"Model loaded on {self.device}")
            
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    def generate(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 256
    ) -> str:
        import torch
        
        # Build prompt
        if context:
            prompt = f"""Context: {context}

Question: {question}

Answer:"""
        else:
            prompt = f"""Question: {question}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in full_output:
            answer = full_output.split("Answer:")[-1].strip()
        else:
            # Remove the prompt from output
            answer = full_output[len(prompt):].strip()
        
        return answer


class HuggingFaceAPIGenerator(ResponseGenerator):
    """
    Generate responses using HuggingFace Inference API.
    
    No local GPU needed - runs on HuggingFace's servers.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        hf_token: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model name
            hf_token: HuggingFace API token
        """
        self.model_name = model_name
        self.token = hf_token or os.environ.get("HF_TOKEN")
        
        if not self.token:
            raise ValueError("HF_TOKEN required for API access")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    def generate(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 256
    ) -> str:
        import requests
        
        # Build prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        
        return str(result)


class OllamaGenerator(ResponseGenerator):
    """
    Generate responses using Ollama (local).
    
    Requires Ollama installed: https://ollama.ai
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b", "mistral")
            base_url: Ollama API URL
        """
        self.model_name = model_name
        self.base_url = base_url
        
        # Check if Ollama is running
        try:
            import requests
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"Warning: Ollama may not be running at {base_url}")
        except:
            print(f"Warning: Cannot connect to Ollama at {base_url}")
    
    def generate(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 256
    ) -> str:
        import requests
        
        # Build prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.status_code}")
        
        return response.json().get("response", "").strip()


class SimpleGenerator(ResponseGenerator):
    """
    Simple context-based generator for testing.
    
    No LLM required - extracts answers from provided context.
    """
    
    def __init__(self, knowledge_base: dict = None):
        """
        Args:
            knowledge_base: Dict mapping keywords/questions to answers
        """
        self.knowledge_base = knowledge_base or {}
    
    def add_knowledge(self, keyword: str, answer: str):
        """Add a keyword -> answer mapping"""
        self.knowledge_base[keyword.lower()] = answer
    
    def generate(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 256
    ) -> str:
        question_lower = question.lower()
        
        # First: Check if context contains useful info (from Wikipedia/knowledge retrieval)
        # This is the primary source - context is populated by the knowledge retriever
        if context:
            # Look for bracketed source info like [Wikipedia: Jackson]
            # and extract the content after it
            lines = context.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('[') and not line.startswith('User:') and not line.startswith('Assistant:'):
                    # This is content, not metadata
                    # Check if it's relevant to the question
                    if any(word in line.lower() for word in question_lower.split() if len(word) > 3):
                        # Found relevant context - return first sentence
                        sentences = line.split('.')
                        if sentences:
                            return sentences[0].strip() + '.'
            
            # If we have context but couldn't extract, return first meaningful line
            for line in lines:
                if line.strip() and not line.startswith('[') and len(line) > 20:
                    sentences = line.split('.')
                    if sentences and len(sentences[0]) > 10:
                        return sentences[0].strip() + '.'
        
        # Second: Check local knowledge base
        for keyword, answer in self.knowledge_base.items():
            # Check if keyword words appear in question
            # Match if ANY significant keyword word appears (not ALL)
            keyword_words = keyword.lower().split()
            significant_words = [w for w in keyword_words if len(w) > 3]
            if significant_words:
                matches = sum(1 for word in significant_words if word in question_lower)
                # If at least half of significant words match, return the answer
                if matches >= len(significant_words) / 2:
                    return answer
        
        # Third: Check if any keyword appears in context
        if context:
            context_lower = context.lower()
            for keyword, answer in self.knowledge_base.items():
                if keyword in context_lower:
                    return answer
        
        return "I don't have information about that."


def get_generator(
    generator_type: str = "simple",
    model_name: Optional[str] = None,
    hf_token: Optional[str] = None
) -> ResponseGenerator:
    """
    Get a response generator.
    
    Args:
        generator_type: "huggingface", "huggingface-api", "ollama", "simple", or "auto"
        model_name: Model name (optional)
        hf_token: HuggingFace token (optional)
        
    Returns:
        ResponseGenerator instance
    """
    # Simple generator - no external dependencies
    if generator_type == "simple":
        return SimpleGenerator()
    
    if generator_type == "auto":
        # Try Ollama first (lightest), then fall back to simple
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                return OllamaGenerator(model_name or "llama3.2:3b")
        except:
            pass
        
        # Fall back to simple - don't try HuggingFace API
        print("No LLM available. Using simple keyword matcher.")
        return SimpleGenerator()
    
    elif generator_type == "huggingface":
        return HuggingFaceGenerator(
            model_name or "meta-llama/Llama-3.2-3B-Instruct",
            hf_token=hf_token
        )
    
    elif generator_type == "huggingface-api":
        return HuggingFaceAPIGenerator(
            model_name or "meta-llama/Llama-3.2-3B-Instruct",
            hf_token=hf_token
        )
    
    elif generator_type == "ollama":
        return OllamaGenerator(model_name or "llama3.2:3b")
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
