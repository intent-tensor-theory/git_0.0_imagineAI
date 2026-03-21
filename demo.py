#!/usr/bin/env python3
"""
imagineAI - A language AI built on field dynamics

Quick start:
    python demo.py              # Run demo
    python demo.py --chat       # Interactive chat
    python demo.py --test       # Run tests
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo():
    """Run the basic demo"""
    print("=" * 60)
    print("imagineAI Demo")
    print("A language AI built on field dynamics, not pattern matching")
    print("=" * 60)
    print()
    
    from imagine_ai.api.chat import ImagineAI
    
    # Initialize
    print("Initializing imagineAI...")
    ai = ImagineAI(
        embedding_backend="auto",
        generator_type="simple",  # Simple for demo, use "huggingface" with GPU
        use_wikipedia=True,
        verbose=True
    )
    print()
    
    # Test questions
    questions = [
        "What is the capital of Mississippi?",
        "What is the largest planet?",
        "What is the speed of light?"
    ]
    
    for q in questions:
        print(f"You: {q}")
        response = ai.chat(q)
        print(f"imagineAI: {response}")
        print()
    
    # Stats
    print("-" * 40)
    print(f"Session stats: {ai.get_stats()}")


def run_interactive():
    """Run interactive chat"""
    print("=" * 60)
    print("imagineAI Interactive Chat")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)
    print()
    
    from imagine_ai.api.chat import ImagineAI
    
    ai = ImagineAI(
        embedding_backend="auto",
        generator_type="auto",
        use_wikipedia=True,
        verbose=False
    )
    
    print("Ready! Ask me anything.\n")
    
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
                print("Conversation reset.\n")
                continue
            
            if user_input.lower() == "stats":
                print(f"Stats: {ai.get_stats()}\n")
                continue
            
            response = ai.chat(user_input)
            print(f"imagineAI: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def run_tests():
    """Run basic tests"""
    print("Running imagineAI tests...")
    print()
    
    # Test 1: Phi field
    print("Test 1: Φ Field (embeddings)")
    try:
        from imagine_ai.language.embeddings import get_embedding_backend
        from imagine_ai.core.phi_field import PhiField
        
        backend = get_embedding_backend("auto")
        phi = PhiField(backend)
        
        state1 = phi.embed("hello world")
        state2 = phi.embed("hi there")
        state3 = phi.embed("quantum physics")
        
        d12 = state1.distance_to(state2)
        d13 = state1.distance_to(state3)
        
        assert d12 < d13, "Similar concepts should be closer"
        print(f"  ✓ Φ field working (dim={phi.dimension})")
        print(f"    'hello world' <-> 'hi there': {d12:.3f}")
        print(f"    'hello world' <-> 'quantum physics': {d13:.3f}")
    except Exception as e:
        print(f"  ✗ Φ field failed: {e}")
    
    print()
    
    # Test 2: Operators
    print("Test 2: ITT Operators")
    try:
        from imagine_ai.core.operators import SemanticOperators, compute_sigma
        
        ops = SemanticOperators(phi)
        
        q = phi.embed("What is the capital?")
        a = phi.embed("The capital is Jackson")
        
        grad = ops.gradient(q, a)
        sigma = compute_sigma(q, a)
        
        print(f"  ✓ Gradient magnitude: {grad.magnitude:.3f}")
        print(f"  ✓ σ (residue): {sigma:.3f}")
    except Exception as e:
        print(f"  ✗ Operators failed: {e}")
    
    print()
    
    # Test 3: Knowledge
    print("Test 3: Knowledge Retrieval")
    try:
        from imagine_ai.language.knowledge import create_test_knowledge
        
        kb = create_test_knowledge()
        results = kb.retrieve("capital mississippi")
        
        assert len(results) > 0, "Should find capital info"
        print(f"  ✓ Found {len(results)} knowledge items")
        print(f"    First result: {results[0][:60]}...")
    except Exception as e:
        print(f"  ✗ Knowledge failed: {e}")
    
    print()
    print("Tests complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--chat":
            run_interactive()
        elif sys.argv[1] == "--test":
            run_tests()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python demo.py [--chat | --test]")
    else:
        run_demo()
