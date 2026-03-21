#!/usr/bin/env python3
"""
demo.py - imagineAI v0.2 Demo

This demonstrates the REAL imagineAI:
- No neural networks
- No LLM
- Pure ITT field dynamics

The answer EMERGES from the math.
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from imagine_ai.v02_field_dynamics import (
    ICHTBSpace, ICHTBProjector,
    FieldResolver, FieldResolverConfig,
    populate_space_from_text, create_demo_knowledge
)


def main():
    print("=" * 70)
    print("imagineAI v0.2 - Pure Field Dynamics")
    print("NO NEURAL NETWORKS. NO TRAINING. The math finds the answer.")
    print("=" * 70)
    
    # Load knowledge into 48D ICHTB space
    print("\n[1] Loading knowledge into 48D ICHTB space...")
    knowledge = create_demo_knowledge()
    space, projector = populate_space_from_text(knowledge)
    print(f"    Loaded {space.size()} facts as points in 48D space")
    
    # Create resolver
    print("\n[2] Creating field resolver...")
    config = FieldResolverConfig(
        max_iterations=50,
        sigma_threshold=0.5,
        beam_width=5,
        verbose=False  # Set True for debug output
    )
    resolver = FieldResolver(space, projector, config)
    print("    Resolver ready")
    
    # Interactive loop
    print("\n[3] Ready! Ask questions (type 'quit' to exit)")
    print("    The answer will be FOUND via σ-minimization, not retrieved.\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Resolve using field dynamics
            result = resolver.resolve(question)
            
            # Display result
            if result.answer:
                print(f"\nimagineAI: {result.answer.content}")
                print(f"           [σ={result.sigma:.3f}, "
                      f"iterations={result.iterations}, "
                      f"status={result.status.value}]")
            else:
                print(f"\nimagineAI: Could not resolve. [status={result.status.value}]")
            
            print()  # Blank line
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def run_tests():
    """Run test suite."""
    print("=" * 70)
    print("imagineAI v0.2 - Test Suite")
    print("=" * 70)
    
    # Load knowledge
    knowledge = create_demo_knowledge()
    space, projector = populate_space_from_text(knowledge)
    
    config = FieldResolverConfig(
        max_iterations=50,
        sigma_threshold=0.5,
        verbose=False
    )
    resolver = FieldResolver(space, projector, config)
    
    # Test questions
    tests = [
        ("What is the capital of Mississippi?", "Jackson"),
        ("What is the largest planet?", "Jupiter"),
        ("How fast does light travel?", "speed of light"),
        ("What is the capital of Texas?", "Austin"),
        ("Tell me about Mars", "Red Planet"),
    ]
    
    passed = 0
    for question, expected_keyword in tests:
        print(f"\nQ: {question}")
        
        result = resolver.resolve(question)
        
        if result.answer:
            answer = result.answer.content.lower()
            success = expected_keyword.lower() in answer
            
            if success:
                print(f"✓ PASS: {result.answer.content[:60]}...")
                print(f"  σ={result.sigma:.3f}, iter={result.iterations}")
                passed += 1
            else:
                print(f"✗ FAIL: Expected '{expected_keyword}' in answer")
                print(f"  Got: {result.answer.content[:60]}...")
        else:
            print(f"✗ FAIL: No answer resolved")
    
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{len(tests)} passed")
    
    return passed == len(tests)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        main()
