#!/usr/bin/env python3
"""
demo.py - imagineAI v0.3 Demo

GloVe + ICHTB + σ-minimization

The semantic geometry is real (GloVe).
The navigation is pure math (σ-minimization).
No neural network in the solver.
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def main():
    print("=" * 70)
    print("imagineAI v0.3 - GloVe + ICHTB + σ-Minimization")
    print("Real semantic geometry. Pure mathematical navigation.")
    print("=" * 70)
    
    from imagine_ai.v03_glove_ichtb.solver import create_demo_solver
    
    print("\n[1] Initializing...")
    solver = create_demo_solver(verbose=False)
    
    print("\n[2] Ready! Ask questions (type 'quit' to exit)")
    print("    The answer is found via σ-minimization in ICHTB space.\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'trace':
                print("Enter question for detailed trace:")
                trace_q = input(">>> ").strip()
                if trace_q:
                    trace = solver.solve_with_trace(trace_q)
                    print("\n--- TRACE ---")
                    print(f"Question: {trace['question']}")
                    print(f"Vector norm: {trace['question_vector_norm']:.4f}")
                    print(f"Zone magnitudes: {trace['zone_magnitudes']}")
                    print(f"Top results:")
                    for r in trace['results'][:5]:
                        print(f"  [{r['rank']}] σ={r['sigma']:.4f}: {r['text'][:50]}...")
                    print("--- END TRACE ---\n")
                continue
            
            # Solve
            result = solver.solve(question)
            
            if result.answer:
                print(f"\nimagineAI: {result.answer}")
                print(f"           [σ={result.sigma:.4f}, status={result.status.value}]")
            else:
                print(f"\nimagineAI: Could not find an answer. [{result.status.value}]")
            
            print()
            
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
    print("imagineAI v0.3 - Test Suite")
    print("=" * 70)
    
    from imagine_ai.v03_glove_ichtb.solver import create_demo_solver
    
    solver = create_demo_solver(verbose=False)
    
    # Test questions with expected keywords
    tests = [
        ("What is the capital of Mississippi?", "jackson"),
        ("What is the largest planet?", "jupiter"),
        ("How fast does light travel?", "speed"),
        ("What is the capital of Texas?", "austin"),
        ("What is the tallest mountain?", "everest"),
        ("Tell me about Mars", "red planet"),
        ("What planet has rings?", "saturn"),
        ("Who wrote Hamlet?", "shakespeare"),
    ]
    
    passed = 0
    for question, expected in tests:
        print(f"\nQ: {question}")
        
        result = solver.solve(question)
        
        if result.answer:
            answer_lower = result.answer.lower()
            success = expected.lower() in answer_lower
            
            if success:
                print(f"✓ PASS: σ={result.sigma:.4f}")
                print(f"  {result.answer[:60]}...")
                passed += 1
            else:
                print(f"✗ FAIL: Expected '{expected}' in answer")
                print(f"  Got: {result.answer[:60]}...")
        else:
            print(f"✗ FAIL: No answer")
    
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{len(tests)} passed")
    
    return passed >= len(tests) * 0.7  # 70% pass rate


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        main()
