#!/usr/bin/env python3
"""
demo.py - imagineAI v0.5 Demo

Anchored Filaments: Structure + Specificity

σ_total = σ_dtw + λ * σ_anchor

This combines:
- What v0.3 got right (word matching)
- What v0.4 got right (structure matching)
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def main():
    print("=" * 70)
    print("imagineAI v0.5 - Anchored Filaments")
    print("Structure (DTW) + Specificity (Anchors)")
    print("=" * 70)
    
    from imagine_ai.v05_anchored.solver import create_demo_solver
    
    print("\n[1] Initializing...")
    solver = create_demo_solver(verbose=False)
    
    print("\n[2] Ready! Ask questions (type 'quit' to exit)")
    print("    Type 'trace' for detailed analysis.\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'trace':
                print("Enter question for trace:")
                trace_q = input(">>> ").strip()
                if trace_q:
                    trace = solver.solve_with_trace(trace_q)
                    print("\n--- ANCHORED TRACE ---")
                    print(f"Question: {trace['question']}")
                    print(f"Anchors: {trace['anchors']}")
                    print(f"Gradients: {trace['num_gradients']}")
                    print(f"λ (anchor weight): {trace['lambda']}")
                    print(f"\nTop results:")
                    for r in trace['results'][:5]:
                        print(f"  [{r['rank']}] σ={r['sigma_total']:.3f} "
                              f"(dtw={r['sigma_dtw']:.3f}, anch={r['sigma_anchor']:.3f}) "
                              f"matches={r['anchor_matches']}")
                        print(f"      {r['text'][:55]}...")
                    print("--- END TRACE ---\n")
                continue
            
            # Solve
            result = solver.solve(question)
            
            if result.answer:
                print(f"\nimagineAI: {result.answer}")
                print(f"           [σ={result.sigma_total:.3f} "
                      f"(dtw={result.sigma_dtw:.3f} + anchor={result.sigma_anchor:.3f}), "
                      f"anchors={result.anchor_matches}/{len(result.anchors)}]")
            else:
                print(f"\nimagineAI: Could not find answer. [{result.status.value}]")
            
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
    print("imagineAI v0.5 - Anchored Filament Test Suite")
    print("=" * 70)
    
    from imagine_ai.v05_anchored.solver import create_demo_solver
    
    solver = create_demo_solver(verbose=False)
    
    # Test questions with expected keywords
    tests = [
        ("What is the capital of Mississippi?", "jackson"),
        ("What is the largest planet?", "jupiter"),
        ("How fast does light travel?", "speed"),
        ("What is the capital of Texas?", "austin"),
        ("What is the tallest mountain?", "everest"),
        ("Tell me about Mars", "red"),
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
            
            anchors_str = f"anchors={result.anchor_matches}/{len(result.anchors)}"
            
            if success:
                print(f"✓ PASS: σ={result.sigma_total:.3f} ({anchors_str})")
                print(f"  {result.answer[:60]}...")
                passed += 1
            else:
                print(f"✗ FAIL: Expected '{expected}' ({anchors_str})")
                print(f"  Got: {result.answer[:60]}...")
        else:
            print(f"✗ FAIL: No answer")
    
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{len(tests)} passed ({100*passed/len(tests):.0f}%)")
    
    # Compare to previous versions
    print(f"\nProgression:")
    print(f"  v0.3 (mean):      3/8 = 37%")
    print(f"  v0.4 (filament):  1/8 = 12%")
    print(f"  v0.5 (anchored):  {passed}/8 = {100*passed/8:.0f}%")
    
    return passed


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        passed = run_tests()
        sys.exit(0 if passed >= 6 else 1)
    else:
        main()
