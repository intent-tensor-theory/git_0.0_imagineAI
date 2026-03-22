"""
demo.py - Test the Emergence Solver

This tests v0.6 against the same questions as v0.5.
The difference: v0.5 was retrieval, v0.6 is emergence.

Usage:
    python -m imagine_ai.v06_emergence.demo
    python -m imagine_ai.v06_emergence.demo --test
    python -m imagine_ai.v06_emergence.demo --verbose
"""

import sys
import argparse
from pathlib import Path

# Test questions (same as v0.5)
TEST_QUESTIONS = [
    ("What is the capital of Mississippi?", "Jackson"),
    ("What is the capital of Alabama?", "Montgomery"),
    ("Which planet has rings?", "Saturn"),
    ("What is the largest planet?", "Jupiter"),
    ("What is the tallest mountain?", "Everest"),
    ("What is the longest river in Africa?", "Nile"),
    ("What freezes at zero degrees?", "Water"),
    ("What is the capital of France?", "Paris"),
]


def load_glove():
    """Load GloVe embeddings."""
    from gensim.models import KeyedVectors
    
    # Try standard paths
    paths = [
        Path.home() / "glove" / "glove.6B.300d.word2vec.txt",
        Path("/workspace/glove/glove.6B.300d.word2vec.txt"),
        Path("glove.6B.300d.word2vec.txt"),
    ]
    
    for path in paths:
        if path.exists():
            print(f"Loading GloVe from: {path}")
            return KeyedVectors.load_word2vec_format(str(path), binary=False)
    
    raise FileNotFoundError("GloVe not found. Expected at ~/glove/ or /workspace/glove/")


def run_interactive(solver):
    """Interactive mode."""
    print("\n" + "="*60)
    print("imagineAI v0.6 - Emergence Solver")
    print("Not retrieval. Field dynamics.")
    print("="*60)
    print("\nType your question (or 'quit' to exit):")
    
    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ('quit', 'exit', 'q'):
                break
            if not question:
                continue
            
            result = solver.solve(question)
            
            print(f"\n{'─'*40}")
            print(f"Answer: {result.answer}")
            print(f"S = {result.selection.S:.4f} ({result.selection.regime})")
            print(f"Locked: {result.locked}")
            print(f"Emerged: {result.emerged}")
            print(f"Confidence: {result.confidence:.4f}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_test(solver, verbose: bool = False):
    """Run test suite."""
    print("\n" + "="*60)
    print("imagineAI v0.6 - Emergence Test Suite")
    print("="*60)
    
    correct = 0
    total = len(TEST_QUESTIONS)
    
    for question, expected in TEST_QUESTIONS:
        result = solver.solve(question)
        
        # Check if answer contains expected keyword
        is_correct = False
        if result.answer:
            is_correct = expected.lower() in result.answer.lower()
        
        status = "✓" if is_correct else "✗"
        
        print(f"\n{status} Q: {question}")
        print(f"  Expected: {expected}")
        print(f"  Got: {result.answer}")
        print(f"  S={result.selection.S:.2f} | Locked={result.locked} | Emerged={result.emerged}")
        
        if is_correct:
            correct += 1
    
    accuracy = correct / total * 100
    print(f"\n{'='*60}")
    print(f"RESULT: {correct}/{total} = {accuracy:.0f}%")
    print(f"{'='*60}")
    
    if accuracy == 100:
        print("🎯 PERFECT! The math found all answers.")
    elif accuracy >= 87.5:  # Same as v0.5 (7/8)
        print("📊 Same as v0.5. Need to tune evolution parameters.")
    else:
        print("🔧 Lower than v0.5. Debugging needed.")
    
    return correct, total


def main():
    parser = argparse.ArgumentParser(description="imagineAI v0.6 Emergence Demo")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("Loading GloVe embeddings...")
    glove = load_glove()
    print(f"Loaded {len(glove)} vectors, dim={glove.vector_size}")
    
    # Create solver
    from .solver import create_demo_solver
    solver = create_demo_solver(glove, verbose=args.verbose)
    print(f"Substrate: {len(solver.facts)} facts")
    
    if args.test:
        run_test(solver, verbose=args.verbose)
    else:
        run_interactive(solver)


if __name__ == "__main__":
    main()
