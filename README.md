# imagineAI

**A language AI built on field dynamics, not pattern matching.**

imagineAI resolves tension fields to answers. A question is excitation. The answer is the only possible stable state.

## The Core Idea

Traditional AI: `Input → Neural Network → Probability Distribution → Sample Output`

imagineAI: `Question → Φ Field Excitation → σ-Minimization → Stable Answer`

The name comes from the **imaginary operator i₀** at the center of the ICHTB (Inverse Cartesian Heisenberg Tensor Box) from Intent Tensor Theory. The imaginary point is where recursion anchors—where nothing becomes something. When you ask a question, you excite the field. The answer is where the field collapses to equilibrium.

## Mathematical Foundation

From Intent Tensor Theory, we use four core operators:

| Operator | Symbol | Role in Language |
|----------|--------|------------------|
| Scalar Potential | Φ | Semantic embedding space |
| Gradient | ∇Φ | Direction toward meaning |
| Curl | ∇×F | Context memory / loops |
| Laplacian | ∇²Φ | Expansion/compression of meaning |
| Residue | σ | Semantic distance to resolution |
| Boundary | ρ_q | Constraint satisfaction |

**The Resolution Equation:**
```
While σ > threshold:
    Compute field operators on current state
    Find transformation that minimizes σ
    Apply transformation
    Update Φ field
```

When σ → 0, the field has stabilized. That stable state IS the answer.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/intent-tensor-theory/git_0.0_imagineAI.git
cd git_0.0_imagineAI

# Install dependencies
pip install -r requirements.txt

# Run the demo
python -m imagine_ai.demo
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      APEX (Output)                       │
│              Answer = argmin(σ) over field              │
├─────────────────────────────────────────────────────────┤
│                    FIELD DYNAMICS                        │
│   ∇Φ (gradient) | ∇×F (curl) | ∇²Φ (laplacian) | ρ_q   │
├─────────────────────────────────────────────────────────┤
│                    SUBSTRATE (Φ)                         │
│         Sentence Embeddings + Knowledge Base            │
│              + LLM Field Generator                       │
└─────────────────────────────────────────────────────────┘
```

## How It Differs From LLMs

| Aspect | Standard LLM | imagineAI |
|--------|--------------|-----------|
| Output selection | Token probability | σ-minimization |
| Quality metric | Perplexity | Semantic residue |
| Constraints | Implicit (training) | Explicit (ρ_q) |
| Explainability | Black box | Traceable σ path |
| Refinement | Single pass | Iterative collapse |

## Repository Structure

```
git_0.0_imagineAI/
├── imagine_ai/
│   ├── core/           # ITT operators
│   │   ├── phi_field.py
│   │   ├── operators.py
│   │   ├── sigma.py
│   │   ├── rho_q.py
│   │   └── resolver.py
│   ├── language/       # Language processing
│   │   ├── embeddings.py
│   │   ├── generator.py
│   │   └── knowledge.py
│   └── api/            # Conversation interface
│       └── chat.py
├── notebooks/          # Development notebooks
├── configs/            # Configuration files
└── docker/             # Deployment
```

## Theory

See [THEORY.md](THEORY.md) for the full mathematical derivation from Intent Tensor Theory.

## Links

- [Intent Tensor Theory](https://intent-tensor-theory.com)
- [ICHTB Coordinate System](https://intent-tensor-theory.com/coordinate-system/)
- [Astrosynthesis Book 3](https://intent-tensor-theory.github.io/git_0.0_-astrosynthesis/book3/)

## License

MIT

## Author

Armstrong Knight / Intent Tensor Theory Institute
