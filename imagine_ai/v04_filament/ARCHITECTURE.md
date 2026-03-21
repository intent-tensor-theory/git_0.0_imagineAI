# imagineAI v0.4: Semantic Filaments + Gradient Flow

## The Breakthrough Insight

**A sentence is not a POINT in semantic space. It's a PATH.**

Averaging word vectors throws away structure. But meaning IS structure.
The transition from word to word carries the meaning.

---

## Semantic Filaments

A sentence traces a CURVE through GloVe 300D space:

```
"What is the capital of Mississippi?"

     what ──→ is ──→ the ──→ capital ──→ of ──→ Mississippi
       │       │       │         │        │           │
      v₁      v₂      v₃        v₄       v₅          v₆
       
This is a FILAMENT - a path through semantic space.
```

The GRADIENTS between consecutive words encode the FLOW of meaning:

```
∇₁ = v₂ - v₁  (what → is)
∇₂ = v₃ - v₂  (is → the)
∇₃ = v₄ - v₃  (the → capital)
∇₄ = v₅ - v₄  (capital → of)
∇₅ = v₆ - v₅  (of → Mississippi)
```

The sentence becomes a GRADIENT TENSOR: G = [∇₁, ∇₂, ∇₃, ∇₄, ∇₅]

---

## Why This Works

**Question**: "What is the capital of Mississippi?"
**Answer**: "Jackson is the capital of Mississippi"

Their gradient tensors SHARE STRUCTURE:

```
Q: what→is→the→capital→of→Mississippi
   [  ∇₁  |  ∇₂  |   ∇₃   |  ∇₄  |      ∇₅       ]

A: Jackson→is→the→capital→of→Mississippi
   [   ∇₁'  | ∇₂' |   ∇₃'  |  ∇₄' |      ∇₅'      ]
```

The gradients `capital→of→Mississippi` are IDENTICAL in both.
The gradients `is→the→capital` are similar.

The WRONG answer "Jupiter is the largest planet" has NO matching gradients.

---

## Dynamic Time Warping (DTW)

Two filaments may have different lengths. DTW aligns them optimally.

DTW is:
- Dynamic programming (not neural network)
- Finds optimal alignment between sequences
- Respects sequential structure
- Returns distance metric

This is NOT trained. It's pure math.

---

## ICHTB Zone Classification

Each gradient ∇ gets classified by which ZONE it activates:

| Zone | Gradient Type | Example |
|------|---------------|---------|
| Δ₁ Forward | Subject→Action | "cat→runs" |
| Δ₂ Memory | Reference→Referent | "the→cat" (recall) |
| Δ₃ Expansion | General→Specific | "animal→cat" |
| Δ₄ Compression | Specific→General | "cat→it" |
| Δ₅ Apex | Assertion | "is→true" |
| Δ₆ Core | Identity | "I→am" |

Each gradient projects to a 6D zone signature:
```
zone_sig = [Δ₁_activation, Δ₂_activation, ..., Δ₆_activation]
```

The sentence becomes a sequence of zone signatures.
We DTW on THOSE.

---

## Full Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Sentence                                                 │
│ "What is the capital of Mississippi?"                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: Tokenize + GloVe Lookup                               │
│ [what, is, the, capital, of, mississippi]                      │
│ → [v₁, v₂, v₃, v₄, v₅, v₆]  (each vᵢ is 300D)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 2: Gradient Tensor                                        │
│ ∇ᵢ = vᵢ₊₁ - vᵢ                                                 │
│ → [∇₁, ∇₂, ∇₃, ∇₄, ∇₅]  (each ∇ᵢ is 300D)                     │
│ This is the FILAMENT - the path derivative                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3: ICHTB Zone Projection                                  │
│ Each ∇ᵢ → 6D zone signature                                    │
│ → [z₁, z₂, z₃, z₄, z₅]  (each zᵢ is 6D)                       │
│ This captures WHAT KIND of transition each gradient is         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 4: σ via Dynamic Time Warping                            │
│                                                                  │
│ For each fact in knowledge base:                                │
│   σ = DTW_distance(question_filament, fact_filament)           │
│                                                                  │
│ Return fact with minimum σ                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## What's NOT Trained

| Component | Method | Training? |
|-----------|--------|-----------|
| Word vectors | GloVe | Pre-trained (inherited) |
| Tokenization | String split | None |
| Gradients | Vector subtraction | None |
| Zone projection | Learned matrix | **Optional** |
| DTW | Dynamic programming | None |
| σ selection | argmin | None |

We could make zone projection learnable, but start with random orthogonal.

---

## The Curl: Detecting Relations

The ITT curl ∇×F detects LOOPS in the filament.

When words form relational structures (subject-verb-object), 
the gradient path curves back, creating non-zero curl.

"Jackson is the capital of Mississippi"
```
Jackson ←────────────────────┐
    ↓                         │
   is → the → capital → of → Mississippi
```

The path from "Mississippi" relates back to "Jackson" conceptually.
This creates CURL in the gradient field.

High curl = high relational density = likely answer, not question.

---

## Implementation Libraries

We use:
- **NumPy**: Gradient computation
- **Gensim**: GloVe vectors
- **dtaidistance** or **tslearn**: DTW computation
- **scipy**: Fallback DTW

All deterministic. No neural networks in the solver.

---

## Expected Behavior

| Query | v0.3 (mean) | v0.4 (filament) |
|-------|-------------|-----------------|
| "capital of Mississippi" | ✓ (same words) | ✓ (gradient match) |
| "How fast is light?" | ✗ (avg wrong) | ✓ (speed→light gradient) |
| "Who wrote Hamlet?" | ✗ | ✓ (wrote→Hamlet gradient) |

The filament approach should find matches based on STRUCTURAL similarity,
not just word overlap.

---

## Files

```
imagine_ai/v04_filament/
├── filament.py    # Gradient tensor computation
├── zones.py       # ICHTB zone classification
├── dtw.py         # Dynamic Time Warping
├── solver.py      # Filament σ-minimization
└── demo.py        # Interactive demo
```
