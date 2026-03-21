# imagineAI v0.5: Anchored Filaments

## The Evolution

| Version | Captures | Loses | Result |
|---------|----------|-------|--------|
| v0.3 | Specificity (word presence) | Structure (syntax) | 37% |
| v0.4 | Structure (gradient flow) | Specificity (entities) | 12% |
| **v0.5** | **BOTH** | — | ? |

## The Problem We're Solving

**v0.3 failure** (How fast does light travel?):
```
Q: "how fast does light travel"
   → average lands near motion/physics concepts
   → but "Jupiter is the largest" is closer by word average
```

**v0.4 failure** (What is the capital of Mississippi?):
```
Q: "what is the capital of mississippi"
   → gradient: what→is→the→capital→of→mississippi

A1: "Jackson is the capital of Mississippi"  
   → gradient: jackson→is→the→capital→of→mississippi

A2: "Montgomery is the capital of Alabama"
   → gradient: montgomery→is→the→capital→of→alabama

Problem: A1 and A2 have IDENTICAL gradient structure!
DTW can't distinguish them.
```

## The Solution: Anchored Filaments

**Anchor words** are the specific entities that must appear in the answer.

From "What is the capital of Mississippi?":
- Anchor: "mississippi" (the specific entity)
- Pattern: "capital of X" (the structure)

The answer must:
1. Match the STRUCTURE (gradient similarity via DTW)
2. Contain the ANCHOR (word presence)

## Combined σ Formula

$$\sigma = \sigma_{DTW} + \lambda \cdot \sigma_{anchor}$$

Where:
- $\sigma_{DTW}$ = normalized DTW distance (structure)
- $\sigma_{anchor}$ = penalty for missing anchor words
- $\lambda$ = balancing weight (default: 1.0)

$$\sigma_{anchor} = \frac{\text{missing anchors}}{\text{total anchors}}$$

## Anchor Extraction

**What makes a word an anchor?**

1. **Named entities**: Proper nouns, places, people
   - "Mississippi", "Shakespeare", "Jupiter"
   
2. **Specific concepts**: Non-function words that aren't question words
   - "capital", "largest", "speed", "wrote"
   
3. **NOT anchors**: Stop words, question words
   - "what", "is", "the", "a", "how", "does"

Simple heuristic:
- Remove stop words
- Remove question words (what, who, where, when, why, how)
- Keep nouns and specific terms

## Architecture

```
Question: "What is the capital of Mississippi?"
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: ANCHOR EXTRACTION                                    │
│                                                              │
│ Stop words: what, is, the, of                               │
│ Anchors: ["capital", "mississippi"]                         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: FILAMENT CREATION                                    │
│                                                              │
│ Gradient tensor: [what→is, is→the, the→capital, ...]      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: COMBINED σ FOR EACH CANDIDATE                       │
│                                                              │
│ For each fact:                                              │
│   σ_dtw = DTW_distance(q_gradients, fact_gradients)        │
│   σ_anchor = missing_anchor_ratio(anchors, fact_words)      │
│   σ_total = σ_dtw + λ * σ_anchor                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: RETURN MINIMUM σ                                     │
│                                                              │
│ "Jackson is the capital of Mississippi"                     │
│   σ_dtw ≈ 0.3 (similar structure)                          │
│   σ_anchor = 0.0 (both anchors present)                    │
│   σ_total ≈ 0.3 ✓ WINNER                                   │
│                                                              │
│ "Montgomery is the capital of Alabama"                      │
│   σ_dtw ≈ 0.3 (similar structure)                          │
│   σ_anchor = 0.5 (missing "mississippi")                   │
│   σ_total ≈ 0.8 ✗ REJECTED                                 │
└─────────────────────────────────────────────────────────────┘
```

## Expected Results

| Query | v0.4 (structure only) | v0.5 (structure + anchors) |
|-------|----------------------|---------------------------|
| capital of Mississippi | FAIL (got Alabama) | PASS (anchor filters) |
| How fast is light? | PASS | PASS (still works) |
| largest planet | FAIL (got ocean) | PASS? (anchor: planet) |
| Who wrote Hamlet? | FAIL | PASS? (anchors: wrote, hamlet) |

## ITT Connection

This is the **ρ_q (boundary condition)** concept from ITT:

> The question creates constraints that the answer must satisfy.
> These constraints are the "boundaries" within which the answer must lie.

- **Gradient DTW** = field dynamics (how meaning flows)
- **Anchors** = ρ_q (boundary constraints the answer must satisfy)
- **Combined σ** = total tension that must be minimized

The answer is the configuration that:
1. Flows correctly (low DTW σ)
2. Satisfies constraints (contains anchors)

## Files

```
imagine_ai/v05_anchored/
├── anchors.py    # Anchor extraction
├── combined.py   # Combined σ calculation  
├── solver.py     # Anchored filament solver
└── demo.py       # Interactive demo
```
