# Phase 5 Design: Dynamic Style Engine (KL-Gated)

## Goal
Keep the LLM's coding style consistent with the specific module being edited.

## Design Decisions

### 1. Style Profile

Extract from sample of functions:
- **Naming conventions**: snake_case vs camelCase vs PascalCase
- **Type hints**: Yes/No + coverage percentage
- **Docstring style**: Google/NumPy/reST/None
- **Quote preference**: Single vs double
- **Line length**: Average line length
- **Functional vs OOP**: Function/method ratio
- **Import style**: sorted, grouped, etc.

### 2. KL Divergence Calculation

Compare local file style (P_local) vs global/repo style (P_global).

Use epsilon smoothing to avoid division by zero: ε = 1e-10

### 3. Threshold Gating

Only update style-augmentation in system prompt if D_KL > ε_threshold.

Recommended ε_threshold = 0.15

This prevents "jittery" prompt changes and saves compute.

## Implementation Notes

### Sampling

- Local: 5 functions from current file
- Global: 20 functions from repository

### Style Features

Categorical features (naming, docstring style, etc.) are converted to probability distributions.

KL divergence calculated for each feature, then averaged.

## Testing Strategy

Unit tests for:
1. Style profile extraction
2. KL divergence calculation
3. Threshold gating logic
4. Smoothing (avoiding division by zero)
