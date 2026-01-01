## 2026-01-01 - [Baseline Profiling]
**Learning:** `GoogleJules` simulation uses O(n^2) nested loops for interaction calculation.
**Insight:** For N=200, it takes ~0.07s. This will explode for larger N.
**Action:** Replace nested loops with NumPy broadcasting/vectorization.
