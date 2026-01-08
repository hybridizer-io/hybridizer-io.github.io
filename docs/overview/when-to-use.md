---
id: when-to-use
title: When to Use Hybridizer
description: Use cases, suitability checklist, and limitations to set expectations.
keywords: [use cases, suitability, limitations, GPU, SIMD]
---

Great fit when:

- Data-parallel workloads (SIMD/SIMT), large arrays, linear algebra, image/signal processing.
- Hot paths in C# that dominate runtime.
- Need cross-platform performance without rewriting in CUDA/C++.

Considerations:

- Memory transfer cost GPU↔CPU.
- Algorithm parallelizability; control flow divergence.
- Interop with existing native libraries.
- Known limitations (see Reference → Glossary/Limitations).
