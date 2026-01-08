---
id: terminology
title: Preferred Terminology
---

- Use “Hybridizer” to refer to the compiler toolchain; avoid generic “translator”.
- Refer to "managed IL (MSIL)" as inputs; "native backends" as outputs.
- Use “CUDA backend”, “vector backends (AVX/AVX512/NEON/POWER)”, “OMP+CUDA”.
- Use “kernel” for GPU functions; “vectorized function” for CPU SIMD.
- Use “data marshalling” for host↔backend data movement and transformation.
