---
id: overview
title: Platforms Overview
---

Hybridizer targets multiple backends from the same managed source:

- CUDA GPUs (NVIDIA): highest parallel throughput for SIMT workloads.
- OMP + CUDA: mixed CPU/GPU paths for portability.
- Vector backends: AVX/AVX512 (x86), NEON (ARM), POWER.

Choose a backend based on deployment hardware, perf goals, and ops constraints.
