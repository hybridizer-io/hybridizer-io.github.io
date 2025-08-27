---
id: optimize-kernels
title: Optimize Kernels
---

- Choose grid/block sizes based on occupancy and memory patterns
- Use shared memory for data reuse; avoid bank conflicts
- Ensure coalesced global memory accesses
- Minimize divergence; restructure conditionals
- Use profiler-guided iteration (Nsight Systems/Compute)
