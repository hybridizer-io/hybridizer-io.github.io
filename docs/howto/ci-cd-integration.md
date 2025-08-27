---
id: ci-cd-integration
title: CI/CD Integration
---

- Pin toolchain versions (Hybridizer, CUDA toolkit, compilers) for reproducible builds
- Cache generated artifacts to speed up CI runs
- GPU runners vs CPU-only: use vector backend tests when GPU not available
- Smoke tests: numeric parity against CPU reference; performance budgets as gates
- Artifact publishing: package kernels/libraries + symbol maps for debugging
