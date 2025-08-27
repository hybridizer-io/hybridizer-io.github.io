---
id: cli-options
title: CLI Options
---

Common CLI flags (illustrative; adjust to your toolchain):

- `--input <assembly>`: Input managed assembly (MSIL/bytecode)
- `--backend <cuda|omp-cuda|avx|avx512|neon|power>`: Target backend
- `--out <dir>`: Output directory for generated sources/artifacts
- `--line-info`: Emit line mapping for debugging
- `--opt <level>`: Optimization level
- `--defines K=V`: Preprocessor-style defines

Tip: Version lock CUDA toolkit and compilers in CI for reproducibility.
