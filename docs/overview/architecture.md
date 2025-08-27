---
id: architecture
title: Architecture Overview
description: End-to-end pipeline from IL/bytecode to native backends, toolchain components, and artifacts.
keywords: [Hybridizer, architecture, pipeline, MSIL, bytecode, CUDA, AVX]
---

High-level pipeline:

1. Frontend ingests MSIL (C#) or Java bytecode.
2. Analyzer resolves symbols, attributes/annotations, and reachable graph.
3. Code generator emits C++/CUDA/vector code leveraging backend libraries (e.g., phivect).
4. Backend compilers (CUDA, clang/LLVM, etc.) produce native binaries/kernels.
5. Host interop binds generated artifacts back to .NET/Java.

Artifacts:

- Generated source layout
- Kernels, libraries, and metadata (line info, mapping)
- Interop/shims for invocation
