---
id: what-is-hybridizer
title: What is Hybridizer
description: Overview of Hybridizer, inputs, outputs, supported platforms and developer workflow.
---

Hybridizer compiles managed IL (MSIL for .NET, Java bytecode) into high-performance native code targeting multiple backends such as NVIDIA CUDA and vector ISAs (AVX/AVX512/NEON/POWER).

Key ideas:

- __Single-source__: write idiomatic C# or Java; Hybridizer compiles MSIL/bytecode to native.
- __Multiple backends__: CUDA kernels, OpenMP+CUDA, and vector backends (AVX/AVX512/NEON/POWER).
- __Performance__: Leverages backend-specific optimizations (e.g., SIMT on GPU, SIMD on CPU).
- __Interoperability__: Generated code callable from your .NET/Java host.

See also: `overview/architecture` and `guide/compilation-pipeline`.
