---
id: architecture
title: Architecture Overview
description: End-to-end pipeline from IL/bytecode to native backends, hardware ecosystem, and parallelization concepts.
keywords: [Hybridizer, architecture, pipeline, MSIL, bytecode, CUDA, AVX, x86, GPU]
---

# Architecture Overview

The Hybridizer can generate source code optimized for a variety of hardware architectures. This allows developers to embrace, with a single version of the source code, execution platforms ranging from low-powered platforms such as ARM processors, to high-end GPUs, through conventional CPUs with ever-wider vector units.

![Execution Environments](../images/execution-environments.png)

## Supported Hardware Ecosystem

### Intel and AMD x86

These are the most commoditized processors. The instruction set has constantly expanded while maintaining binary compatibility with former generations. Modern processors offer AVX instructions, enabling SIMD on 4 double precision or 8 single precision registers. The number of cores has continuously increased over the past decade.

Making use of the full capacity of x86 hardware requires:
- **Multiple concurrent threads** to use all cores
- **Vector operations** to leverage AVX facilities
- **Cache-aware programming** to optimize memory access

#### Parallelization Challenges on x86

| Challenge | Solutions |
|-----------|-----------|
| **Multithreading** | Intel TBB, OpenMP, pthreads |
| **Vectorization** | Compiler pragmas (`#pragma simd`), AVX intrinsics |
| **Cache-awareness** | Memory prefetch, local variable usage |

### NVIDIA GPU

Since 2004, GPUs have surpassed conventional processors in raw compute power. In early 2007, NVIDIA introduced the CUDA development environment, making General Purpose GPU (GPGPU) computing commonplace. Today, many high-end supercomputers use GPUs to accelerate calculations.

![CUDA By Numbers](../images/cuda-by-numbers.png)

#### Parallelization Challenges on GPU

| Challenge | CUDA Concept |
|-----------|-------------|
| **Multithreading** | CUDA blocks, `blockIdx` |
| **Vectorization** | CUDA threads, `threadIdx` (32-wide warps) |
| **Cache-awareness** | Shared memory, L1 cache |

## Concept Mapping

The Hybridizer maps parallelization concepts across platforms:

| CUDA | OpenCL | Hybridizer Vector |
|------|--------|-------------------|
| block | work-group | thread (stack frame) |
| thread | work-item | vector entry (within a vector unit) |

This mapping delivers the best performance across platforms while allowing a single version of the source code.

## Compilation Pipeline

![Compilation Flow](../images/compilation-flow.png)

The compilation process involves several steps:

1. **Input generation**: Compile your code using standard tools (C# compiler). Output: .NET binary
2. **Build parameters**: Configuration controls how source code for a given flavor is generated
3. **Configuration file generation**: The Hybridizer generates a config file based on attributes and annotations
4. **Flavor code generation**: Source code is generated for the selected flavor

:::tip
Hybridizer Essentials wraps all these steps in its Visual Studio integration.
:::

## Next Steps

- [Input Formats](../guide/compilation-pipeline) — Learn about MSIL processing
- [Flavors](../platforms/overview) — Explore available output targets
