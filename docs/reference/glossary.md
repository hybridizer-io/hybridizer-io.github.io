---
id: glossary
title: Glossary
description: Definitions of key terms used in Hybridizer documentation.
keywords: [Hybridizer, glossary, terminology, definitions]
---

# Glossary

Definitions of key terms used throughout the Hybridizer documentation.

## A

### AVX (Advanced Vector Extensions)
SIMD instruction set for x86 processors enabling 256-bit (AVX/AVX2) or 512-bit (AVX512) vector operations.

## B

### Block
A group of threads in CUDA that can share memory and synchronize. Maps to work-group in OpenCL.

### Blittable
A data type with identical layout in managed and native memory, allowing direct memory copy without marshalling.

## C

### Coalescence
Memory access pattern where consecutive threads access consecutive memory addresses, enabling efficient GPU memory transactions.

### CUDA
Compute Unified Device Architecture — NVIDIA's parallel computing platform and programming model.

## D

### Device
The GPU or accelerator that executes kernels. Contrasts with "Host" (CPU).

### Device Function
A function executed on the device, callable from other device code. Marked with `[Kernel]` in Hybridizer.

## E

### Entry Point
A method marked for GPU execution, callable from host code. Marked with `[EntryPoint]` in Hybridizer.

## F

### Flavor
The target output format of the Hybridizer (CUDA, OMP, AVX, etc.). Each flavor generates code for a specific backend.

## G

### Global Memory
Main GPU memory accessible by all threads. High capacity but high latency.

### Grid
The collection of all blocks launched for a kernel execution.

### Grid-Stride Loop
A loop pattern where each thread processes multiple elements by striding through the data.

## H

### Host
The CPU that launches kernels and manages device memory.

### HybRunner
The runtime class for invoking Hybridizer-generated kernels.

## I

### IL (Intermediate Language)
The bytecode format compiled from high-level languages. MSIL for .NET.

### Intrinsic
A function or constant that maps directly to a hardware-specific operation.

## K

### Kernel
In Hybridizer, a device function marked with `[Kernel]`. In CUDA, synonymous with entry point.

## M

### Marshalling
The process of converting data between managed (.NET) and native (C/C++) representations.

### MSIL
Microsoft Intermediate Language — the bytecode format for .NET assemblies.

## N

### NEON
SIMD instruction set for ARM processors, supporting 128-bit vector operations.

## O

### Occupancy
The ratio of active warps to maximum warps per SM. Higher occupancy can hide memory latency.

### OMP (OpenMP)
A parallel programming standard for shared-memory multi-threading.

## P

### Phivect
Internal vector library used by Hybridizer for CPU vectorization backends.

### Pinned Memory
Page-locked host memory enabling faster GPU transfers and async copies.

## S

### Shared Memory
Fast on-chip memory shared by threads within a block. ~100× faster than global memory.

### SIMD
Single Instruction Multiple Data — executing the same operation on multiple data elements.

### SIMT
Single Instruction Multiple Thread — NVIDIA's execution model where threads in a warp execute together.

### SM (Streaming Multiprocessor)
The fundamental compute unit of an NVIDIA GPU. Each SM contains multiple CUDA cores.

### Stream
A sequence of operations that execute in order on the GPU. Different streams can execute concurrently.

## T

### Template Concept
An interface marked with `[HybridTemplateConcept]` for C++ template specialization.

### Thread
The smallest unit of parallel execution. In CUDA, threads within a warp execute in lockstep.

## W

### Warp
A group of 32 threads that execute the same instruction simultaneously on NVIDIA GPUs.

### Work Item / Work Group
OpenCL terminology for thread and block, respectively.

## Known Limitations

The following are not supported in Hybridizer kernel code:

| Category | Unsupported |
|----------|-------------|
| Memory | Heap allocation, `new` for classes |
| Types | `string`, `object`, dynamic |
| Control Flow | `try/catch/finally`, `lock` |
| Loops | `foreach` (use `for` instead) |
| Generics | Generic methods (types OK) |
| Recursion | Direct recursion (use interfaces) |

See [What is Hybridizer - Limitations](../overview/what-is-hybridizer#known-limitations) for details.

## Next Steps

- [API Index](./api-index) — Full API reference
- [Core Concepts](../guide/concepts) — Understanding key concepts
- [What is Hybridizer](../overview/what-is-hybridizer) — Project overview
