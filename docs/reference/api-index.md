---
id: api-index
title: API Index
description: Quick reference index to all Hybridizer APIs and documentation sections.
keywords: [Hybridizer, API, reference, index]
---

# API Index

This index provides quick access to all Hybridizer APIs, attributes, and documentation sections.

## Attributes

| Attribute | Description | Reference |
|-----------|-------------|-----------|
| `[EntryPoint]` | Mark kernel entry point | [Attributes](./attributes-and-annotations#entrypoint) |
| `[Kernel]` | Mark device function | [Attributes](./attributes-and-annotations#kernel) |
| `[HybridTemplateConcept]` | Template interface | [Generics](../guide/generics-virtuals-delegates) |
| `[HybridRegisterTemplate]` | Template specialization | [Generics](../guide/generics-virtuals-delegates) |
| `[IntrinsicConstant]` | Map to constant | [Intrinsics](../guide/intrinsics-builtins) |
| `[IntrinsicFunction]` | Map to function | [Intrinsics](../guide/intrinsics-builtins) |

## Runtime Classes

### `HybRunner`

Main entry point for kernel invocation.

```csharp
// Create runner
dynamic wrapper = HybRunner.Cuda();

// Configure
wrapper.SetDistrib(gridSize, blockSize);
wrapper.SetStream(stream);

// Invoke
wrapper.MyKernel(args...);
```

See: [Invoke Generated Code](../guide/invoke-generated-code)

### `cuda` (CUDA Imports)

Low-level CUDA runtime bindings.

```csharp
cuda.Malloc(out ptr, size);
cuda.Memcpy(dst, src, size, direction);
cuda.DeviceSynchronize();
cuda.GetLastError();
```

### `CUDAIntrinsics`

Device-side intrinsic functions.

```csharp
CUDAIntrinsics.SyncThreads();
CUDAIntrinsics.SyncWarp(mask);
CUDAIntrinsics.AtomicAdd(ref value, delta);
```

## Built-in Variables

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx` | dim3 | Thread index in block |
| `blockIdx` | dim3 | Block index in grid |
| `blockDim` | dim3 | Block dimensions |
| `gridDim` | dim3 | Grid dimensions |

## Documentation Sections

### Getting Started
- [What is Hybridizer?](../overview/what-is-hybridizer)
- [Installation](../quickstart/install)
- [Hello GPU](../quickstart/hello-gpu)

### Concepts
- [Core Concepts](../guide/concepts)
- [Compilation Pipeline](../guide/compilation-pipeline)
- [Data Marshalling](../guide/data-marshalling)

### Platforms
- [CUDA Backend](../platforms/cuda)
- [OMP Backend](../platforms/omp-cuda)
- [Vector Backends (AVX/NEON)](../platforms/vector-avx-neon)

### Advanced
- [Generics & Virtuals](../guide/generics-virtuals-delegates)
- [Intrinsics](../guide/intrinsics-builtins)
- [Optimize Kernels](../howto/optimize-kernels)

### Reference
- [Attributes](./attributes-and-annotations)
- [CLI Options](./cli-options)
- [Glossary](./glossary)

## External Resources

| Resource | Description |
|----------|-------------|
| [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/) | CUDA reference |
| [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | CUDA concepts |
| [Nsight Compute](https://developer.nvidia.com/nsight-compute) | Profiling tool |

## Next Steps

- [Glossary](./glossary) — Terminology reference
- [Attributes Reference](./attributes-and-annotations) — Full attribute docs
- [CLI Options](./cli-options) — Command-line reference
