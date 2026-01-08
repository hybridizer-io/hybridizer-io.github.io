---
id: omp-cuda
title: OMP Backend
description: Using the OpenMP flavor for CPU multi-threading and as a debugging/testing environment.
keywords: [Hybridizer, OMP, OpenMP, CPU, multi-threading, debugging]
---

# OMP Backend

The OMP flavor is a plain C/C++ flavor enabling support for OpenMP features. While this flavor does not offer dramatic performance improvements over JIT-compiled code, it provides valuable capabilities for development and testing.

## Use Cases

### 1. Testing and Debugging

The OMP flavor is an excellent **testing environment** to disambiguate issues between:
- Input code core processing
- Flavor source-code generation

:::tip
If generated code provides accurate values for OMP but not for CUDA, the issue is likely that the code is not parallelization-safe (incorrect work distribution).
:::

### 2. Sequential Execution

OMP code, if compiled **without** any OpenMP library/compiler, provides **plain sequential execution** of the algorithm. This is useful for:
- Validating algorithmic correctness
- Debugging without parallel complexity
- Running on systems without OpenMP support

### 3. Optimizing Compiler Benefits

OMP code outputs plain C++ that can be used as input for an **optimizing compiler**, which might generate better machine code than what the JIT would provide.

## Example: Vector Addition

```csharp
[EntryPoint]
public void VectorAdd(int n, float[] a, float[] b, float[] c)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x)
    {
        c[i] = a[i] + b[i];
    }
}
```

When compiled to OMP flavor:

```cpp
// Work distribution maps to OpenMP threads
#pragma omp parallel for
for (int i = 0; i < n; i++)
{
    c[i] = a[i] + b[i];
}
```

## Concept Mapping

| CUDA Concept | OMP Equivalent |
|--------------|----------------|
| Grid | Total iteration space |
| Block | OpenMP thread |
| Thread | Loop iteration |

## When to Use OMP

| Scenario | Recommendation |
|----------|----------------|
| No GPU available | ✅ Use OMP |
| Algorithm validation | ✅ Use OMP first |
| Debugging race conditions | ✅ Run sequential OMP |
| Production with GPU | ❌ Use CUDA |
| Maximum CPU performance | ⚠️ Consider AVX |

## Build Configuration

To compile OMP flavor output:

```bash
# With OpenMP (parallel)
g++ -fopenmp -O3 -o program program.cpp

# Without OpenMP (sequential)
g++ -O3 -o program program.cpp
```

## Requirements

- GCC, Clang, or MSVC with OpenMP support
- No special hardware required

## Next Steps

- [CUDA Backend](./cuda) — For GPU acceleration
- [Vector Backends](./vector-avx-neon) — For CPU SIMD
- [Port Existing Code](../howto/port-existing-code) — Migration guide
