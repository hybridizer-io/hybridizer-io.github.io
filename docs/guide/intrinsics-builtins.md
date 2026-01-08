---
id: intrinsics-builtins
title: Intrinsics & Builtins
description: How Hybridizer maps math and hardware intrinsics to backend equivalents.
keywords: [Hybridizer, intrinsics, builtins, CUDA, AVX, math, SIMD]
---

# Intrinsics & Builtins

The Hybridizer provides a mechanism to map methods and properties to backend-specific intrinsics. This allows you to write portable code that automatically uses optimized implementations on each platform.

## How Intrinsics Work

Any method or property getter may be marked with an Intrinsic attribute:
- `IntrinsicConstant`: Maps to a backend constant
- `IntrinsicFunction`: Maps to a backend function

### Example: CUDA Intrinsics

```csharp
public static class CUDAIntrinsics
{
    // Map to CUDA thread index
    [IntrinsicConstant("threadIdx.x")]
    public static int ThreadIdxX { get; }

    [IntrinsicConstant("blockIdx.x")]
    public static int BlockIdxX { get; }

    // Map to CUDA synchronization
    [IntrinsicFunction("__syncthreads")]
    public static void SyncThreads() { }

    // Map to CUDA math
    [IntrinsicFunction("__expf")]
    public static float FastExp(float x) { return 0; }
}
```

## Built-in Intrinsics

The Hybridizer ships with built-in files that define common intrinsics:

| Category | Examples |
|----------|----------|
| **Work Distribution** | `threadIdx`, `blockIdx`, `blockDim`, `gridDim` |
| **Synchronization** | `__syncthreads`, `__threadfence` |
| **Math (Fast)** | `__expf`, `__logf`, `__sinf`, `__cosf` |
| **Math (IEEE)** | `exp`, `log`, `sin`, `cos` |
| **Atomic** | `atomicAdd`, `atomicCAS`, `atomicExch` |
| **Shuffle** | `__shfl_sync`, `__shfl_down_sync` |

## Platform-Specific Availability

| Intrinsic | CUDA | OMP | AVX |
|-----------|------|-----|-----|
| `threadIdx.x` | ✅ Native | ✅ Mapped | ✅ Vector lane |
| `__syncthreads` | ✅ Native | ❌ No-op | ❌ No-op |
| `__shfl_sync` | ✅ Native | ❌ N/A | ❌ N/A |
| `atomicAdd` | ✅ Native | ✅ OpenMP atomic | ✅ Lock-based |
| `__expf` | ✅ Fast GPU | ⚠️ Standard `expf` | ⚠️ Standard `expf` |

:::warning
CUDA-specific intrinsics like shuffle instructions have no equivalent on CPU targets, so they should be avoided in portable code.
:::

## Writing Portable Code

### Pattern: Conditional Compilation

```csharp
[Kernel]
public float Reduce(float[] data, int n)
{
    float sum = 0;
    
    // This loop works on all platforms
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x)
    {
        sum += data[i];
    }
    
    return sum;
}
```

### Pattern: Platform-Specific Paths

For advanced optimizations, you can use conditional logic based on runtime detection, though the Hybridizer typically handles this at code generation time.

## Math Function Mapping

| C# / .NET | CUDA | AVX/OMP |
|-----------|------|---------|
| `Math.Sin(x)` | `sin(x)` | `sin(x)` |
| `Math.Exp(x)` | `exp(x)` | `exp(x)` |
| `Math.Sqrt(x)` | `sqrt(x)` | `sqrt(x)` |
| `MathF.Sin(x)` | `sinf(x)` | `sinf(x)` |

:::tip
For maximum performance on CUDA, consider using fast math intrinsics like `__expf` when accuracy requirements allow.
:::

## Custom Intrinsics

You can define your own intrinsics to map to library functions:

```csharp
public static class MyLibraryIntrinsics
{
    [IntrinsicFunction("cuBLAS_dgemm")]
    public static void MatrixMultiply(
        double[] A, double[] B, double[] C,
        int M, int N, int K) 
    { 
        // Fallback implementation for testing
    }
}
```

## Next Steps

- [Data Marshalling](./data-marshalling) — Passing data to kernels
- [CUDA Backend](../platforms/cuda) — CUDA-specific features
- [Vector Backends](../platforms/vector-avx-neon) — AVX intrinsics mapping
