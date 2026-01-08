---
id: port-existing-code
title: Port Existing Code to Hybridizer
description: Step-by-step guide to migrating existing C# code to run on GPU with Hybridizer.
keywords: [Hybridizer, migration, port, GPU, optimize, parallel]
---

# Port Existing Code to Hybridizer

This guide walks you through the process of migrating existing C# code to run on GPU or other accelerators using Hybridizer.

## Overview

```mermaid
flowchart LR
    A[Identify Hot Paths] --> B[Check Compatibility]
    B --> C[Add Attributes]
    C --> D[Refactor if Needed]
    D --> E[Validate Results]
    E --> F[Optimize]
```

## Step 1: Identify Hot Paths

Use profiling to find code that:
- Consumes significant CPU time
- Has data-parallel structure (same operation on many elements)
- Works with large arrays or matrices

```csharp
// Before: CPU-bound loop
for (int i = 0; i < N; i++)
{
    result[i] = Math.Sin(data[i]) * Math.Cos(data[i]);
}
```

## Step 2: Check Compatibility

Review the [known limitations](../overview/what-is-hybridizer#known-limitations):

| Supported | Not Supported |
|-----------|---------------|
| Arrays of primitives | Heap allocation in kernel |
| Blittable structs | Strings |
| Math operations | Exceptions (partial) |
| Generics (with templates) | `foreach` loops |
| Virtual functions | Generic methods |

## Step 3: Add Attributes

Mark your method as an entry point:

```csharp
using Hybridizer.Runtime.CUDAImports;

public class MyProcessor
{
    [EntryPoint]
    public static void Process(double[] data, double[] result, int N)
    {
        for (int i = threadIdx.x + blockIdx.x * blockDim.x;
             i < N;
             i += blockDim.x * gridDim.x)
        {
            result[i] = Math.Sin(data[i]) * Math.Cos(data[i]);
        }
    }
}
```

Key changes:
1. Added `[EntryPoint]` attribute
2. Replaced sequential loop with grid-stride loop
3. Used `threadIdx` and `blockIdx` for indexing

## Step 4: Refactor Unsupported Patterns

### Replace `foreach` with `for`

```csharp
// Before (not supported)
foreach (var item in collection) { ... }

// After
for (int i = 0; i < collection.Length; i++) { ... }
```

### Extract Helper Methods

```csharp
// Mark helpers as [Kernel]
[Kernel]
public static double ComputeValue(double x)
{
    return Math.Sin(x) * Math.Cos(x);
}

[EntryPoint]
public static void Process(double[] data, double[] result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        result[i] = ComputeValue(data[i]);
}
```

### Handle Object Allocations

```csharp
// Before (not supported - heap allocation)
var temp = new MyClass();

// After - use structs passed as parameters
public struct MyStruct { public float Value; }

[EntryPoint]
public static void Process(MyStruct[] data, int N) { ... }
```

## Step 5: Validate Results

Always compare GPU results with a CPU reference:

```csharp
// CPU reference
double[] cpuResult = new double[N];
for (int i = 0; i < N; i++)
    cpuResult[i] = Math.Sin(data[i]) * Math.Cos(data[i]);

// GPU result
wrapper.Process(data, gpuResult, N);

// Compare
for (int i = 0; i < N; i++)
{
    double diff = Math.Abs(cpuResult[i] - gpuResult[i]);
    if (diff > 1e-10)
        Console.WriteLine($"Mismatch at {i}: {diff}");
}
```

:::warning
Floating-point results may differ slightly between CPU and GPU due to different instruction ordering and precision.
:::

## Step 6: Optimize

After validating correctness, optimize performance:

1. **Launch configuration**: Use enough threads
2. **Memory access**: Ensure coalescence
3. **Reduce transfers**: Keep data on GPU

See [Optimize Kernels](./optimize-kernels) for details.

## Common Porting Patterns

| Original Pattern | Hybridizer Equivalent |
|-----------------|----------------------|
| `for (i = 0; i < N; i++)` | Grid-stride loop |
| `Parallel.For(...)` | `[EntryPoint]` with threading |
| LINQ operations | Explicit loops |
| Object creation | Pre-allocated struct arrays |

## Next Steps

- [Optimize Kernels](./optimize-kernels) — Performance tuning
- [Core Concepts](../guide/concepts) — Work distribution
- [CUDA Threading](../cuda/basics-threading) — Threading model
