---
id: constant-memory
title: "Constant Memory"
description: "Using GPU constant memory for read-only coefficients."
keywords: [Hybridizer, constant memory, HybridConstant, stencil]
---

# Constant Memory

> **Sample source**: [`5.CUDA_runtime/ConstantMemory`](https://github.com/hybridizer-io/hybridizer-basic-samples/tree/master/src/5.CUDA_runtime/ConstantMemory)

This example shows how to place read-only data in CUDA **constant memory** — a cached, broadcast-optimized memory space ideal for coefficients and lookup tables.

## GPU Memory Hierarchy

| Memory | Scope | Speed | Size | Use Case |
|--------|-------|-------|------|----------|
| Registers | Thread | Fastest | ~255 per thread | Variables |
| Shared | Block | Very fast | 48-96 KB | Shared work |
| **Constant** | **Device** | **Cached, broadcast** | **64 KB** | **Coefficients** |
| Global | Device | Slow | GBs | Arrays |

:::tip
Constant memory is **broadcast** to all threads in a warp simultaneously. When all 32 threads read the same address, it's as fast as a register access.
:::

## Declaring Constant Memory

```csharp
[HybridConstant(Location = ConstantLocation.ConstantMemory)]
public static float[] data = [-2.0f, -1.0f, 0.0f, 1.0f, 2.0f];
```

This array is placed in CUDA `__constant__` memory at compile time.

## Stencil Kernel Using Constants

```csharp
[EntryPoint]
public static void Run([Out] float[] output, [In] float[] input, int N)
{
    for (int k = 2 + threadIdx.x + blockDim.x * blockIdx.x;
         k < N - 2;
         k += blockDim.x * gridDim.x)
    {
        float tmp = 0;
        for (int p = -2; p <= 2; ++p)
        {
            tmp += data[p + 2] * input[k];
        }
        output[k] = tmp;
    }
}
```

All threads read the same `data[p + 2]` values — perfect for constant memory broadcast.

## Launch

```csharp
HybRunner runner = SatelliteLoader.Load();
dynamic wrapped = runner.Wrap(new Program());
wrapped.Run(output, input, N);
```

No special configuration needed — the `[HybridConstant]` attribute handles it.

## When to Use Constant Memory

| ✅ Good for | ❌ Bad for |
|------------|-----------|
| Stencil coefficients | Large lookup tables (> 64 KB) |
| Physical constants | Per-thread unique data |
| Small lookup tables | Frequently updated data |
| Filter kernels | Sparse access patterns |

## Next Steps

- [Streams](./streams) — Async multi-stream execution
- [Data Marshalling](../guide/data-marshalling) — Memory transfer details
- [Attributes Reference](../reference/attributes-and-annotations) — All attributes
