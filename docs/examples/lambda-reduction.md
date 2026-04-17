---
id: lambda-reduction
title: "Lambda Reduction"
description: "Using delegates and lambdas with Hybridizer — capabilities and performance implications."
keywords: [Hybridizer, lambda, delegate, Func, reduction, performance]
---

# Lambda Reduction

> **Sample source**: [`6.Advanced/LambdaReduction`](https://github.com/hybridizer-io/hybridizer-basic-samples/tree/master/src/6.Advanced/LambdaReduction)

This example implements a reduction using **lambdas/delegates** instead of generics. It demonstrates how Hybridizer handles C# functional programming constructs — and the **performance tradeoffs** involved.

## Inner Reduction with Func

```csharp
[Kernel]
public static void InnerReduce(
    [Out] float[] result, [In] float[] input, int N,
    float neutral, Func<float, float, float> reductor)
{
    var cache = new SharedMemoryAllocator<float>().allocate(blockDim.x);
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int cacheIndex = threadIdx.x;

    float tmp = neutral;
    while (tid < N)
    {
        tmp = reductor(tmp, input[tid]);
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = tmp;
    CUDAIntrinsics.__syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] = reductor(cache[cacheIndex], cache[cacheIndex + i]);
        CUDAIntrinsics.__syncthreads();
        i >>= 1;
    }

    if (cacheIndex == 0)
        AtomicExpr.apply(ref result[0], cache[0], reductor);
}
```

## Entry Points with Lambdas

```csharp
[EntryPoint]
public static void ReduceAdd(float[] result, float[] input, int N)
{
    InnerReduce(result, input, N, 0.0f, (x, y) => x + y);
}

[EntryPoint]
public static void ReduceMax(float[] result, float[] input, int N)
{
    InnerReduce(result, input, N, float.MinValue, (x, y) => Math.Max(x, y));
}
```

This is elegant and concise, but there's a performance cost.

## Performance Comparison

| Approach | Bandwidth | % of Peak | Code Complexity |
|----------|-----------|-----------|-----------------|
| Plain code | 328 GB/s | 92% | High (copy-paste) |
| **Generics** | **328 GB/s** | **92%** | Medium |
| Lambda (optimized) | 255 GB/s | 72% | **Low** |
| Virtual functions | 154 GB/s | 43% | Medium |
| Lambda (naïve) | 59 GB/s | 17% | Low |

:::warning
**Lambda/delegate calls cannot be inlined** on GPU because the function pointer is not known at compile time. This introduces indirect call overhead on every reduction step.
:::

## Optimization: Cache the Lambda

A critical optimization — save the delegate to a local variable:

```csharp
[Kernel]
public void Reduce(int N, float[] a, float[] result)
{
    // Cache lambda in a register — this is the key optimization!
    Func<float, float, float> f = localReductor;

    var cache = new SharedMemoryAllocator<float>().allocate(blockDim.x);
    // ... use f instead of localReductor
}
```

This allows `nvcc` to optimize the indirect call. Without this trick, performance drops from 255 GB/s to 59 GB/s.

## When to Use What

| Need | Recommended Approach | Performance |
|------|---------------------|-------------|
| Maximum performance | Generics (`[HybridTemplateConcept]`) | 100% |
| Good performance + concise code | Lambda (with local variable trick) | ~72% |
| Expressiveness | Virtual functions | ~43% |
| Quick prototyping | Lambda (naïve) | ~17% |

## Next Steps

- [Generic Reduction](./generic-reduction) — Full-performance alternative
- [Generics, Virtuals, Delegates](../guide/generics-virtuals-delegates) — Detailed guide
- [Reduction](./reduction) — Simpler non-generic version
