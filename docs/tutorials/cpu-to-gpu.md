---
id: cpu-to-gpu
title: "4. From CPU to GPU"
description: "Transform existing sequential C# code into GPU-accelerated code, step by step."
keywords: [Hybridizer, tutorial, migration, CPU, GPU, Parallel.For, optimization]
sidebar_position: 4
---

# From CPU to GPU

This tutorial takes existing C# code and transforms it for GPU execution, step by step. At each step, we measure the improvement.

Inspired by the [Hybridizer "From Zero to Hero"](https://www.yourwebsite.com) approach.

## The Problem: Apply a Function to Every Element

We want to apply `f(x) = sin(x) * cos(x) + sqrt(abs(x))` to every element of a large array:

### Step 0: Sequential C#

```csharp
static void Compute(float[] input, float[] output, int N)
{
    for (int i = 0; i < N; i++)
    {
        float x = input[i];
        output[i] = (float)(Math.Sin(x) * Math.Cos(x) + Math.Sqrt(Math.Abs(x)));
    }
}
```

This runs on a single core. For 16 million elements, it takes ~**800 ms**.

### Step 1: Parallel.For (CPU Multi-Core)

The easiest win — use all CPU cores:

```csharp
static void Compute(float[] input, float[] output, int N)
{
    Parallel.For(0, N, i =>
    {
        float x = input[i];
        output[i] = (float)(Math.Sin(x) * Math.Cos(x) + Math.Sqrt(Math.Abs(x)));
    });
}
```

Time: ~**150 ms** on an 8-core CPU. That's a **5× speedup** with one line changed.

### Step 2: Add `[EntryPoint]`

Now, make it GPU-ready:

```csharp
[EntryPoint]
public static void Compute(
    [In]  float[] input,
    [Out] float[] output,
    int N)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x;
         i < N;
         i += blockDim.x * gridDim.x)
    {
        float x = input[i];
        output[i] = (float)(Math.Sin(x) * Math.Cos(x) + Math.Sqrt(Math.Abs(x)));
    }
}
```

Changes:
1. Added `[EntryPoint]`
2. Added `[In]` / `[Out]` for transfer optimization
3. Replaced the loop with a grid-stride loop

```csharp
// Launch
cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);
dynamic wrapper = HybRunner.Cuda()
    .SetDistrib(prop.multiProcessorCount * 16, 256);

wrapper.Compute(input, output, N);
cuda.DeviceSynchronize();
```

Time: ~**12 ms** (including transfers). That's **67× faster** than sequential!

### Step 3: Use Fast Math Intrinsics

The GPU has special function units (SFU) for math. Map to them:

```csharp
[IntrinsicFunction("sinf")]
public static float Sinf(float x) => (float)Math.Sin(x);

[IntrinsicFunction("cosf")]
public static float Cosf(float x) => (float)Math.Cos(x);

[IntrinsicFunction("sqrtf")]
public static float Sqrtf(float x) => (float)Math.Sqrt(x);

[IntrinsicFunction("fabsf")]
public static float Fabsf(float x) => Math.Abs(x);

[EntryPoint]
public static void Compute([In] float[] input, [Out] float[] output, int N)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x;
         i < N;
         i += blockDim.x * gridDim.x)
    {
        float x = input[i];
        output[i] = Sinf(x) * Cosf(x) + Sqrtf(Fabsf(x));
    }
}
```

Each `[IntrinsicFunction]` maps a C# method to a CUDA hardware instruction:
- The body is the **CPU fallback** (normal Math functions)
- On GPU, the fast hardware instruction is used instead

Time: ~**5 ms**. Even faster.

## Summary Table

| Step | Change | Time | Speedup |
|------|--------|------|---------|
| 0. Sequential | Baseline | 800 ms | 1× |
| 1. `Parallel.For` | One line | 150 ms | 5× |
| 2. `[EntryPoint]` + GPU | Grid-stride loop | 12 ms | 67× |
| 3. `[IntrinsicFunction]` | Fast GPU math | 5 ms | **160×** |

## Rules for Porting

### ✅ Do

- **Remove loop side effects**: no `i++` on pointers, no external state mutation in the loop body
- **Use `[In]`/`[Out]`**: saves transfer time
- **Use intrinsics for math**: `expf`, `logf`, `sinf`, `sqrtf`
- **Match iteration to thread**: each thread handles `i`, stride by total threads

### ❌ Don't

- **Don't allocate with `new` in kernels**: heap allocation is extremely slow on GPU. Use `StackArray<T>` instead
- **Don't use `string`, `List<T>`, `Dictionary`**: reference types aren't supported on GPU
- **Don't use `Console.WriteLine`** in kernels (use `printf` via `[IntrinsicFunction]` if needed for debug)
- **Don't forget `DeviceSynchronize`**: results aren't ready until sync

### StackArray for GPU-local Storage

If you need a temporary array inside a kernel:

```csharp
// ❌ BAD: heap allocation — very slow on GPU
var buffer = new float[64];

// ✅ GOOD: stack allocation — fast, uses registers/cache
var buffer = new StackArray<float>(64);
```

## Exercise

Take this CPU code and port it to GPU:

```csharp
static void Normalize(float[] data, int N)
{
    float sum = 0;
    for (int i = 0; i < N; i++)
        sum += data[i];
    for (int i = 0; i < N; i++)
        data[i] /= sum;
}
```

Hint: you'll need **two kernels** — one for the sum ([reduction](../examples/reduction)) and one for the division.

## Next

Let's apply these skills to a visual example: [Working with Images →](./working-with-images)
