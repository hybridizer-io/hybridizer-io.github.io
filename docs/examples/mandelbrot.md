---
id: mandelbrot
title: "Mandelbrot Set"
description: "2D kernel with helper functions — computing a fractal on GPU from C#."
keywords: [Hybridizer, Mandelbrot, fractal, 2D, Kernel, imaging]
---

# Mandelbrot Set

> **Sample source**: [`1.Simple/Mandelbrot`](https://github.com/hybridizer-io/hybridizer-basic-samples/tree/master/src/1.Simple/Mandelbrot)

This example renders a 4096×4096 Mandelbrot fractal. It demonstrates:
- 2D grid distribution
- `[Kernel]` helper functions
- CPU vs GPU benchmarking
- Image output

## Helper Function

The `[Kernel]` attribute marks a **device function** — callable from the GPU but not an entry point:

```csharp
const int maxiter = 32;

[Kernel]
public static int IterCount(float cx, float cy)
{
    int result = 0;
    float x = 0.0f, y = 0.0f;
    float xx = 0.0f, yy = 0.0f;
    while (xx + yy <= 4.0f && result < maxiter)
    {
        xx = x * x;
        yy = y * y;
        float xtmp = xx - yy + cx;
        y = 2.0f * x * y + cy;
        x = xtmp;
        result++;
    }
    return result;
}
```

This function is compiled to a CUDA `__device__` function and **inlined** by the backend compiler.

## 2D Entry Point

The kernel iterates over a 2D image, splitting work across both dimensions:

```csharp
const int N = 4096;
const float fromX = -2.0f, fromY = -2.0f, size = 4.0f;
const float h = size / N;

[EntryPoint]
public static void Run(IntResidentArray light, int lineFrom, int lineTo)
{
    for (int line = lineFrom + threadIdx.y + blockDim.y * blockIdx.y;
         line < lineTo;
         line += gridDim.y * blockDim.y)
    {
        for (int j = threadIdx.x + blockIdx.x * blockDim.x;
             j < N;
             j += blockDim.x * gridDim.x)
        {
            float x = fromX + line * h;
            float y = fromY + j * h;
            light[line * N + j] = IterCount(x, y);
        }
    }
}
```

### Thread Distribution

The kernel uses a **2D grid of 2D blocks**:

```csharp
HybRunner runner = SatelliteLoader.Load()
    .SetDistrib(32, 32, 16, 16, 1, 0);
//              ├──────┤  ├──────┤
//              gridDim   blockDim
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `gridDim.x` | 32 | Blocks in X |
| `gridDim.y` | 32 | Blocks in Y |
| `blockDim.x` | 16 | Threads per block in X |
| `blockDim.y` | 16 | Threads per block in Y |

Total: 32×32×16×16 = **262,144 threads** covering a 4096×4096 image.

## CPU vs GPU Comparison

The sample benchmarks both implementations:

```csharp
// GPU version
wrapper.Run(light_cuda, 0, N);

// CPU version — uses Parallel.For as fallback
Parallel.For(0, N, (line) =>
{
    Run(light_net, line, line + 1);
});
```

:::info
The same C# code runs on both CPU and GPU. On GPU, `threadIdx`/`blockIdx` are real registers. On CPU, Hybridizer maps them to loop iterators.
:::

## `IntResidentArray`

`IntResidentArray` is a Hybridizer managed array type that:
- Allocates memory on both host and device
- Provides explicit `RefreshHost()` / `RefreshDevice()` for control

```csharp
IntResidentArray light = new(N * N);

// After GPU computation, bring results back
light.RefreshHost();

// Use as regular array
int value = light[i * N + j];
```

## Next Steps

- [Sobel Filter](./sobel-filter) — 2D image processing with stencils
- [Reduction](./reduction) — Shared memory and atomic operations
- [Hello World](./hello-world) — Simpler starting point
