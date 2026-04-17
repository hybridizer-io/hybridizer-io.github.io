---
id: sobel-filter
title: "Sobel Edge Detection"
description: "2D image processing with stencil operations on GPU."
keywords: [Hybridizer, Sobel, image processing, stencil, 2D]
---

# Sobel Edge Detection

> **Sample source**: [`2.Imaging/Sobel`](https://github.com/hybridizer-io/hybridizer-basic-samples/tree/master/src/2.Imaging/Sobel)

This example applies a Sobel edge-detection filter to an image — a classic stencil operation. It demonstrates 2D grid distribution and neighborhoodaccess patterns.

## The Sobel Operator

The Sobel operator computes image gradients in X and Y:

```
Gx kernel:        Gy kernel:
-1  0 +1          -1 -2 -1
-2  0 +2           0  0  0
-1  0 +1          +1 +2 +1
```

Final magnitude: `output = √(Gx² + Gy²)`

## Kernel Code

```csharp
[EntryPoint]
public static void ComputeSobel(
    [Out] byte[] outputPixel,
    [In]  byte[] inputPixel,
    int width, int height, int from, int to)
{
    for (int i = from + threadIdx.y + blockIdx.y * blockDim.y;
         i < to;
         i += blockDim.y * gridDim.y)
    {
        for (int j = threadIdx.x + blockIdx.x * blockDim.x;
             j < width;
             j += blockDim.x * gridDim.x)
        {
            int pixelId = i * width + j;
            int output = 0;

            if (i != 0 && j != 0 && i != height - 1 && j != width - 1)
            {
                byte topl = inputPixel[pixelId - width - 1];
                byte top  = inputPixel[pixelId - width];
                byte topr = inputPixel[pixelId - width + 1];
                byte l    = inputPixel[pixelId - 1];
                byte r    = inputPixel[pixelId + 1];
                byte botl = inputPixel[pixelId + width - 1];
                byte bot  = inputPixel[pixelId + width];
                byte botr = inputPixel[pixelId + width + 1];

                int sobelx = topl + 2*l + botl - topr - 2*r - botr;
                int sobely = topl + 2*top + topr - botl - 2*bot - botr;

                output = (int)Math.Sqrt(sobelx*sobelx + sobely*sobely);
                output = Math.Min(Math.Max(output, 0), 255);
            }

            outputPixel[pixelId] = (byte)output;
        }
    }
}
```

### Launching

```csharp
HybRunner runner = SatelliteLoader.Load()
    .SetDistrib(32, 32, 16, 16, 1, 0);
dynamic wrapper = runner.Wrap(new Program());

wrapper.ComputeSobel(outputPixels, inputPixels, width, height, 0, height);
```

## What Happens on GPU

Each thread processes one pixel, reading its 8 neighbors:

```
Thread (j, i) reads:
    [i-1,j-1] [i-1,j] [i-1,j+1]
    [i  ,j-1] [i  ,j] [i  ,j+1]
    [i+1,j-1] [i+1,j] [i+1,j+1]
```

Memory access is mostly coalesced along `j` (threadIdx.x), which is efficient for GPU.

:::tip
For even better performance on stencil operations, consider loading the neighborhood to shared memory first (see the [`Sobel_2D`](https://github.com/hybridizer-io/hybridizer-basic-samples/tree/master/src/2.Imaging/Sobel_2D) variant and the [From Zero to Hero](../howto/optimize-kernels) optimization guide).
:::

## Next Steps

- [Mandelbrot](./mandelbrot) — Another 2D computation
- [Optimize Kernels](../howto/optimize-kernels) — Shared memory optimization
- [CUDA Memory](../cuda/memory-and-profiling) — Profiling stencil access
