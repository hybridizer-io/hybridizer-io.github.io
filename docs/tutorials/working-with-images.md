---
id: working-with-images
title: "5. Working with Images"
description: "2D kernels and stencil operations — apply a Sobel edge detector on GPU."
keywords: [Hybridizer, tutorial, image processing, Sobel, 2D, stencil]
sidebar_position: 5
---

# Working with Images

Images are naturally 2D — and GPUs are built for 2D parallelism. In this tutorial, you'll apply a **Sobel edge detection** filter to an image.

> **Full sample**: [`2.Imaging/Sobel`](https://github.com/hybridizer-io/hybridizer-basic-samples/tree/master/src/2.Imaging/Sobel)

## What You'll Learn

- 2D thread distribution: `threadIdx.x/y`, `blockIdx.x/y`
- Stencil operations (reading neighbor pixels)
- Working with images via ImageSharp
- 2D `SetDistrib` configuration

## The Sobel Operator

The Sobel filter detects edges by computing gradients in X and Y:

```
Gradient X:          Gradient Y:
-1   0  +1           -1  -2  -1
-2   0  +2            0   0   0
-1   0  +1           +1  +2  +1
```

For each pixel, we read its 8 neighbors, apply both kernels, and compute:

```
edge_strength = √(Gx² + Gy²)
```

## Step 1: Load the Image

```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// Add NuGet: SixLabors.ImageSharp
var image = Image.Load<Rgba32>("photo.png");
int width = image.Width;
int height = image.Height;

// Convert to grayscale byte array
byte[] pixels = new byte[width * height];
for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
        var p = image[j, i];
        pixels[i * width + j] = (byte)(p.R * 0.2126 + p.G * 0.7152 + p.B * 0.0722);
    }
```

## Step 2: The 2D Kernel

```csharp
[EntryPoint]
public static void ComputeSobel(
    [Out] byte[] output,
    [In]  byte[] input,
    int width, int height)
{
    // Outer loop: rows (Y dimension)
    for (int i = threadIdx.y + blockIdx.y * blockDim.y;
         i < height;
         i += blockDim.y * gridDim.y)
    {
        // Inner loop: columns (X dimension)
        for (int j = threadIdx.x + blockIdx.x * blockDim.x;
             j < width;
             j += blockDim.x * gridDim.x)
        {
            // Skip border pixels (no neighbors)
            if (i == 0 || j == 0 || i == height - 1 || j == width - 1)
            {
                output[i * width + j] = 0;
                continue;
            }

            int pos = i * width + j;

            // Read 8 neighbors
            byte tl = input[pos - width - 1];  // top-left
            byte t  = input[pos - width];      // top
            byte tr = input[pos - width + 1];  // top-right
            byte l  = input[pos - 1];          // left
            byte r  = input[pos + 1];          // right
            byte bl = input[pos + width - 1];  // bottom-left
            byte b  = input[pos + width];      // bottom
            byte br = input[pos + width + 1];  // bottom-right

            // Apply Sobel kernels
            int gx = tl + 2*l + bl - tr - 2*r - br;
            int gy = tl + 2*t + tr - bl - 2*b - br;

            // Magnitude
            int mag = (int)Math.Sqrt(gx*gx + gy*gy);
            output[pos] = (byte)Math.Min(mag, 255);
        }
    }
}
```

### Key Difference: 2D Loops

Instead of one loop over `i`, we have **two nested loops** — one for rows (`threadIdx.y`) and one for columns (`threadIdx.x`).

```
Thread (threadIdx.x=3, threadIdx.y=5) in Block (blockIdx.x=1, blockIdx.y=2):
  j = 3 + 1 * 16 = 19  (column)
  i = 5 + 2 * 16 = 37  (row)
  → processes pixel at position (row=37, col=19)
```

## Step 3: Launch with 2D Grid

```csharp
// 2D grid and 2D blocks
dynamic wrapper = HybRunner.Cuda()
    .SetDistrib(32, 32, 16, 16, 1, 0);
//              ├──────┤  ├──────┤
//              gridDim   blockDim
//              (blocks)  (threads/block)

byte[] result = new byte[width * height];
wrapper.ComputeSobel(result, pixels, width, height);
cuda.DeviceSynchronize();
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Grid X | 32 | 32 blocks horizontally |
| Grid Y | 32 | 32 blocks vertically |
| Block X | 16 | 16 threads per block (columns) |
| Block Y | 16 | 16 threads per block (rows) |

Total: 32 × 32 × 16 × 16 = **262,144 threads** covering the image.

## Step 4: Save the Result

```csharp
var resultImage = new Image<Rgba32>(width, height);
for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
        byte v = result[i * width + j];
        resultImage[j, i] = new Rgba32(v, v, v, 255);
    }

resultImage.Save("edges.png");
Console.WriteLine("Saved edges.png");
```

## Why GPUs Excel at Image Processing

| Aspect | CPU | GPU |
|--------|-----|-----|
| Threads | 8-16 | 262,144 |
| Each pixel | Processed sequentially | Processed in parallel |
| Neighbor access | Cache-friendly | **Coalesced** (consecutive threads read consecutive bytes) |

Memory access pattern for the inner loop (`j` mapped to `threadIdx.x`):

```
Thread 0 reads pixel[i*W + 0]
Thread 1 reads pixel[i*W + 1]
Thread 2 reads pixel[i*W + 2]
...
```

These are **consecutive memory addresses** — the GPU loads them in a single transaction. This is called **coalesced access** and is critical for performance.

## Going Further: Shared Memory

For even better performance, load the pixel neighborhood into shared memory first. See:
- The [Sobel example](../examples/sobel-filter) for the complete code
- [From Zero to Hero](../howto/optimize-kernels) for advanced optimizations

## Exercise

Modify the kernel to apply a **blur** filter instead of edge detection. A simple 3×3 box blur averages all 9 pixels (center + 8 neighbors):

```
output[pos] = (tl + t + tr + l + center + r + bl + b + br) / 9;
```

## Next

Let's tackle the most important GPU pattern: [Shared Memory & Reduction →](./shared-memory-reduction)
