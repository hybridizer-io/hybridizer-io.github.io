---
id: streams
title: "CUDA Streams"
description: "Overlapping compute and memory transfers with multiple streams."
keywords: [Hybridizer, streams, async, overlap, pinned memory]
---

# CUDA Streams

> **Sample source**: [`5.CUDA_runtime/Streams`](https://github.com/hybridizer-io/hybridizer-basic-samples/tree/master/src/5.CUDA_runtime/Streams)

This example shows how to use CUDA streams with Hybridizer to **overlap computation and memory transfers** for better throughput.

## Why Streams?

Without streams, operations are sequential:
```
Copy H→D ──> Kernel ──> Copy D→H
```

With streams, operations from different streams execute concurrently:
```
Stream 0: Copy H→D ──> Kernel ──> Copy D→H
Stream 1:     Copy H→D ──> Kernel ──> Copy D→H
Stream 2:         Copy H→D ──> Kernel ──> Copy D→H
```

## The Kernel

```csharp
[EntryPoint]
public static void Add(float[] a, [In] float[] b, int start, int stop, int iter)
{
    for (int k = start + threadIdx.x + blockDim.x * blockIdx.x;
         k < stop;
         k += blockDim.x * gridDim.x)
    {
        for (int p = 0; p < iter; ++p)
        {
            a[k] += b[k];
        }
    }
}
```

## Stream Setup

```csharp
int nStreams = 8;
cudaStream_t[] streams = new cudaStream_t[nStreams];
dynamic wrapped = SatelliteLoader.Load().Wrap(new Program());

// Create streams
for (int k = 0; k < nStreams; ++k)
    cuda.StreamCreate(out streams[k]);
```

## Manual Device Memory

Streams require manual memory management (automatic marshalling is synchronous):

```csharp
int N = 1024 * 1024 * 32;
IntPtr d_a, d_b;   // device pointers
float[] a = new float[N];
float[] b = new float[N];

cuda.Malloc(out d_a, N * sizeof(float));
cuda.Malloc(out d_b, N * sizeof(float));
```

## Pinned Host Memory

For async copies, host memory must be **pinned** (page-locked):

```csharp
GCHandle handle_a = GCHandle.Alloc(a, GCHandleType.Pinned);
GCHandle handle_b = GCHandle.Alloc(b, GCHandleType.Pinned);
IntPtr h_a = handle_a.AddrOfPinnedObject();
IntPtr h_b = handle_b.AddrOfPinnedObject();
```

:::warning
Pinned memory is a limited system resource. Always free handles when done: `handle_a.Free()`.
:::

## Multi-Stream Execution

Split data into slices and process each on a different stream:

```csharp
// Initial copy
cuda.Memcpy(d_a, h_a, N * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
cuda.Memcpy(d_b, h_b, N * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);

int slice = N / nStreams;

// Launch kernels on different streams
for (int k = 0; k < nStreams; ++k)
{
    int start = k * slice;
    int stop = start + slice;
    wrapped.SetStream(streams[k]).Add(d_a, d_b, start, stop, 100);
}

// Async copy results back per stream
for (int k = 0; k < nStreams; ++k)
{
    int start = k * slice;
    cuda.MemcpyAsync(
        h_a + start * sizeof(float),
        d_a + start * sizeof(float),
        slice * sizeof(float),
        cudaMemcpyKind.cudaMemcpyDeviceToHost,
        streams[k]);
}

// Wait for completion
for (int k = 0; k < nStreams; ++k)
{
    cuda.StreamSynchronize(streams[k]);
    cuda.StreamDestroy(streams[k]);
}
```

### Key API: `SetStream`

```csharp
wrapped.SetStream(streams[k]).Add(d_a, d_b, start, stop, 100);
```

`SetStream` returns the wrapper itself, enabling fluent calls.

## Cleanup

```csharp
// Free pinned handles
handle_a.Free();
handle_b.Free();

// Free device memory
cuda.Free(d_a);
cuda.Free(d_b);
```

## When to Use Streams

| Scenario | Benefit |
|----------|---------|
| Large data with partitioned work | Overlap copy + compute |
| Multiple independent kernels | Concurrent execution |
| Pipeline processing | Hide latency |

## Next Steps

- [Constant Memory](./constant-memory) — Another CUDA runtime feature
- [Manage Memory](../howto/manage-memory) — Memory best practices
- [Invoke Generated Code](../guide/invoke-generated-code) — Stream API details
