---
id: manage-memory
title: Manage Memory
description: Best practices for memory management with Hybridizer.
keywords: [Hybridizer, memory, pinned, streams, allocation, transfer]
---

# Manage Memory

Efficient memory management is critical for GPU performance. This guide covers ownership models, transfer optimization, and debugging techniques.

## Memory Ownership Models

### Host-Owned (Default)

The host (.NET) manages memory, Hybridizer handles transfers:

```csharp
// Host allocates
float[] data = new float[N];

// Hybridizer copies to GPU, executes, copies back
wrapper.Process(data, result, N);

// Host uses result
Console.WriteLine(result[0]);
```

**Pros**: Simple, automatic
**Cons**: Transfer overhead on every call

### Device-Owned

Keep data on GPU between kernel calls:

```csharp
// Allocate on device
IntPtr d_data, d_result;
cuda.Malloc(out d_data, N * sizeof(float));
cuda.Malloc(out d_result, N * sizeof(float));

// Copy once
cuda.Memcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Multiple kernels, no intermediate transfers
wrapper.Step1(d_data, d_result, N);
wrapper.Step2(d_result, d_data, N);
wrapper.Step3(d_data, d_result, N);

// Copy result once
cuda.Memcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

// Cleanup
cuda.Free(d_data);
cuda.Free(d_result);
```

**Pros**: Minimal transfer overhead
**Cons**: Manual memory management

## Minimize Transfers

### Batch Operations

```csharp
// Bad: Transfer on every call
for (int i = 0; i < 100; i++)
{
    wrapper.Process(data, result, N);  // ❌ 100 transfers
}

// Good: Keep data on device
cuda.Memcpy(d_data, h_data, size, H2D);
for (int i = 0; i < 100; i++)
{
    wrapper.Process(d_data, d_result, N);  // ✅ No transfers
}
cuda.Memcpy(h_result, d_result, size, D2H);
```

### Reuse Allocations

```csharp
// Bad: Allocate every call
void ProcessBatch(float[] input)
{
    float[] temp = new float[N];  // ❌ Allocation overhead
    wrapper.Process(input, temp, N);
}

// Good: Reuse buffers
private float[] _tempBuffer;

void ProcessBatch(float[] input)
{
    if (_tempBuffer == null)
        _tempBuffer = new float[N];  // ✅ Allocate once
    wrapper.Process(input, _tempBuffer, N);
}
```

## Pinned Memory

Pinned (page-locked) memory enables faster transfers:

```csharp
// Allocate pinned memory
IntPtr pinnedPtr;
cuda.MallocHost(out pinnedPtr, N * sizeof(float));

// Wrap for .NET access
float[] pinnedArray = new float[N];
// Copy data to pinned region...

// Use with async copies
cuda.MemcpyAsync(d_data, pinnedPtr, size, H2D, stream);
```

**Benefits**:
- Direct DMA transfer (no staging)
- Required for async copies
- Lower latency

**Cautions**:
- Limited system resource
- Must be freed explicitly

## Async Transfers with Streams

Overlap compute and data transfer:

```csharp
// Create streams
cudaStream_t stream1, stream2;
cuda.StreamCreate(out stream1);
cuda.StreamCreate(out stream2);

// Double buffering pattern
for (int batch = 0; batch < numBatches; batch++)
{
    int current = batch % 2;
    int next = (batch + 1) % 2;
    cudaStream_t currentStream = (current == 0) ? stream1 : stream2;
    cudaStream_t nextStream = (next == 0) ? stream1 : stream2;
    
    // Start next transfer while current computes
    if (batch + 1 < numBatches)
        cuda.MemcpyAsync(d_data[next], h_data[next], size, H2D, nextStream);
    
    // Compute current batch
    wrapper.SetStream(currentStream).Process(d_data[current], d_result[current], N);
    
    // Copy current results back
    cuda.MemcpyAsync(h_result[current], d_result[current], size, D2H, currentStream);
}

cuda.StreamSynchronize(stream1);
cuda.StreamSynchronize(stream2);
```

## Memory Debugging

### Check for Leaks

```csharp
// Track allocations
long beforeMem, afterMem;
cuda.MemGetInfo(out beforeMem, out _);

// Your code...

cuda.MemGetInfo(out afterMem, out _);
if (beforeMem != afterMem)
    Console.WriteLine($"Possible leak: {beforeMem - afterMem} bytes");
```

### CUDA-MEMCHECK

```bash
# Check for memory errors
compute-sanitizer --tool memcheck ./my_program

# Detect leaks
compute-sanitizer --tool memcheck --leak-check full ./my_program
```

## Best Practices Summary

| Practice | Impact |
|----------|--------|
| Keep data on GPU | High |
| Reuse allocations | Medium |
| Use pinned memory | Medium |
| Overlap with streams | Medium |
| Batch small transfers | Medium |

## Next Steps

- [Data Marshalling](../guide/data-marshalling) — Transfer details
- [Optimize Kernels](./optimize-kernels) — Performance tuning
- [Memory & Profiling](../cuda/memory-and-profiling) — Profiling tools
