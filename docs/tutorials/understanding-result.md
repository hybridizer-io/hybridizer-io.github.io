---
id: understanding-result
title: "3. Understanding the Result"
description: "Debug, verify, and understand what Hybridizer generates behind the scenes."
keywords: [Hybridizer, debug, OMP, generated code, profiling, error checking]
sidebar_position: 3
---

# Understanding the Result

You've run your first kernel. Now let's understand what happened, how to debug issues, and how to optimize data transfers.

## What Hybridizer Generated

When you build your project, Hybridizer:

1. **Reads** your .NET assembly (`.dll`)
2. **Finds** methods marked with `[EntryPoint]`
3. **Generates** CUDA C++ source code
4. **Compiles** it with `nvcc` into a native library
5. **Links** it at runtime via `HybRunner`

You can inspect the generated code in your project's build output directory:

```
bin/Debug/
├── YourProject.dll          ← Your .NET code
├── YourProject_CUDA.cu      ← Generated CUDA source
├── YourProject_CUDA.dll     ← Compiled GPU binary
└── ...
```

:::tip
**Reading the generated `.cu` file is the best way to understand what Hybridizer does.** Open it — you'll see your C# translated to standard CUDA C++.

*Note: the generated `.cu` source file is only available in the **Enterprise** edition.*
:::

## Error Checking

GPU errors are **silent by default**. Always check for them:

```csharp
// After any GPU call
wrapper.VectorAdd(a, b, result, N);

// Check for launch errors
cudaError_t err = cuda.GetPeekAtLastError();
if (err != cudaError_t.cudaSuccess)
{
    Console.Error.WriteLine($"Kernel launch error: {cuda.GetErrorString(err)}");
    return;
}

// Wait for completion and check
err = cuda.DeviceSynchronize();
if (err != cudaError_t.cudaSuccess)
{
    Console.Error.WriteLine($"Kernel execution error: {cuda.GetErrorString(err)}");
    return;
}
```

Or use the shorthand:

```csharp
wrapper.VectorAdd(a, b, result, N);
cuda.ERROR_CHECK(cuda.DeviceSynchronize());
```

## `DeviceSynchronize` — Why It Matters

GPU kernels run **asynchronously**. When `wrapper.VectorAdd(...)` returns, the GPU might still be working:

```csharp
wrapper.VectorAdd(a, b, result, N);
// ⚠️ result[] might not be ready yet!
Console.WriteLine(result[0]);  // Could print 0 (old value)

cuda.DeviceSynchronize();
// ✅ Now result[] is guaranteed to be read back
Console.WriteLine(result[0]);  // Correct value
```

:::warning
**Always call `cuda.DeviceSynchronize()` before reading results.** Forgetting this is the #1 cause of "wrong results" bugs.
:::

## Optimize Transfers with `[In]` / `[Out]`

By default, Hybridizer copies every array **both ways** (host↔device). That's wasteful:

```csharp
// Without attributes: a, b, result all copied both ways = 6 transfers
wrapper.VectorAdd(a, b, result, N);
```

Use marshalling attributes to specify direction:

```csharp
using System.Runtime.InteropServices;

[EntryPoint]
public static void VectorAdd(
    [In]  float[] a,       // Host → Device only (read-only)
    [In]  float[] b,       // Host → Device only (read-only)
    [Out] float[] result,  // Device → Host only (write-only)
    int N)
{
    // ...
}
```

| Attribute | Transfer | Use When |
|-----------|----------|----------|
| (none) | ↔ Both ways | Array is read AND written |
| `[In]` | → Host to Device | Array is read-only on GPU |
| `[Out]` | ← Device to Host | Array is write-only on GPU |

Impact: with 1M floats (4 MB each), proper attributes **save 12 MB of PCI-e transfer**.

## Debug with OMP Backend

Don't have a GPU? Or want to debug with breakpoints? Use the **OMP backend**:

```csharp
// Instead of GPU...
dynamic wrapper = HybRunner.Cuda();

// Use OpenMP (runs on CPU, same code path)
dynamic wrapper = HybRunner.OMP();
```

This runs the same generated code on CPU with OpenMP threads. Useful for:
- Machines without GPU
- Setting breakpoints in the generated code
- Verifying numerical correctness

:::tip
If results are **correct with OMP but wrong with CUDA**, the bug is likely a parallelization issue (race condition, missing sync).
:::

## First Profiling

### Quick Timing

```csharp
var sw = System.Diagnostics.Stopwatch.StartNew();

wrapper.VectorAdd(a, b, result, N);
cuda.DeviceSynchronize();

sw.Stop();
Console.WriteLine($"GPU time: {sw.ElapsedMilliseconds} ms");
```

:::info
The **first call** is always slow (CUDA context initialization). Measure the second call for accurate timing.
:::

### NVIDIA Nsight

For detailed profiling, use NVIDIA's tools:

```bash
# Command-line profiling
ncu --set full YourProject.exe

# System-wide timeline
nsys profile YourProject.exe
```

## Recap: The Debugging Checklist

When something goes wrong:

1. ✅ Check `cuda.ERROR_CHECK(cuda.DeviceSynchronize())`
2. ✅ Compare GPU result with CPU reference
3. ✅ Try OMP backend — if it works, the issue is parallelization
4. ✅ Inspect the generated `.cu` file
5. ✅ Make sure `DeviceSynchronize()` is called before reading results
6. ✅ Check `[In]`/`[Out]` attributes match your usage

## Next

Now let's learn the real skill — transforming existing CPU code for GPU: [From CPU to GPU →](./cpu-to-gpu)
