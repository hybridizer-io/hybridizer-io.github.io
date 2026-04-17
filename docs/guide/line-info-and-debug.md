---
id: line-info-and-debug
title: Line Info & Debugging
description: Source-level debugging and line mappings between C# and generated CUDA code.
keywords: [Hybridizer, debug, line info, Nsight, source mapping]
---

# Line Info & Debugging

Hybridizer can emit **line information** that maps generated CUDA code back to your original C# source. This enables source-level debugging and profiling.

## Enabling Line Info

Add to your `.csproj`:

```xml
<PropertyGroup>
  <HybridizerEmitLineInfo>true</HybridizerEmitLineInfo>
</PropertyGroup>
```

With this enabled, the generated `.cu` files include `#line` directives:

```cpp
#line 42 "C:/MyProject/Program.cs"
    float result = a[i] + b[i];
#line 43 "C:/MyProject/Program.cs"
    output[i] = result;
```

## Using NVIDIA Nsight with Line Info

### Nsight Compute

When profiling with `ncu`, line info allows you to see **which C# line** is the bottleneck:

```bash
ncu --set full --source-folders "C:/MyProject" MyProject.exe
```

The Source view will show your C# code with per-line metrics (cycles, memory operations, etc.).

### Nsight Systems

```bash
nsys profile --cuda-graph-trace=true MyProject.exe
```

Shows a timeline with kernel names matching your C# method names.

## Debugging Workflow

### 1. Start with OMP Backend

For initial debugging, use the CPU backend:

```csharp
dynamic wrapper = HybRunner.OMP();
```

This lets you:
- Set breakpoints in Visual Studio
- Step through the generated code
- Inspect variables normally

### 2. Compare OMP vs CUDA

```csharp
// Run both
HybRunner.OMP().Wrap(new Program()).MyKernel(cpu_args);
HybRunner.Cuda().Wrap(new Program()).MyKernel(gpu_args);
cuda.DeviceSynchronize();

// If CPU result is correct but GPU isn't → parallelization bug
```

### 3. Inspect Generated Code

The `.cu` file is your best friend. Common things to check:

| What to Check | Where to Look |
|---------------|--------------|
| Loop bounds | `for` loop in `__global__` function |
| Memory access | Array indexing patterns |
| Shared memory | `__shared__` declarations |
| Sync barriers | `__syncthreads()` placement |
| Intrinsic mapping | Function names (`sinf`, `expf`, etc.) |

### 4. Use printf for GPU Debugging

```csharp
[IntrinsicFunction("printf")]
public static int DevicePrintf(string format, params object[] args)
{
    Console.Write(format, args);
    return 0;
}
```

:::warning
GPU `printf` is buffered and only flushed on `DeviceSynchronize`. It's useful for debugging but impacts performance heavily. Remove from production code.
:::

## Common Debug Scenarios

### Kernel doesn't launch

```csharp
wrapper.MyKernel(args);
cudaError_t err = cuda.GetPeekAtLastError();
// Check err — common: cudaErrorInvalidConfiguration (bad SetDistrib)
```

### Race condition in shared memory

Symptoms: non-deterministic results with CUDA, correct with OMP.

Fix: ensure every path calls `CUDAIntrinsics.__syncthreads()` at the same point.

### Numerical differences

GPU and CPU floating-point differ due to:
- Different operation ordering (associativity)
- GPU fused multiply-add (`fma`) vs separate multiply+add
- Different rounding modes

Always use tolerance: `Math.Abs(gpu - cpu) < 1e-5f`

## Tips

- Build in **Debug** mode for better line info mapping
- Keep generated `.cu` files under version control for diff comparison
- Use `[IntrinsicFunction("__trap")]` to create GPU breakpoints
- The `.config.xml` file documents all type mappings — useful for understanding complex generics
