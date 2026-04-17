---
id: run-and-debug
title: Run and Debug
description: Build, run, and debug Hybridizer projects with line-info mapping and profilers.
keywords: [Hybridizer, debug, run, Nsight, profiling, OMP, line info]
---

# Run and Debug

## Build Process

When you build a Hybridizer project, three things happen:

1. **Standard C# compilation** → Your `.dll` assembly
2. **Hybridizer code generation** → CUDA `.cu` source files
3. **nvcc compilation** → Native GPU library (`_CUDA.dll`)

```
YourProject.csproj
  → dotnet build
    → YourProject.dll (MSBuild step 1)
    → YourProject_CUDA.cu (Hybridizer generates)
    → nvcc YourProject_CUDA.cu (MSBuild step 2)
    → YourProject_CUDA.dll (native GPU code)
```

## Run Configurations

### CUDA (GPU)

```csharp
dynamic wrapper = HybRunner.Cuda()
    .SetDistrib(128, 256);
```

### OMP (CPU — for debugging)

```csharp
dynamic wrapper = HybRunner.OMP();
```

Runs the same generated code on CPU with OpenMP threads. Useful for:
- Debugging without GPU
- Setting breakpoints in generated code
- Validating numerical correctness

## Enable Line Information

Build with debug info to map generated CUDA code back to your C# source:

```xml
<!-- In your .csproj -->
<PropertyGroup>
  <HybridizerEmitLineInfo>true</HybridizerEmitLineInfo>
</PropertyGroup>
```

With line info enabled:
- NVIDIA Nsight shows your **C# source lines** in the profiler
- Errors reference your original C# code, not the generated `.cu`

## Debugging Workflow

### Step 1: Verify with OMP

```csharp
#if DEBUG
    dynamic wrapper = HybRunner.OMP();
#else
    dynamic wrapper = HybRunner.Cuda();
#endif
```

### Step 2: Check for Errors

```csharp
wrapper.MyKernel(args);
cuda.ERROR_CHECK(cuda.DeviceSynchronize());
```

### Step 3: Compare GPU vs CPU

```csharp
// Direct C# call = CPU reference
MyKernel(args_cpu);

// GPU call
wrapper.MyKernel(args_gpu);
cuda.DeviceSynchronize();

// Compare
for (int i = 0; i < N; i++)
    Assert.AreEqual(cpu[i], gpu[i], 1e-5f);
```

### Step 4: Inspect Generated Code

Open the `.cu` file in your build output. Look for:
- Correct loop bounds
- Memory access patterns
- Shared memory declarations

## Profiling

### Quick Timing

```csharp
var sw = Stopwatch.StartNew();
wrapper.MyKernel(args);
cuda.DeviceSynchronize();
sw.Stop();
Console.WriteLine($"{sw.ElapsedMilliseconds} ms");
```

### NVIDIA Nsight Systems (timeline)

```bash
nsys profile --stats=true YourProject.exe
```

Shows:
- Kernel launch timeline
- Memory copy durations
- CPU/GPU overlap

### NVIDIA Nsight Compute (kernel analysis)

```bash
ncu --set full YourProject.exe
```

Shows:
- Achieved occupancy
- Memory throughput
- Compute throughput
- Warp stalls

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | Select GPU | `0` (first GPU) |
| `HYBRIDIZER_VERBOSE` | Verbose build output | `1` |

## Common Debugging Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Results all zeros | Missing `DeviceSynchronize` | Add sync before reading |
| Random wrong values | Race condition | Check `__syncthreads` placement |
| Build succeeds, crash at runtime | DLL not found | Check bin/ for `_CUDA.dll` |
| Correct with OMP, wrong with CUDA | Parallelization issue | Check shared memory / atomics |

See also: [FAQ & Troubleshooting](./faq-troubleshooting) for more solutions.
