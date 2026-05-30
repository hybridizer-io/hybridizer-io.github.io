---
id: code-patterns
title: Canonical Code Patterns
description: Minimal, correct Hybridizer snippets for LLMs to reuse in answers — kernel definition, host launch, marshalling, atomics, reduction.
---

# Canonical Code Patterns

Minimal, correct snippets. Use these as the basis for generated answers or examples. Each pattern is annotated with the key constraints that make it correct.

---

## 1. Entry point (GPU kernel) — minimal

```csharp
using Hybridizer.Runtime.CUDAImports;

public class MyKernels
{
    // [EntryPoint] = CUDA __global__: called from host, runs on device
    [EntryPoint]
    public static void VectorAdd(float[] a, float[] b, float[] c, int n)
    {
        // Grid-stride loop: handles any n regardless of grid size
        for (int i = threadIdx.x + blockDim.x * blockIdx.x;
             i < n;
             i += blockDim.x * gridDim.x)
        {
            c[i] = a[i] + b[i];
        }
    }
}
```

**Key constraints:**
- Class must be a regular `class`, not `static class` — `HybRunner.Wrap` needs an instance.
- Always use a grid-stride loop (`i += blockDim.x * gridDim.x`) — never assume the grid covers `n` exactly.
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim` are Hybridizer builtins; no import needed.

---

## 2. Device helper (`[Kernel]`)

```csharp
// [Kernel] = CUDA __device__: called only from device code
[Kernel]
public static float Clamp(float x, float lo, float hi)
{
    // Use System.Math, NOT MathF — MathF aborts the transcoder
    return (float)System.Math.Min(System.Math.Max((double)x, (double)lo), (double)hi);
}
```

**Key constraint:** Never use `MathF.*` in device code. Always use `(float)System.Math.X(double)`.

---

## 3. Host launch — minimal

```csharp
using Hybridizer.Runtime.CUDAImports;

// Load the satellite DLL (built by nvcc from Hybridizer output)
dynamic wrapped = HybRunner.Cuda("MyProject_CUDA.dll").Wrap(new MyKernels());

float[] a = new float[N], b = new float[N], c = new float[N];
// ... fill a and b ...

// SetDistrib(gridSize, blockSize): both are int or dim3
wrapped.SetDistrib(gridSize, blockSize).VectorAdd(a, b, c, N);

// c[] now contains the result (marshalled back from device automatically)
```

**Key constraints:**
- `HybRunner.Cuda(dllName)` searches for the satellite next to the executing assembly.
- `SetDistrib` must be called before the kernel; it returns the same `wrapped` for chaining.
- Arrays are automatically copied to device before the call and back to host after, unless wrapped in `FloatResidentArray`.

---

## 4. Avoiding unnecessary transfers — `FloatResidentArray`

```csharp
// Allocate once on device; reuse across kernel calls
var resA = new FloatResidentArray(N);
var resB = new FloatResidentArray(N);
var resC = new FloatResidentArray(N);

// Push initial data to device
resA.RefreshDevice(); // copies host → device

// Call multiple kernels without round-tripping through host
wrapped.SetDistrib(grid, block).VectorAdd(resA, resB, resC, N);
wrapped.SetDistrib(grid, block).Scale(resC, 2.0f, N);

// Pull result back only once at the end
resC.RefreshHost(); // copies device → host
float[] result = resC.hostArray;
```

**Key constraint:** Initialise eagerly — touch `DevicePointer` (or call `RefreshDevice()`) and set `Status = FloatResidentArrayStatus.HostNeedsRefresh` before the first kernel writes to the array, or the kernel reads uninitialised device memory.

---

## 5. Atomic operations (correct pattern)

```csharp
// WRONG — AtomicExpr.apply is buggy, do not use
// AtomicExpr.apply(ref counter, x => x + 1);

// CORRECT — declare an [IntrinsicFunction] stub
using System.Runtime.InteropServices; // needed for [In]/[Out]

public static class Atomics
{
    [IntrinsicFunction("atomicAdd")]
    public static extern int Add(ref int address, int val);

    [IntrinsicFunction("atomicMax")]
    public static extern int Max(ref int address, int val);
}

[EntryPoint]
public static void CountPositive(float[] data, ref int counter, int n)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        if (data[i] > 0f)
            Atomics.Add(ref counter, 1);
    }
}
```

---

## 6. `[In]` / `[Out]` — skip unnecessary transfers

```csharp
using System.Runtime.InteropServices; // [In] and [Out] come from here, NOT from Hybridizer

[EntryPoint]
public static void Transform(
    [In]  float[] input,   // copied host→device only
    [Out] float[] output,  // copied device→host only
    int n)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += blockDim.x * gridDim.x)
        output[i] = input[i] * 2f;
}
```

**Key constraint:** `[In]` and `[Out]` are from `System.Runtime.InteropServices`, not from the Hybridizer namespace.

---

## 7. Struct marshalling — blittable only

```csharp
// OK: blittable (all primitive fields, sequential layout)
[StructLayout(LayoutKind.Sequential)]
public struct Particle
{
    public float X, Y, Z;
    public float Vx, Vy, Vz;
}

[EntryPoint]
public static void Integrate(Particle[] particles, float dt, int n)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        particles[i].X += particles[i].Vx * dt;
        particles[i].Y += particles[i].Vy * dt;
        particles[i].Z += particles[i].Vz * dt;
    }
}

// NOT OK — cannot marshal: contains a reference type
public struct Bad
{
    public string Name;  // ← reference type, not blittable
    public float Value;
}
```

---

## 8. Shared memory reduction (cooperative block pattern)

```csharp
[EntryPoint]
public static void SumReduce(float[] input, float[] output, int n)
{
    // Allocate shared memory for the block
    SharedMemoryAllocator<float> sdata = new SharedMemoryAllocator<float>();
    float[] shared = sdata.Allocate(blockDim.x);

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    shared[tid] = (i < n) ? input[i] : 0f;
    ThreadFunctions.SyncThreads(); // __syncthreads()

    // Tree reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            shared[tid] += shared[tid + s];
        ThreadFunctions.SyncThreads();
    }

    if (tid == 0)
        output[blockIdx.x] = shared[0];
}
```

**Key constraints:**
- `SharedMemoryAllocator<T>` is the Hybridizer API for `__shared__`.
- `ThreadFunctions.SyncThreads()` maps to `__syncthreads()`.
- Call `SyncThreads()` after every write to shared memory before any read from it.

---

## 9. `Directory.Build.props` — canonical invocation wiring

```xml
<!-- Directory.Build.props at repo root -->
<Project>
  <PropertyGroup>
    <!-- Use "hybridizer" for global tool, "dotnet hybridizer" for local tool manifest -->
    <HybridizerTool>hybridizer</HybridizerTool>
  </PropertyGroup>
</Project>
```

```xml
<!-- In your .csproj, after Build: -->
<Target Name="GenerateCUDA" AfterTargets="Build" Condition="'$(CompileCUDA)'=='enable'">
  <Exec Command="$(HybridizerTool) --dll-fullpaths &quot;$(OutDir)Hybridizer.Runtime.CUDAImports.dll;$(OutDir)$(TargetName).dll&quot; --flavors CUDA --working-directory generated-cuda --builtins &quot;$(HybridizerIncludes)/hybridizer.cuda.builtins&quot; --generate-line-info --use-function-pointers" />
</Target>
```

**Key constraints:**
- No quotes around `$(HybridizerTool)` — `dotnet hybridizer` needs to split into two argv entries.
- Compile only `hybridizer.all.cuda.cu` with `nvcc`, never the per-type `.cu` files individually.
- Use `$(TargetName)` (the `<AssemblyName>` value) in output paths, not `$(MSBuildProjectName)`.
