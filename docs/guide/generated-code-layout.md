---
id: generated-code-layout
title: Generated Code Layout
description: Understanding the files Hybridizer generates and how to customize the build.
keywords: [Hybridizer, generated code, build, output, .cu, nvcc]
---

# Generated Code Layout

When Hybridizer processes your project, it generates several files. Understanding this layout helps with debugging and customization.

## Output Directory Structure

```
bin/Release/
├── YourProject.dll              ← Your .NET assembly
├── YourProject_CUDA.cu          ← Generated CUDA source
├── YourProject_CUDA.dll         ← Compiled GPU binary
├── YourProject_CUDA.config.xml  ← Hybridizer configuration
├── YourProject_CUDA.ptx         ← PTX intermediate (if kept)
└── ...
```

## The Generated `.cu` File

This is the most informative file. It contains:

### 1. Struct Definitions

Your C# structs are translated to C++ structs:

```cpp
// From: public struct MyData { public float x; public int count; }
struct MyData {
    float x;
    int count;
};
```

### 2. Device Functions

Methods marked `[Kernel]`:

```cpp
// From: [Kernel] public static float Compute(float x) { ... }
__device__ float Compute(float x) {
    // ...
}
```

### 3. Global Kernels

Methods marked `[EntryPoint]`:

```cpp
// From: [EntryPoint] public static void Run(float[] a, int N) { ... }
__global__ void Run(float* a, int N) {
    // Grid-stride loop generated from Parallel.For or manual loop
}
```

### 4. Wrapper Functions

Host-callable functions that handle marshalling:

```cpp
extern "C" __declspec(dllexport)
void Run_wrapper(float* a, int N) {
    Run<<<gridDim, blockDim, sharedMem, stream>>>(a, N);
}
```

## The Configuration File

`YourProject_CUDA.config.xml` contains:
- Entry point list
- Type mappings
- Generic specializations (`[HybridRegisterTemplate]`)
- Intrinsic mappings

Each generated source file includes a sub-configuration comment for **reproducibility**.

## Customizing the Build

### MSBuild Properties

Add to your `.csproj`:

```xml
<PropertyGroup>
  <!-- Emit C# line info in generated CUDA -->
  <HybridizerEmitLineInfo>true</HybridizerEmitLineInfo>

  <!-- Target specific GPU architecture -->
  <HybridizerCudaArch>sm_86</HybridizerCudaArch>

  <!-- Keep intermediate PTX -->
  <HybridizerKeepPTX>true</HybridizerKeepPTX>

  <!-- Additional nvcc flags -->
  <HybridizerNvccFlags>--use_fast_math</HybridizerNvccFlags>
</PropertyGroup>
```

### Common Architecture Codes

| GPU Series | Compute Capability | `sm_` Value |
|-------|-----|------|
| GTX 10xx | 6.1 | `sm_61` |
| RTX 20xx | 7.5 | `sm_75` |
| RTX 30xx | 8.6 | `sm_86` |
| RTX 40xx | 8.9 | `sm_89` |
| H100 | 9.0 | `sm_90` |

## Reading Generated Code for Debugging

When a kernel gives wrong results:

1. Open the `.cu` file
2. Find your entry point function
3. Check:
   - Are loop bounds correct?
   - Are `threadIdx`/`blockIdx` used properly?
   - Are shared memory declarations present?
   - Are `[In]`/`[Out]` reflected in the wrapper?

:::tip
The generated code is valid, readable CUDA C++. If you know CUDA, reading it is the fastest way to understand what Hybridizer does with your code.
:::
