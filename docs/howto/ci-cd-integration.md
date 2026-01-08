---
id: ci-cd-integration
title: CI/CD Integration
description: Setting up continuous integration for Hybridizer projects.
keywords: [Hybridizer, CI, CD, DevOps, testing, build, GitHub Actions]
---

# CI/CD Integration

This guide covers setting up continuous integration and deployment pipelines for Hybridizer projects.

## Overview

```mermaid
flowchart LR
    A[Push Code] --> B[Build .NET]
    B --> C[Generate CUDA/AVX]
    C --> D[Compile Native]
    D --> E[Run Tests]
    E --> F[Publish Artifacts]
```

## Pin Toolchain Versions

Always pin versions for reproducible builds:

```xml
<!-- Directory.Build.props -->
<PropertyGroup>
  <HybridizerVersion>2.3.0</HybridizerVersion>
  <CudaToolkitVersion>12.3</CudaToolkitVersion>
</PropertyGroup>
```

```yaml
# CI environment
environment:
  CUDA_VERSION: "12.3"
  HYBRIDIZER_VERSION: "2.3.0"
```

## GitHub Actions Example

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: '8.0.x'
      
      - name: Build
        run: dotnet build --configuration Release
      
      - name: Test (OMP/AVX backend)
        run: dotnet test --configuration Release

  build-gpu:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup CUDA
        run: |
          export PATH=/usr/local/cuda-12.3/bin:$PATH
          nvcc --version
      
      - name: Build with CUDA
        run: dotnet build --configuration Release -p:EnableCuda=true
      
      - name: Test GPU
        run: dotnet test --configuration Release --filter "Category=GPU"
```

## Cache Strategies

### Cache Generated Code

```yaml
- name: Cache Hybridizer output
  uses: actions/cache@v4
  with:
    path: |
      **/generated-sources/
      **/obj/hybridizer/
    key: hybridizer-${{ hashFiles('**/*.cs') }}-${{ env.HYBRIDIZER_VERSION }}
```

### Cache CUDA Compilation

```yaml
- name: Cache PTX/cubin
  uses: actions/cache@v4
  with:
    path: ~/.nv/ComputeCache
    key: cuda-cache-${{ runner.os }}-${{ env.CUDA_VERSION }}
```

## Testing Strategy

### GPU vs CPU Runners

| Test Type | Runner | Backend |
|-----------|--------|---------|
| Unit tests | CPU | OMP |
| Numeric parity | CPU | AVX |
| Performance | GPU | CUDA |
| Integration | GPU | CUDA |

### Numeric Parity Tests

```csharp
[Test]
public void VectorAdd_MatchesCpuReference()
{
    // CPU reference
    float[] cpuResult = ComputeOnCpu(data);
    
    // GPU result
    float[] gpuResult = ComputeOnGpu(data);
    
    // Compare with tolerance
    Assert.That(gpuResult, Is.EqualTo(cpuResult).Within(1e-5f));
}
```

### Performance Gates

```csharp
[Test]
[Category("Performance")]
public void MatrixMultiply_MeetsPerformanceBudget()
{
    var sw = Stopwatch.StartNew();
    wrapper.MatMul(A, B, C, N);
    cuda.DeviceSynchronize();
    sw.Stop();
    
    double gflops = (2.0 * N * N * N) / (sw.Elapsed.TotalSeconds * 1e9);
    Assert.That(gflops, Is.GreaterThan(1000), "Expected > 1 TFLOP/s");
}
```

## Artifact Publishing

### Package Structure

```
artifacts/
├── lib/
│   ├── MyProject.dll
│   ├── MyProject.CUDA.ptx
│   └── MyProject.AVX.dll
├── symbols/
│   ├── MyProject.pdb
│   └── MyProject.line.map
└── docs/
    └── api/
```

### NuGet Packaging

```xml
<PropertyGroup>
  <IncludeNativeOutputs>true</IncludeNativeOutputs>
  <GeneratePackageOnBuild>true</GeneratePackageOnBuild>
</PropertyGroup>

<ItemGroup>
  <Content Include="runtimes\win-x64\native\*.dll">
    <PackagePath>runtimes\win-x64\native</PackagePath>
  </Content>
</ItemGroup>
```

## Multi-Platform Matrix

```yaml
strategy:
  matrix:
    include:
      - os: windows-latest
        cuda: "12.3"
        backend: CUDA
      - os: ubuntu-latest
        cuda: "12.3"
        backend: CUDA
      - os: ubuntu-latest
        backend: AVX512
      - os: macos-latest
        backend: AVX
```

## Best Practices

| Practice | Benefit |
|----------|---------|
| Pin all versions | Reproducibility |
| Cache aggressively | Speed |
| Test on CPU first | Faster feedback |
| GPU tests on merge | Cost efficiency |
| Publish symbols | Debuggability |

## Next Steps

- [Compilation Pipeline](../guide/compilation-pipeline) — Build process details
- [Use Libraries](./use-libraries) — Library integration
- [Performance Metrics](../cuda/perf-metrics) — Performance testing
