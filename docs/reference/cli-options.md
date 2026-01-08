---
id: cli-options
title: CLI Options Reference
description: Command-line options for the Hybridizer compiler and build tools.
keywords: [Hybridizer, CLI, command line, options, build, compiler]
---

# CLI Options Reference

This reference documents the command-line options for the Hybridizer compiler and build satellite tools.

## Hybridizer.exe

Main compiler executable.

### Basic Options

```bash
Hybridizer.exe --input <assembly> --config <config.xml> --out <directory>
```

| Option | Description | Default |
|--------|-------------|---------|
| `--input <path>` | Input .NET assembly (.dll/.exe) | Required |
| `--config <path>` | Configuration XML file | Auto-generated |
| `--out <dir>` | Output directory | `./generated-sources` |
| `--flavor <name>` | Target backend | CUDA |

### Backend Selection

| Flavor | Description |
|--------|-------------|
| `CUDA` | NVIDIA GPU (generates .cu files) |
| `OMP` | OpenMP CPU multi-threading |
| `AVX` | Intel/AMD AVX vectorization |
| `AVX2` | AVX with integer extensions |
| `AVX512` | 512-bit vector instructions |
| `NEON` | ARM NEON vectorization |
| `POWER` | IBM POWER VSX |

### Debug Options

```bash
Hybridizer.exe --input app.dll --line-info --debug-output
```

| Option | Description |
|--------|-------------|
| `--line-info` | Generate source line mapping |
| `--debug-output` | Verbose debug output |
| `--trace-level <n>` | Trace verbosity (0-3) |

### Optimization Options

```bash
Hybridizer.exe --input app.dll --opt 3 --fast-math
```

| Option | Description |
|--------|-------------|
| `--opt <level>` | Optimization level (0-3) |
| `--fast-math` | Allow fast math approximations |
| `--inline-threshold <n>` | Inlining depth limit |

## BuildSatelliteTask (MSBuild)

MSBuild task for Visual Studio integration.

```xml
<BuildSatelliteTask
  DllFullPath="MyProject.dll"
  ResultFileName="MyProject.CUDA.xml"
  GenerateLineInformation="true"
  GenerateHiddenStubs="true"
  UseFunctionPointers="true"
  WorkingDirectory="generated-sources"
  Flavors="CUDA;AVX" />
```

### Task Properties

| Property | Type | Description |
|----------|------|-------------|
| `DllFullPath` | string | Input assembly path |
| `ResultFileName` | string | Output config filename |
| `GenerateLineInformation` | bool | Include debug line info |
| `GenerateHiddenStubs` | bool | Generate internal stubs |
| `UseFunctionPointers` | bool | Enable function pointers |
| `WorkingDirectory` | string | Output directory |
| `Flavors` | string | Semicolon-separated backends |
| `BuiltInFiles` | string | Path to builtin definitions |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HYBRIDIZER_PATH` | Installation directory |
| `CUDA_PATH` | CUDA Toolkit location |
| `HYBRIDIZER_CACHE` | Cache directory for generated code |

## Configuration File

The XML configuration file controls detailed processing:

```xml
<?xml version="1.0" encoding="utf-8"?>
<HybridizerMetaConfigFile>
  <ConfigFile 
      DllFullPath="MyApp.exe" 
      GenerateLineInformation="true">
    
    <HybridFlavor FlavorName="CUDA" 
                  GenerateCWrapper="true" />
    
    <HybridElements HybridType="MyClass">
      <HybridizedMethod MethodName="Process" />
    </HybridElements>
    
  </ConfigFile>
</HybridizerMetaConfigFile>
```

See [Compilation Pipeline](../guide/compilation-pipeline) for details.

## Example Workflows

### Generate CUDA Source

```bash
Hybridizer.exe \
  --input MyProject.dll \
  --flavor CUDA \
  --out ./cuda-sources \
  --line-info
```

### Generate AVX for CI (No GPU)

```bash
Hybridizer.exe \
  --input MyProject.dll \
  --flavor AVX \
  --out ./avx-sources
```

### Full Build Pipeline

```bash
# Step 1: Build .NET
dotnet build -c Release

# Step 2: Generate CUDA
Hybridizer.exe --input bin/Release/MyProject.dll --flavor CUDA

# Step 3: Compile CUDA
nvcc -o libkernels.so generated-sources/*.cu -shared

# Step 4: Run
dotnet run
```

## Next Steps

- [Compilation Pipeline](../guide/compilation-pipeline) — Full build process
- [CI/CD Integration](../howto/ci-cd-integration) — Automation
- [Attributes Reference](./attributes-and-annotations) — Code annotations
