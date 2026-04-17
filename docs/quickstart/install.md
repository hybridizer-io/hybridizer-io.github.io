---
id: install
title: Install
description: Quick installation checklist for Hybridizer on Windows and Linux.
---

# Install

For a detailed, step-by-step guide, see [Tutorial 1: Setup & Installation](../tutorials/setup).

## Quick Checklist

### Prerequisites

- **.NET 8 SDK** — [download](https://dotnet.microsoft.com/download/dotnet/8.0)
- **NVIDIA CUDA Toolkit** (13.x recommended) — [download](https://developer.nvidia.com/cuda-downloads)
- **NVIDIA GPU** with compatible drivers
- **C++ toolchain**: MSVC (Windows) or GCC (Linux)

### Install

```bash
dotnet tool install -g Hybridizer
dotnet new install Hybridizer.App.Template
dotnet new hybridizer-app -n MyProject
cd MyProject
```

### Verify

```bash
nvidia-smi       # GPU detected?
nvcc --version   # CUDA toolkit installed?
dotnet --version # .NET 8+?
dotnet build     # Project builds?
dotnet run       # Kernel runs?
```

### Next

[Hello GPU →](./hello-gpu) or [Tutorial 1 →](../tutorials/setup) for the full walkthrough.
