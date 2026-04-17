---
id: setup
title: "1. Setup & Installation"
description: "Install all prerequisites and verify your Hybridizer environment in 15 minutes."
keywords: [Hybridizer, install, setup, CUDA, .NET 8, Linux, Windows]
sidebar_position: 1
---

# Setup & Installation

This tutorial gets you from zero to a working Hybridizer environment. By the end, you'll have compiled and run your first GPU-accelerated build.

Hybridizer supports **Windows** and **Linux**, using **.NET 8** (or later).

## Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 x64 / Ubuntu 22.04 | Windows 11 x64 / Ubuntu 24.04 |
| **.NET SDK** | .NET 8.0 | .NET 8.0 (latest patch) |
| **GPU** | Any NVIDIA (Compute ≥ 5.0) | RTX 2060 or newer |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 10 GB free | SSD recommended |

## Step 1: Install NVIDIA Drivers

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="windows" label="Windows" default>

1. Go to [nvidia.com/drivers](https://www.nvidia.com/drivers) or open **NVIDIA GeForce Experience**
2. Download and install the latest driver
3. Verify:

```bash
nvidia-smi
```

</TabItem>
<TabItem value="linux" label="Linux">

Install the recommended driver for your distribution:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y nvidia-driver-560
sudo reboot

# After reboot, verify:
nvidia-smi
```

Alternatively, install via the [CUDA toolkit](#step-2-install-cuda-toolkit) which bundles a compatible driver.

</TabItem>
</Tabs>

You should see your GPU name, driver version, and CUDA version:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 560.xx    Driver Version: 560.xx    CUDA Version: 12.6          |
|-------------------------------+----------------------+----------------------+
| GPU  Name        | ...       |                      |
| GeForce RTX 4070 | ...       |                      |
+-------------------------------+----------------------+----------------------+
```

## Step 2: Install CUDA Toolkit

Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

<Tabs>
<TabItem value="windows" label="Windows" default>

1. Choose **Windows → x86_64 → exe (local)**
2. Run the installer — default options are fine
3. Verify:

```bash
nvcc --version
```

:::tip
If `nvcc` is not found, add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` to your system PATH.
:::

</TabItem>
<TabItem value="linux" label="Linux">

```bash
# Ubuntu 22.04 / 24.04 — network install
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
```

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Verify:

```bash
source ~/.bashrc
nvcc --version
```

</TabItem>
</Tabs>

Expected output:

```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.x, V12.x.xxx
```

## Step 3: Install .NET 8 SDK

<Tabs>
<TabItem value="windows" label="Windows" default>

Download and install from [dotnet.microsoft.com/download](https://dotnet.microsoft.com/download/dotnet/8.0).

Alternatively, if using Visual Studio 2022 (17.8+), .NET 8 is included with the **.NET desktop development** workload.

</TabItem>
<TabItem value="linux" label="Linux">

```bash
# Ubuntu (via Microsoft packages)
sudo apt install -y dotnet-sdk-8.0

# Or via the install script
curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 8.0
```

Verify:

```bash
dotnet --version    # Should print 8.0.xxx
```

</TabItem>
</Tabs>

## Step 4: Install a C++ Toolchain

Hybridizer generates C++ code that needs to be compiled by `nvcc` and a host C++ compiler.

<Tabs>
<TabItem value="windows" label="Windows" default>

Install **Visual Studio 2022** with the following workloads:
- ✅ **.NET desktop development**
- ✅ **Desktop development with C++** (provides MSVC, needed by `nvcc`)

Or install the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (lighter, no IDE).

</TabItem>
<TabItem value="linux" label="Linux">

```bash
# GCC (required by nvcc)
sudo apt install -y build-essential

# Verify
g++ --version
```

</TabItem>
</Tabs>

## Step 5: Install Hybridizer

Create a new .NET 8 console project and add the Hybridizer packages:

```bash
dotnet new console -n MyFirstHybridizer --framework net8.0
cd MyFirstHybridizer
dotnet add package Hybridizer.Runtime.CUDAImports
```

:::tip
On **Windows with Visual Studio**, you can also install the **Hybridizer Community Edition** extension from the Visual Studio Marketplace for integrated project templates.
:::

## Step 6: Verify Your Setup

Replace the content of `Program.cs` with:

```csharp
using System;
using Hybridizer.Runtime.CUDAImports;

class Program
{
    [EntryPoint]
    public static void TestKernel(int[] output, int N)
    {
        for (int i = threadIdx.x + blockDim.x * blockIdx.x;
             i < N;
             i += blockDim.x * gridDim.x)
        {
            output[i] = i * 2;
        }
    }

    static void Main()
    {
        // Check GPU
        cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);
        Console.WriteLine($"GPU: {new string(prop.name)}");
        Console.WriteLine($"SMs: {prop.multiProcessorCount}");
        Console.WriteLine($"Memory: {prop.totalGlobalMem / (1024*1024)} MB");

        // Run kernel
        int N = 1024;
        int[] output = new int[N];

        dynamic wrapper = HybRunner.Cuda()
            .SetDistrib(32, 256);
        wrapper.TestKernel(output, N);
        cuda.DeviceSynchronize();

        // Verify
        bool ok = true;
        for (int i = 0; i < N; i++)
        {
            if (output[i] != i * 2) { ok = false; break; }
        }

        Console.WriteLine(ok ? "✅ Hybridizer is working!" : "❌ Something went wrong");
    }
}
```

Build and run:

```bash
dotnet build
dotnet run
```

Expected output:

```
GPU: NVIDIA GeForce RTX 4070
SMs: 46
Memory: 12282 MB
✅ Hybridizer is working!
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found | Install/update NVIDIA drivers |
| `nvcc` not found | Reinstall CUDA Toolkit, check PATH |
| `dotnet` not found | Install .NET 8 SDK |
| Build error "Hybridizer not found" | Verify NuGet package: `dotnet list package` |
| Runtime "No CUDA device" | Check GPU with `nvidia-smi`, update drivers |
| `libcudart.so` not found (Linux) | Set `LD_LIBRARY_PATH` to CUDA lib directory |
| DLL not found at runtime (Windows) | Ensure CUDA bin directory is in PATH |

## Next

You're ready! Proceed to [Your First Kernel →](./first-kernel)
