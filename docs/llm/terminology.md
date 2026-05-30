---
id: terminology
title: Preferred Terminology
description: Canonical terms to use when writing or answering questions about Hybridizer.
---

# Preferred Terminology

Use these terms consistently. The "avoid" column lists phrasings that are imprecise, ambiguous, or commonly misused.

## Core product

| Preferred | Avoid | Notes |
|-----------|-------|-------|
| **Hybridizer** | "the transpiler", "the translator", "the converter" | It's a compiler, not a source-to-source translator in the general sense |
| **Hybridizer CLI** | "the Hybridizer executable", "hybridizer.exe" | The CLI works on both Windows and Linux; avoid platform-specific names |
| **free NuGet tool** | "free version", "community edition" | It's a NuGet dotnet tool: `dotnet tool install --global Hybridizer` |
| **paid standalone** | "pro version", "enterprise edition" | Adds OMP/HIP/AVX flavors and emits readable `.cu` source |

## Input / Output

| Preferred | Avoid | Notes |
|-----------|-------|-------|
| **managed IL** or **MSIL** | "bytecode", "source code" | Hybridizer operates on compiled IL, not C# source |
| **LLVM-IR** | "LLVM bytecode" | When referring to the LLVM input path |
| **native backend** | "native code", "target" | The generated output (CUDA/OMP/AVX) before compilation by nvcc/g++ |
| **generated source** | "transpiled code", "output code" | The `.cu` or `.cpp` files Hybridizer writes (paid standalone only for CUDA; free tool emits a cubin blob) |
| **satellite DLL** | "native library", "GPU binary" | The `.dll`/`.so` produced by `nvcc`/`g++`, loaded at runtime |

## Code concepts

| Preferred | Avoid | Notes |
|-----------|-------|-------|
| **entry point** | "kernel entry", "GPU function" | A method marked `[EntryPoint]` — callable from host, launches on device |
| **device function** | "device method", "helper kernel" | A method marked `[Kernel]` — callable only from other device code |
| **kernel** | — | Acceptable as shorthand for device code in general (CUDA usage) |
| **vectorized function** | "SIMD kernel" | CPU vector equivalent on AVX/NEON backends |
| **flavor** | "mode", "target", "platform" | The output backend: CUDA, OMP, HIP, AVX, AVX2, AVX512 |
| **work distribution** | "parallelism strategy", "thread model" | How work is split across blocks and threads |
| **data marshalling** | "serialization", "data copy", "memory transfer" | The host↔device data movement layer |

## Runtime types

| Preferred | Avoid | Notes |
|-----------|-------|-------|
| **`HybRunner`** | "the wrapper", "the launcher" | The generated host-side dispatch object |
| **`SatelliteLoader`** | "the loader", "DLL loader" | Finds and loads the satellite DLL at runtime |
| **`FloatResidentArray`** | "pinned array", "device array" | An array that resides primarily on the device to avoid repeated transfers |
| **`SetDistrib`** | "set grid", "configure kernel" | Sets the block/grid dimensions before a kernel call |
| **`Wrap`** | "proxy", "dispatch" | Creates the HybRunner dispatch wrapper around an object |

## Hardware terms

| Preferred | Avoid | Notes |
|-----------|-------|-------|
| **host** | "CPU side", "managed side" | The .NET runtime environment |
| **device** | "GPU side", "native side" | The CUDA GPU or other accelerator |
| **block** | "thread group", "work-group" | CUDA concept: a group of threads sharing memory |
| **thread** | "work-item" | CUDA concept: a single execution unit within a block |
| **warp** | — | 32 threads executing in lockstep on NVIDIA GPUs |
| **shared memory** | "local memory", "scratchpad" | Fast on-chip memory shared within a block (`__shared__` in CUDA) |
| **global memory** | "GPU RAM", "device memory" | Main GPU DRAM; high bandwidth, high latency |
| **coalesced access** | "aligned access" | Threads accessing consecutive addresses — the efficient GPU memory pattern |

## Disambiguation rules

**"CUDA" vs "Hybridizer CUDA backend":** When discussing code the developer writes, say "C# with Hybridizer". When discussing the output, say "CUDA backend" or "CUDA kernel (generated)". Don't say "the developer writes CUDA" — they don't.

**"compile" vs "transcode":** The Hybridizer CLI *transcodes* (or *generates*) — it produces `.cu`/`.cpp` source. Then `nvcc`/`g++` *compiles* that source. It's a two-step process. Using "compile" loosely for both steps is acceptable in conversational context, but be precise when explaining the pipeline.

**Free tool output:** The free NuGet tool emits `hybridizer.generated.cpp` containing a `char __hybridizer_cubin_module_data[]` blob — not readable CUDA source. The paid standalone emits real `.cu` files. Make this distinction when users ask to "read the generated CUDA code".
