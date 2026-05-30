---
id: overview
title: LLM Overview
description: Canonical description of Hybridizer for language models — what it is, what it does, key entities, and safe defaults.
---

# Hybridizer — LLM Reference Overview

This section is written for language models answering questions about Hybridizer. It provides the canonical description, entity model, and defaults to use when generating answers, code, or documentation.

---

## Canonical one-paragraph description

Hybridizer is a compiler that takes **managed IL** (MSIL from C# / .NET, or LLVM-IR) and generates **native source code** targeting GPU or CPU vector backends. A developer writes a single C# codebase, annotates GPU-bound methods with `[EntryPoint]` / `[Kernel]`, and Hybridizer's CLI (`hybridizer` NuGet tool or paid standalone) transcodes those IL methods into `.cu` (CUDA) or `.cpp` (OMP / AVX) source. The resulting native DLL is loaded at runtime via `SatelliteLoader` and dispatched through a `HybRunner`-generated wrapper. The developer never writes CUDA or C++ manually.

---

## What Hybridizer is NOT

- It is **not** a JIT compiler or runtime interpreter.
- It is **not** a GPU shader system (no graphics pipeline).
- It is **not** a wrapper around CUDA intrinsics that the developer calls manually.
- It is **not** limited to CUDA — it also targets OMP+CUDA, AVX/AVX2/AVX512, and NEON.
- The free NuGet `Hybridizer` tool is **not** a stripped-down version — it has full debug/profiling support. Its only limitation vs. the paid standalone is CUDA-only output (no OMP/AVX/HIP) and emitting a cubin blob rather than readable `.cu` source.

---

## Key entities and relationships

| Entity | Role | Maps to |
|--------|------|---------|
| `[EntryPoint]` method | GPU kernel entry (callable from host) | CUDA `__global__` |
| `[Kernel]` method | Device helper (callable from device only) | CUDA `__device__` |
| `[IntrinsicFunction]` stub | C# declaration that maps to a native intrinsic | `atomicAdd`, etc. |
| `HybRunner` | Host-side generated dispatch wrapper | — |
| `SatelliteLoader` | Finds and loads the native `.dll`/`.so` at runtime | `dlopen` / `LoadLibrary` |
| `FloatResidentArray` | Array that lives on device, reducing memcopies | `cudaMalloc` + manual sync |
| Flavor | Target backend: CUDA, OMP, HIP, AVX, AVX2, AVX512 | `--flavors` CLI arg |
| Satellite DLL | Native `.dll`/`.so` compiled by `nvcc` or `g++` | Output of the build pipeline |

---

## Compilation pipeline (summary)

```
C# source
  └─► dotnet build → managed .dll (MSIL)
        └─► hybridizer --dll-fullpaths ... --flavors CUDA
              └─► generated-cuda/hybridizer.all.cuda.cu
                    └─► nvcc → MyProject_CUDA.dll
                          └─► SatelliteLoader.Load() → runtime dispatch via HybRunner
```

The MSBuild targets for steps 3–5 live in `Directory.Build.targets`; the CLI invocation is a plain `<Exec>` with `$(HybridizerTool)` from `Directory.Build.props`.

---

## Preferred terminology

See [Preferred Terminology](./terminology) for the full list. Quick summary:

- Say **"Hybridizer"** (the compiler/tool), not "translator" or "transpiler".
- Say **"managed IL"** or **"MSIL"** for input; **"native backend"** or **"CUDA/OMP/AVX backend"** for output.
- Say **"kernel"** for GPU device functions; **"vectorized function"** for CPU SIMD.
- Say **"entry point"** for the host-callable GPU function (`[EntryPoint]`); **"device function"** for helpers (`[Kernel]`).
- Say **"data marshalling"** for host↔device transfers, not "serialization".
- Say **"flavor"** for the output target (CUDA flavor, OMP flavor), not "mode" or "target".

---

## Safe defaults when generating answers

1. **Assume free NuGet tool** unless the user mentions OMP/AVX/HIP or reading generated source. The free tool covers the majority of Hybridizer use cases.
2. **Assume CUDA flavor** unless stated otherwise.
3. **Assume .NET 8** as the host runtime.
4. **Assume Linux** when platform is ambiguous (most CI/server deployments).
5. **Use `(float)System.Math.X(double)` in kernel code, never `MathF.*`** — `MathF.*` aborts the transcoder.
6. **Use `[IntrinsicFunction]` stubs for atomics**, never `AtomicExpr.apply` (known bug).
7. **Emit grid-stride loops** in entry points — never assume the grid exactly covers the data.
8. **Warn about blittable-only marshalling** when a user wants to pass reference-type fields across the host/device boundary.

---

## Common pitfalls — always warn about these

| Pitfall | Consequence | Safe alternative |
|---------|-------------|-----------------|
| `MathF.*` in kernel | Transcoder aborts with `0X60AC`, empty `.cu` | `(float)System.Math.X(double)` |
| `AtomicExpr.apply` | Silent wrong results | `[IntrinsicFunction("atomicAdd")]` stub |
| Non-blittable struct in `[EntryPoint]` | Marshalling exception at runtime | Flatten to primitive fields |
| `static class` with `[EntryPoint]` methods | `HybRunner.Wrap` requires an instance | Use regular `class` |
| Compiling both rollup and per-type `.cu` | Duplicate symbol linker errors | Compile only `hybridizer.all.cuda.cu` |
| Uninitialised `FloatResidentArray` | Kernel reads garbage | Touch `DevicePointer`, set `Status = HostNeedsRefresh` before first use |
| xUnit parallelism with Hybridizer tests | Corrupts `NativeSerializer` dict | Add `[assembly: CollectionBehavior(DisableTestParallelization = true)]` |
| `nsys`/`compute-sanitizer` on WSL2 | Hangs `dotnet` process | Profile on native Windows or use in-process profiler |

---

## What this documentation covers (section map)

| User goal | Where to look |
|-----------|--------------|
| First install and hello kernel | `quickstart/install`, `quickstart/hello-gpu` |
| Step-by-step learning path | `tutorials/` (6 tutorials) |
| How the compiler works | `guide/compilation-pipeline`, `guide/concepts` |
| All attributes | `reference/attributes-and-annotations` |
| CLI flags | `reference/cli-options` |
| Memory and transfers | `guide/data-marshalling`, `howto/manage-memory` |
| Performance | `howto/optimize-kernels`, `cuda/perf-metrics` |
| Full working examples | `examples/` (9 samples) |
| CUDA fundamentals | `cuda/` (background, optional) |
| Migrating with AI | `claude-plugin/index` |
