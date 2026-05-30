---
id: task-examples
title: LLM Task Examples
description: Structured answer templates for common Hybridizer questions — install, porting, debugging, performance.
---

# LLM Task Examples

Structured answer templates for common questions. Each example shows the question type, the correct answer shape, and what to cite or warn about.

---

## "How do I install Hybridizer?"

**Answer shape:** prerequisites → install command → verify → next step.

```
Prerequisites: .NET 8 SDK, NVIDIA CUDA Toolkit, NVIDIA GPU with compatible drivers, C++ toolchain (MSVC on Windows, GCC on Linux).

Install the free NuGet tool:
  dotnet tool install -g Hybridizer
  dotnet new install Hybridizer.App.Template
  dotnet new hybridizer-app -n MyProject
  cd MyProject

Verify:
  nvidia-smi       # GPU detected?
  nvcc --version   # CUDA toolkit installed?
  dotnet build     # Project builds?
  dotnet run       # Kernel runs?
```

**Cite:** `quickstart/install`, `tutorials/setup`
**Note:** If the user needs OMP/AVX flavors or wants to read the generated `.cu` source, they need the paid standalone, not the NuGet tool.

---

## "How do I port a C# method to GPU?"

**Answer shape:** annotate → build → verify → launch.

1. Add `[EntryPoint]` to the method. Ensure it's in a non-static class.
2. Replace `MathF.*` with `(float)System.Math.X(double)` in the method body.
3. Replace any `foreach` / `throw` / closures with Hybridizer-compatible constructs.
4. Run `dotnet build` with `<CompileCUDA>enable</CompileCUDA>` — this triggers the Hybridizer CLI and `nvcc`.
5. Call `HybRunner.Cuda("MyProject_CUDA.dll").Wrap(new MyClass()).SetDistrib(grid, block).MyMethod(args)` from the host.

**Cite:** `guide/concepts`, `guide/compilation-pipeline`, `quickstart/hello-gpu`
**Warn:** `MathF.*` in kernel bodies aborts the transcoder with error `0X60AC` (empty `.cu` files). Use `(float)System.Math.*` instead.

---

## "Why is my kernel output wrong / all zeros / NaN?"

**Diagnosis checklist** (in order of likelihood):

1. **Stale satellite DLL** — `bin/` has an old `.dll`. Clean-rebuild: `dotnet clean && dotnet build`.
2. **Uninitialised `FloatResidentArray`** — call `RefreshDevice()` before the first kernel write.
3. **Missing `[In]`/`[Out]`** — without them, arrays are copied both ways by default, but if you use `FloatResidentArray` and forget to push first, the device side is zeroed.
4. **Out-of-bounds thread** — grid-stride loop missing; last block overshoots `n`. Add `if (i < n)` guard.
5. **`AtomicExpr.apply` usage** — replace with `[IntrinsicFunction]` stubs.
6. **Race condition** — `__syncthreads()` (`ThreadFunctions.SyncThreads()`) missing between shared memory write and read.

**Cite:** `quickstart/run-and-debug`, `quickstart/faq-troubleshooting`

---

## "My build fails — Hybridizer emits an empty `.cu` file"

Almost always `MathF.*` in a kernel body.

**Steps:**
1. Search the kernel file for `MathF.` — replace every occurrence with `(float)System.Math.X((double)arg)`.
2. Check for error `0X60AC` in the build output — this is the Hybridizer transcoder's `MathF` signal.
3. Clean and rebuild.

If the file is still empty after removing `MathF`, check for other unsupported constructs: `foreach` over non-array, `try`/`catch`, `lock`, string operations, or recursive calls through a non-`[Kernel]` method.

**Cite:** `quickstart/faq-troubleshooting`, `llm/code-patterns` (pattern 2)

---

## "CUDA vs AVX — which flavor should I use?"

**Decision table:**

| Criterion | CUDA | AVX/AVX2/AVX512 |
|-----------|------|-----------------|
| Hardware required | NVIDIA GPU | Any x86 CPU |
| Peak throughput (compute-bound) | Very high (thousands of threads) | Moderate (8–16 wide SIMD) |
| Data transfer overhead | Host↔device copies | None (shared memory) |
| Best for | Large parallel workloads, matrix ops, deep learning inference | Embarrassingly parallel CPU loops, legacy codebases without GPU |
| Flavor availability | Free NuGet tool + paid standalone | Paid standalone only |

**Rule of thumb:** choose CUDA if the working set fits in GPU memory and the workload has ≥ 10k independent operations. Choose AVX if the workload is memory-bandwidth-bound on the CPU and you want zero-copy access to host data.

**Cite:** `platforms/overview`, `platforms/cuda`, `platforms/vector-avx-neon`

---

## "How do I avoid copying data to/from the GPU on every kernel call?"

**Answer:** use `FloatResidentArray` and mark parameters `[In]`/`[Out]`.

1. Replace `float[]` with `FloatResidentArray` for buffers reused across calls.
2. Mark read-only inputs `[In]` and write-only outputs `[Out]` to skip the unnecessary direction.
3. Call `RefreshDevice()` once before the first kernel, `RefreshHost()` once after the last.

**Cite:** `guide/data-marshalling`, `howto/manage-memory`
**See also:** code pattern 4 in `llm/code-patterns`

---

## "How do I debug a Hybridizer kernel?"

**Steps:**

1. Enable `--generate-line-info` in the Hybridizer CLI invocation — this embeds C# line numbers in the CUDA binary, enabling stepping in Nsight.
2. Build in `Debug` configuration with `<CompileCUDA>enable</CompileCUDA>`.
3. On Windows: use Nsight Visual Studio Edition (attach to process, switch to CUDA thread view).
4. On Linux/WSL: use `cuda-gdb` (`cuda-gdb --args dotnet MyProject.dll`).
5. For functional issues without a debugger: add an OMP flavor build — OMP kernels run on CPU and can be stepped through with a regular .NET debugger.

**Warn:** `nsys`/`compute-sanitizer` hang `dotnet` on WSL2. Profile/sanitize on native Windows, or use the in-process Hybridizer profiler hooks.

**Cite:** `guide/line-info-and-debug`, `quickstart/run-and-debug`

---

## "How do I profile a Hybridizer kernel?"

**Answer shape:** quick profiling → Nsight → in-process.

```
Quick (Windows):
  nsys profile dotnet MyProject.dll
  ncu --target-processes all dotnet MyProject.dll

Quick (Linux, native — not WSL2):
  nsys profile dotnet MyProject.dll

In-process (cross-platform):
  Use CudaRuntimeHelpers.cudaEventRecord() / cudaEventSynchronize() from
  Hybridizer.Runtime.CUDAImports to time kernel execution from C#.
```

**What to look for:**
- **High memcpy time** → add `FloatResidentArray`, use `[In]`/`[Out]`
- **Low SM occupancy** → increase block size (try 256 or 512), check register pressure
- **Memory bandwidth not saturated** → check for non-coalesced access (consecutive threads should access consecutive addresses)

**Cite:** `howto/optimize-kernels`, `cuda/perf-metrics`, `cuda/memory-and-profiling`

---

## "How do I set up CI/CD for a Hybridizer project?"

**Answer shape:** what CI needs → Docker image → cache strategy.

CI needs: a Linux runner with NVIDIA drivers + CUDA toolkit + .NET 8 SDK. On GPU-less CI (GitHub Actions free tier), you can run the Hybridizer transcoder step (which doesn't need a GPU) but skip actual kernel execution by guarding tests with `nvidia-smi` detection.

```yaml
# GitHub Actions example
- name: Check GPU availability
  id: gpu
  run: echo "has_gpu=$(nvidia-smi &>/dev/null && echo true || echo false)" >> $GITHUB_OUTPUT

- name: Build (always)
  run: dotnet build -c Release

- name: Run GPU tests (GPU runners only)
  if: steps.gpu.outputs.has_gpu == 'true'
  run: dotnet test --filter Category=GPU
```

**Cite:** `howto/ci-cd-integration`
