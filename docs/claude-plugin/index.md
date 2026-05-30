---
id: index
title: Claude Plugin
description: A Claude Code plugin that migrates C# codebases to CUDA using Hybridizer — slash commands, subagents, porting log, and an 8-step methodology.
keywords: [Hybridizer, Claude, plugin, Claude Code, CUDA, porting, migration]
---

# Claude Plugin for Hybridizer

The **hybridizer-port** plugin turns Claude Code into a Hybridizer porting assistant. It bundles a curated skill, three subagents, five slash commands, and an optional porting log — all distilled from a real port (TinyLlama Q8_0, ~125 tok/s on RTX 5070).

**Source:** [github.com/hybridizer-io/claude-plugin](https://github.com/hybridizer-io/claude-plugin)

---

## Requirements

- **Claude Code** — any recent version with plugin/marketplace support.
- **Python 3** in `PATH` — required only if you enable the optional porting log (`/hybridizer-log on`). The rest of the plugin works without it.
- **Hybridizer** installed — if you're actually running a port. The plugin references it but doesn't require it at install time.

---

## Install

### From the marketplace (recommended)

```bash
/plugin marketplace add hybridizer-io/claude-plugin
/plugin install hybridizer-port@hybridizer
```

### From a local clone (for development)

```bash
git clone https://github.com/hybridizer-io/claude-plugin /path/to/claude-plugin
/plugin marketplace add /path/to/claude-plugin
/plugin install hybridizer-port@hybridizer
```

---

## Usage

The skill **auto-loads** when Claude detects a Hybridizer context — references to `[Kernel]` / `[EntryPoint]`, Hybridizer transcoder build errors, `HybRunner`, `SatelliteLoader`, etc. You can also invoke it explicitly with the slash commands below.

### Slash commands

| Command | What it does |
|---------|-------------|
| `/hybridizer-init` | Scaffolds `Directory.Build.props` with `$(HybridizerTool)` and wires a hello-world kernel into an existing C# project |
| `/hybridizer-port <file-or-method>` | Proposes a kernel for a target function (dispatches to `hybridizer-porter`) |
| `/hybridizer-review` | Checks a candidate kernel against the gotchas list before you compile (dispatches to `hybridizer-reviewer`) |
| `/hybridizer-profile` | Sets up profiling — in-process on WSL, `nsys` on Windows |
| `/hybridizer-log on\|off` | Toggles the optional porting log |

### Subagents

Three subagents handle specific tasks automatically when invoked via a slash command:

- **hybridizer-porter** — analyses a C# function and proposes a kernel shape (parallelism, atomics, shared memory).
- **hybridizer-reviewer** — checks a candidate kernel against the known gotchas list before you spend time compiling.
- **hybridizer-builder** — discovers the Hybridizer CLI (global tool / local manifest / custom path), reads `--display-license-details` for supported flavors, invokes the transcoder, and parses errors into actionable messages.

### Porting log (opt-in)

```bash
/hybridizer-log on
```

Creates a `.hybridizer-log-enabled` flag in your project. Every subsequent porting prompt is appended to `porting-to-hybridizer.md` — useful for tracking decisions across a long port. Disable with `/hybridizer-log off`.

---

## The 8-step migration methodology

The plugin encodes an ordered migration workflow. Don't skip steps — the ordering is load-bearing.

1. **Lock down behavior with tests.** Unit tests + at least one integration test before touching anything.
2. **Profile the managed code.** Measure where time is actually spent. Don't guess.
3. **Audit and refactor for Hybridizer compliance.** Before writing any kernel: eliminate heap allocations on hot paths, remove `throw`/`catch`, `foreach`, `lock`, `string`, recursion, and captured lambdas from code that will become device code. The codebase still runs on CPU at this stage.
4. **Stand up the build infrastructure.** `Directory.Build.props` with `$(HybridizerTool)`, MSBuild targets, `SatelliteLoader.Load()`, and a trivial smoke-test kernel to prove the round-trip works. This is plumbing, not optimization.
5. **Port a single kernel to OMP and CUDA, and unit-test both.** OMP first (or in parallel) — it catches transcoding bugs without GPU debug latency.
6. **Eliminate host↔device memcopies.** Promote buffers to `FloatResidentArray`, mark parameters `[In]`/`[Out]`, fuse kernels. Goal: eliminate data motion, not peak kernel speed.
7. **Profile GPU execution.** Only now does the GPU profile give meaningful signal — before step 6, memcopies drown everything else.
8. **Improve performance gradually**, guided by profiles. Shared-memory tiling, occupancy tuning, warp-level primitives, kernel fusion.

:::caution Performance regressions between steps 3 and 6 are expected
When porting, simplified scalar shapes replace hand-tuned managed code (AVX2, `Parallel.For`, etc.) so the transcoder can handle them. The regression is the price of getting the shape right. Only investigate *functional* regressions during this phase. Performance comes back at steps 7 and 8.
:::

**The hard invariant:** after every change, rerun all tests. If anything breaks, stop and investigate immediately — don't push through.

---

## House rules

These are the highest-priority facts the plugin encodes. Read them even if you skip everything else.

1. **`MathF.*` does not exist in Hybridizer's builtins.** Use `(float)System.Math.X(double)` in kernels. `MathF.*` aborts the transcoder with error `0X60AC` and emits empty `.cu` files.

2. **Do not use `AtomicExpr.apply`** — flagged buggy. Use `[IntrinsicFunction("atomicAdd")]` / `("atomicMax")` stubs over a static `Atomics` class instead.

3. **Pick the right edition and let `--display-license-details` decide the flavors.** The free `Hybridizer` NuGet tool is CUDA-only (full profiling/debugging — not stripped down). The paid standalone adds OMP, HIP, AVX, AVX2, AVX512. Only pass `--flavors` what the license reports as available.

4. **Initialise `FloatResidentArray` eagerly.** Touch `DevicePointer` and set `Status = HostNeedsRefresh` before the first kernel write. Otherwise the kernel reads uninitialised device memory.

5. **Garbled CUDA output is usually a stale `.dll` satellite.** Clean rebuild before reading kernel output.

6. **`nsys`/`compute-sanitizer` wedge `dotnet` on WSL2.** Profile on native Windows or use the in-process profiler.

7. **Disable xUnit parallelism for Hybridizer tests.** Concurrent kernel launches corrupt `NativeSerializer`'s internal dictionary.

---

## Plugin layout

```
claude-plugin/
├── .claude-plugin/
│   └── marketplace.json
├── hybridizer-port/              ← the plugin
│   ├── .claude-plugin/
│   │   └── plugin.json
│   ├── skills/hybridizer-port/
│   │   ├── SKILL.md              ← index; loads reference files on demand
│   │   └── references/
│   │       ├── methodology.md
│   │       ├── attributes.md
│   │       ├── device-code.md
│   │       ├── build-pipeline.md
│   │       ├── host-launch.md
│   │       ├── kernel-patterns.md
│   │       ├── reductions.md
│   │       ├── graph-capture.md
│   │       ├── perf-tuning.md
│   │       ├── gotchas.md
│   │       └── samples-index.md
│   ├── agents/
│   │   ├── hybridizer-porter/
│   │   ├── hybridizer-reviewer/
│   │   └── hybridizer-builder/
│   ├── commands/
│   │   ├── hybridizer-init/
│   │   ├── hybridizer-port/
│   │   ├── hybridizer-review/
│   │   ├── hybridizer-profile/
│   │   └── hybridizer-log/
│   └── hooks/
├── README.md
└── LICENSE (MIT)
```

The skill uses **progressive disclosure**: `SKILL.md` is a short index; Claude pulls in the relevant reference file only when the task touches that topic.

---

## What the skill covers

The plugin's knowledge base spans the full porting surface:

- Attributes: `[Kernel]`, `[EntryPoint]`, `[IntrinsicFunction]`, `[HybridTemplateConcept]`, `[HybridConstant]`, `[In]`/`[Out]`, `[LaunchBounds]`, `[HybridizerIgnore]`
- Device-code restrictions: language subset, `threadIdx`/`__syncthreads`, `SharedMemoryAllocator`, atomic intrinsics
- Build pipeline: NuGet tool install shapes, `$(HybridizerTool)` in `Directory.Build.props`, MSBuild target order, flavor detection
- Host launch: `HybRunner`, `SatelliteLoader.Load`, `SetDistrib`/`Wrap`, streams, `FloatResidentArray`
- Kernel patterns: cooperative blocks, warp shuffle, CUB BlockReduce, quantized split layout, per-flavor `[EntryPoint]` split
- Reduction skeletons and the four op-delivery styles
- CUDA graph capture: capture mode, pre-warm, legacy↔capture dependency rules
- Performance tuning: `DllImport` bypass for hot kernels, `extern "C"` wrappers in `intrinsics.cuh`, memcopy reduction strategy
- Gotchas: known bugs and edge cases from real ports

---

## Status

Early release — content distilled from a working port. Expect iteration. Issues and PRs welcome on [GitHub](https://github.com/hybridizer-io/claude-plugin).
