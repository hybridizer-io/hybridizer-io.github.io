---
id: what-is-hybridizer
title: What is Hybridizer?
description: Overview of Hybridizer, how it operates, inputs, outputs, and supported platforms.
keywords: [Hybridizer, compiler, MSIL, CUDA, AVX, OMP, .NET]
---

# What is Hybridizer?

The aim of the Hybridizer is to let developers seamlessly use different hardware execution environments. It integrates in the compilation toolchain: from an intermediate language, it generates source code for different types of architectures. The Hybridizer abstracts the specific SDK and language features of processors, hence reducing the learning curve to many-core processor enablement.

## Hybridizer in Operation

The Hybridizer operates on **intermediate language** — code that has been compiled to be either executed by a virtual machine or compiled to machine code. The supported input intermediate languages are:

| Input Language | Description |
|----------------|-------------|
| **MSIL** | Microsoft Intermediate Language — the .NET platform |
| **LLVM-IR** | The intermediate representation of LLVM |

Then, depending on the selected **Flavor** (see [Platforms & Flavors](/docs/platforms/overview)), the Hybridizer generates source code with all the necessary annotations and code hints to make use of the specific features of each hardware architecture.

![Hybridizer Overview](../images/what-is-hybridizer.png)

From a single version of the source intermediate language, **several platforms may be targeted**.

## Key Concepts

- **Single-source**: Write idiomatic C#; Hybridizer compiles MSIL to native code.
- **Multiple backends**: CUDA kernels, OpenMP+CUDA, and vector backends (AVX/AVX512/NEON/POWER).
- **Performance**: Leverages backend-specific optimizations (e.g., SIMT on GPU, SIMD on CPU).
- **Interoperability**: Generated code is callable from your .NET host.

## Known Limitations

As of today, the following constructs are **not supported** within the code to be transformed:

- Allocating class instances (heap data)
- `string` type
- Catching and throwing exceptions (only partially supported)
- `lock` regions
- `foreach` (as it uses `try/catch` and heap-related operations)
- Recursion (though similar features may be achieved using interfaces)
- Generic functions (generic types are supported)
- Some combinations of generic types with vectorization (C++ targets such as AVX)

:::note
These limitations only apply to the code that will be transformed to accelerator code. Your host code has no such restrictions.
:::

## Next Steps

- [Architecture Overview](./architecture) — Understand the end-to-end pipeline
- [Quickstart](../quickstart/install) — Get started in minutes
- [Programming Guide](../guide/concepts) — Learn core concepts
