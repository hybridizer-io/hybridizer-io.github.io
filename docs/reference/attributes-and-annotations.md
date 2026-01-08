---
id: attributes-and-annotations
title: Attributes & Annotations Reference
description: Complete reference of all Hybridizer attributes for marking kernels and configuring code generation.
keywords: [Hybridizer, attributes, annotations, EntryPoint, Kernel, intrinsic]
---

# Attributes & Annotations Reference

This reference documents all Hybridizer attributes used to control code generation and kernel behavior.

## Core Attributes

### `[EntryPoint]`

Marks a method as a GPU entry point (CUDA `__global__`).

```csharp
[EntryPoint]
public static void MyKernel(float[] data, int N)
{
    // ...
}
```

| Property | Type | Description |
|----------|------|-------------|
| `Name` | string | Override generated function name |

### `[Kernel]`

Marks a method as a device function (CUDA `__device__`).

```csharp
[Kernel]
public static float Helper(float x)
{
    return x * x;
}
```

Kernels are called from entry points or other kernels.

## Template & Generics Attributes

### `[HybridTemplateConcept]`

Marks an interface as a C++ template concept for performance.

```csharp
[HybridTemplateConcept]
public interface IArray
{
    float this[int i] { get; set; }
}
```

### `[HybridRegisterTemplate]`

Registers a template specialization.

```csharp
[HybridRegisterTemplate(Specialize = typeof(MyAlgorithm<MyArray>))]
public struct MyArray : IArray
{
    // ...
}
```

| Property | Type | Description |
|----------|------|-------------|
| `Specialize` | Type | Type to specialize for |

## Intrinsic Attributes

### `[IntrinsicConstant]`

Maps a property to a backend constant.

```csharp
public static class ThreadIdx
{
    [IntrinsicConstant("threadIdx.x")]
    public static int X { get; }
}
```

### `[IntrinsicFunction]`

Maps a method to a backend function.

```csharp
[IntrinsicFunction("__syncthreads")]
public static void SyncThreads() { }

[IntrinsicFunction("__expf")]
public static float FastExp(float x) => 0;
```

## Memory Attributes

### `[SharedMemory]`

Declares shared memory allocation.

```csharp
[SharedMemory(typeof(float), 256)]
public static float[] SharedBuffer;
```

### `[ConstantMemory]`

Declares constant memory (read-only, cached).

```csharp
[ConstantMemory]
public static readonly float[] LookupTable;
```

## Configuration Attributes

### `[HybridVectorWidth]`

Specifies vector width for AVX backends.

```csharp
[HybridVectorWidth(8)]  // AVX-256 for float
public class MyProcessor { }
```

### `[InlineHint]`

Suggests inlining to the backend compiler.

```csharp
[Kernel]
[InlineHint(InlinePolicy.Always)]
public static float FastOp(float x) => x * x;
```

| Value | Behavior |
|-------|----------|
| `Always` | Force inline |
| `Never` | Prevent inline |
| `Default` | Compiler decides |

## Assembly-Level Attributes

### `[HybridAssembly]`

Configures assembly-wide settings.

```csharp
[assembly: HybridAssembly(
    GenerateLineInfo = true,
    TargetFlavors = new[] { "CUDA", "AVX" }
)]
```

## Attribute Summary Table

| Attribute | Target | Purpose |
|-----------|--------|---------|
| `[EntryPoint]` | Method | Mark as kernel entry |
| `[Kernel]` | Method | Mark as device function |
| `[HybridTemplateConcept]` | Interface | Template constraint |
| `[HybridRegisterTemplate]` | Struct/Class | Template specialization |
| `[IntrinsicConstant]` | Property | Map to backend constant |
| `[IntrinsicFunction]` | Method | Map to backend function |
| `[SharedMemory]` | Field | Shared memory allocation |
| `[ConstantMemory]` | Field | Constant memory |
| `[HybridVectorWidth]` | Class | Vector width hint |
| `[InlineHint]` | Method | Inlining control |

## Unsupported Patterns

The following patterns are not supported with attributes:

| Pattern | Reason |
|---------|--------|
| `async/await` | No async on device |
| `dynamic` | No runtime dispatch |
| `string` operations | No heap allocation |
| `ref struct` parameters | Marshalling limitations |

## Next Steps

- [CLI Options](./cli-options) — Command-line reference
- [API Index](./api-index) — API overview
- [Intrinsics & Builtins](../guide/intrinsics-builtins) — Intrinsic details
