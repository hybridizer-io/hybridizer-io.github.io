---
id: generics-virtuals-delegates
title: Generics, Virtuals & Delegates
description: How to use interfaces, virtual functions, generics, and delegates in Hybridizer code.
keywords: [Hybridizer, generics, virtual, interface, delegate, template, performance]
---

# Generics, Virtuals & Delegates

Object-oriented programming greatly facilitates code reuse and enables software patterns. The Hybridizer offers support for virtual functions and interfaces with no limitations, enabling the widest variety of code.

## Virtual Functions and Interfaces

The following code presents a typical example of virtual function and interface implementation:

```csharp
public interface ISimple
{
    int f();
}

public class Answer : ISimple
{
    [Kernel]
    public int f()
    {
        return 42;
    }
}

public class Other : ISimple
{
    [Kernel]
    public int f()
    {
        return 12;
    }
}
```

The `[Kernel]` attribute indicates to the build task that the function implementation should be used when scanning implementations of the interface.

:::warning
Virtual functions come with a significant performance penalty due to dispatch table lookups.
:::

## Generics as Templates

To overcome the virtual function overhead, the Hybridizer maps generics to C++ templates. The generated source code can then be inlined by the compiler, providing the flexibility of objects with high performance.

### Performance Comparison

| Implementation | GFLOPS | GCFLOPS | Usage |
|----------------|--------|---------|-------|
| Local (non-virtual) | 975 | 538 | 92% |
| Dispatch (virtual) | 478 | 263 | 45% |
| **Generics** | 985 | 544 | **93%** |
| Peak | 1174 | 587 | — |

*Benchmark: 13th degree Taylor expansion of expm1. Hardware: NVIDIA K20c with CUDA 6.0.*

## Constraint Interfaces

Template concepts in C++ are not explicitly expressed. In .NET, the concept is expressed by **constraints** on the generic type:

```csharp
[HybridTemplateConcept]
public interface IMyArray
{
    double this[int index] { get; set; }
}

[HybridRegisterTemplate(Specialize = typeof(MyAlgorithm<MyArray>))]
public struct MyArray : IMyArray
{
    double[] _data;
    
    [Kernel]
    public double this[int index]
    {
        get { return _data[index]; }
        set { _data[index] = value; }
    }
}

public class MyAlgorithm<T> where T : struct, IMyArray
{
    T a, b;
    
    [Kernel]
    public void Add(int n)
    {
        for (int k = threadIdx.x + blockDim.x * blockIdx.x;
             k < n;
             k += blockDim.x * gridDim.x)
        {
            a[k] += b[k];
        }
    }
}
```

### Key Attributes

| Attribute | Purpose |
|-----------|---------|
| `[HybridTemplateConcept]` | Marks an interface as a template constraint |
| `[HybridRegisterTemplate]` | Registers a template specialization |

With these attributes, virtual function calls are transformed into regular function calls, recovering performance by enabling inlining.

:::note
The generic type is not translated into a complete template — rather into a specialized implementation.
:::

## Known Limitations

- Generic methods are not supported
- Static methods of generic types are not supported
- Some combinations of generic types with vectorization (C++ targets like AVX) are not supported

## Next Steps

- [Intrinsics & Builtins](./intrinsics-builtins) — Hardware-specific features
- [Data Marshalling](./data-marshalling) — Passing data to kernels
