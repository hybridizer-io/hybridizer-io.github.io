# Programming Guide

<a name="workdistribution"></a>
## Work Distribution

The first concept of the Hybridizer is work distribution. It is the description of the way tasks are distributed on hardware execution. We present in this chapter several approaches with different level of control on the distribution of work. Not all hardware architectures support all distribution concepts, but the most basic ones are.

The key element of work distribution is the *entrypoint*. An entrypoint is a method called from a single execution unit that spawns a work grid on a device. The the work-grid: a work grid is composed of work groups, and work group is composed of workitem. We map these concepts this way:

| CUDA | OpenCL   | Hybridizer Vector                 |
|------|----------|-----------------------------------|
|block |work-group|thread (stack frame)               |
|thread|work-item |vector entry (within a vector unit)|

Experiments on various platforms, illustrate that this concept mapping delivers best performances. This mapping also allows for a single version of the source code.
The distribution of operations on a device or accelerator may be done automatically using parallel concepts or explicitly leaving more control to the user. Both approach have benefits and drawbacks as illustrated in this chapter.

### Explicit Work Distribution
Explicit work distribution reuses the concepts of CUDA. In that case, `threadIdx` and `blockIdx` are used to locate the working entity. The developer is free to change the naming of theses elements, indeed the work distribution uses a set of intrinsics to map concepts.

```csharp
[EntryPoint]
public void square(int count, double[] a, double[] b)
{
    for (int k = threadIdx.x + blockDim.x * blockIdx.x ; 
        k < count ; k += blockDim.x * gridDim.x)
    {
        b[k] = a[k] * a[k];
    }
}
```

The `block` dimensions concepts map on the Multithreading challenge, where the `thread` dimension concept maps on the Vectorization challenge. The mapping is perfectly aligned with CUDA allowing a vast majority of code already designed for CUDA to be usable without redesign.

### Parallel-For Constructs

Similarly to `Console.Out.Writeline`, the `Parallel.For` static method is mapped to an internal implementation. It may be used within an entry point or a kernel. By default, the loop will iterate over blocks and threads on CUDA implementation. The following listing illustrates its use:

```csharp
[EntryPoint]
public static void RunParallelFor(int[] input, int[] output, int size)
{
    Parallel.For(0, size, i => output[i] = input[i] + 1);
}
```

The third parameter is an action, and can also hold local data.

<a name="interfacesandvirtual"></a>
## Interface and Virtual Functions
In this chapter, we describe how the Hybridizer implements virtual functions, in their definition in object oriented programming, and interfaces, in their definition in Java or C#. Object oriented programming has been a breakthrough in software engineering. It greatly facilitates code reuse and eases the implementation of software patterns.

The Hybridizer offers support for virtual functions in a natural way, with no limitation, hence enabling the widest variety of code.

### Virtual Function Example

The following code presents a trivial example of virtual function and interface implementation.

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
        return 42 ;
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

The Kernel attribute indicates the pre--processing phase (a.k.a. build task) that the function implementation should be used when scanning implementations of the interface.

## Generics
Virtual functions come with a significant performance penalty. In order to overcome this, we map generics to templates. The generated source code can then be inlined by the compiler, and the flexibility of objects can still be used in source code, with performance. 

Measuring performances of virtual function calls on a small example, we measured the impact of using virtual functions over regular local functions. 
The benchmark we use is our Expm1 benchmark: it is composed of a 13<sup>th</sup> degree of Taylor expansion of the exponential minus one. The argument values were loaded from an array, and the result of the computation was stored in another array.

We implemented three versions: 

* Local: a non virtual implementation of an array class, implementing `this` property. The method was local and could be inlined. Though this implementation did not allow change in the array implementation.
* Dispatch: an implementation using an interface for the `this` property, hence invoking a dispatch function, that is a look--up in a dispatch table for the function pointer, enabling more flexibility in the implementation.
* Generics: the same implementation as Dispatch, adding constraints to allow for template code generation.

| Expm1   | GFLOPS    | GCFLOPS   | Usage   |
|---------|-----------|-----------|---------|
|Local    | 975       | 538       | 92%     |
|Dispatch | 478       | 263       | 45%     |
|Generics | 985       | 544       | 93%     |
|Peak     | 1174      | 587       | -       |
Comparative performance using generics for implementation over dispatching calls. GFLOPS account for 10<sup>9</sup> floating-point operations per seconds, GCFLOPS account for 10<sup>9</sup> floating-point complex instructions - counting 1 for fused multiply and add. Usage is the performance ratio with theoretical peak of the hardware. Hardware is NVIDIA K20c with CUDA 6.0.

### Constraint Interfaces
Template concepts in C++ are not explicitly expressed, as the compiler tells whether the type is compliant with its usage or not. 
In dot net and Java, the concept is expressed by constraints on the generic type. The following example illustrates this idea.

```csharp
[HybridTemplateConcept]
public interface IMyArray {
  double this[int index] { get; set; }
}
[HybridRegisterTemplate(Specialize=typeof(MyAlgorithm<MyArray>))]
public struct MyArray : IMyArray
{
  double[] _data;
  [Kernel] public double this[int index] {
    get { return _data[index]; } 
    set { _data[index] = value; } 
  }
}
public class MyAlgorithm<T> where T : struct, IMyArray
{
  T a, b;
  [Kernel] public void Add(int n) {
    for (int k = threadIdx.x + blockDim.x * blockIdx.x; 
      k < n; k += blockDim.x * gridDim.x)
      a[k] += b[k];
  }
}
```

In this example, the `IMyArray` interface describes the constraints. The generic class `MyAlgorithm` will be specialized by array type. In order to use functions on the type used, we need to specify that the generic parameter type implements some function. This is done with the constraint interface: we demand that the generic type implements `IMyArray`. Without annotations, this translates into a virtual function call. 

To override this behavior, and reduce the execution time overhead of virtual functions, we add an attribute on the interface: `HybridTemplateConcept` attribute. Then, in order to register the template specialization, that is actually implementing the type so that the function pointer may be used for the \verb!Kernel! defined in the generic class, we need to add a `HybridRegisterTemplate` either on the generic type or on the array type (as in our example).

Doing the above, we transform virtual function calls into regular function calls and recover performances enabling inlining and other optimization the target compiler is capable of.
Note however that the generic type is not translated into a complete template, rather in an implementation of the template. It behaves as a template for which no template code is available but for which specialized versions are available.

This feature might not be available on all flavors.


### Known limitations
As of today, it is not possible to use Generic Methods, nor is it possible to use static methods of Generic Types.