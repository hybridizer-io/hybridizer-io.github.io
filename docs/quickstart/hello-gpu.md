---
id: hello-gpu
title: Hello GPU (C#)
---

# Your First "Hello GPU!"

Let's run your first piece of accelerated code. This simple example adds two vectors together, a classic parallel computing task.

### The C# Code

Here is a simple C# console application. The `Add` method is the one we want to run on the GPU. We mark it with the `[EntryPoint]` attribute.

```csharp
using Hybridizer.Runtime.CUDAImports;

namespace HelloWorld
{
    class Program
    {
        [EntryPoint]
        public static void Add(float[] a, float[] b, int n)
        {
            for (int i = 0; i < n; ++i)
            {
                a[i] += b[i];
            }
        }

        static void Main(string[] args)
        {
            const int n = 1024;
            float[] a = new float[n];
            float[] b = new float[n];
            
            // Initialize arrays...

            // Create a cuda thread
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            dynamic wrapper = HybRunner.Cuda().SetDistrib(prop.multiProcessorCount * 16, 128);
            
            // Run on GPU
            wrapper.Add(a, b, n);

            // Verify results...
        }
    }
}
```

Minimal example: vector add in C# compiled to a CUDA kernel.

Steps:

1. Create console project and reference Hybridizer packages.
2. Mark entry method with the required attribute (e.g., `[EntryPoint]`).
3. Build to generate CUDA code and compile.
4. Invoke generated kernel from host.
Notes:

- Show kernel launch configuration and memory copy.
- Verify output and compare to CPU version.

See: Programming Guide → Invoke Generated Code.
