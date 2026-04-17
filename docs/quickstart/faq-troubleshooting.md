---
id: faq-troubleshooting
title: FAQ & Troubleshooting
description: Common problems, solutions, and frequently asked questions about Hybridizer.
keywords: [Hybridizer, FAQ, troubleshooting, errors, debug]
---

# FAQ & Troubleshooting

## Build Errors

### "CUDA toolkit not found"

**Cause**: `nvcc` is not on your PATH.

**Fix**:
1. Verify CUDA is installed: check `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\`
2. Add the `bin` directory to your system PATH
3. Restart Visual Studio

```bash
nvcc --version   # Should print CUDA version
```

### "Hybridizer satellite DLL not found"

**Cause**: The build step that generates the CUDA DLL didn't run.

**Fix**:
1. Ensure the Hybridizer NuGet packages are installed
2. Do a full Rebuild (not just Build)
3. Check bin/Debug/ or bin/Release/ for `*_CUDA.dll`

### "Unsupported IL pattern"

**Cause**: Your kernel code uses a C# feature not supported on GPU.

**Fix**: Check the [Known Limitations](../overview/what-is-hybridizer#known-limitations). Common culprits:
- `foreach` → use `for` loop
- `new MyClass()` → use structs or pass from host
- `string` operations → remove from kernel
- Exception handling → remove try/catch

---

## Runtime Errors

### "No CUDA-capable device detected"

```bash
nvidia-smi    # Check if GPU is detected
```

**If GPU appears**: Update drivers from [nvidia.com/drivers](https://nvidia.com/drivers)  
**If GPU doesn't appear**: Check hardware (PCIe seating, power connector)

### "Out of memory" on GPU

**Cause**: Arrays too large for GPU memory.

**Fix**:
- Check GPU memory with `nvidia-smi`
- Reduce array size or process in chunks
- Use `[In]`/`[Out]` to reduce concurrent allocations

### "Invalid device function" or "Launch failed"

**Cause**: Compiled for wrong GPU architecture.

**Fix**: Ensure your CUDA toolkit version matches your GPU:
- RTX 30xx → Compute 8.6+ → CUDA 11+
- RTX 40xx → Compute 8.9+ → CUDA 12+

---

## Wrong Results

### GPU result differs from CPU

**Checklist**:

1. **Did you call `cuda.DeviceSynchronize()`?** Without it, results may not be copied back yet
2. **Check `[In]`/`[Out]`**: Wrong direction = wrong data
3. **Floating-point precision**: GPU may execute operations in different order. Use tolerance-based comparison:
   ```csharp
   if (Math.Abs(gpu - cpu) > 1e-5f) // Not: if (gpu != cpu)
   ```
4. **Race condition?** Test with OMP backend (`HybRunner.OMP()`) — if OMP works but CUDA doesn't, you have a parallelization bug

### First kernel call returns zeros

The first call may include CUDA context initialization. Try:
```csharp
wrapper.MyKernel(args);           // Warmup
cuda.DeviceSynchronize();
// Now the real call
wrapper.MyKernel(args);
cuda.DeviceSynchronize();
```

---

## Performance

### GPU is slower than CPU

This is normal for:
- **Small arrays** (< 10K elements): Transfer overhead dominates
- **First call**: CUDA context initialization takes ~200ms
- **Already memory-bound**: If CPU is already at peak bandwidth

**Optimization checklist**:
1. Use `[In]`/`[Out]` attributes
2. Increase problem size (>100K elements)
3. Use `[IntrinsicFunction]` for math operations
4. Profile with `nsys profile` or `ncu`

### How to measure kernel time accurately

```csharp
// Warmup
wrapper.MyKernel(args);
cuda.DeviceSynchronize();

// Measure
var sw = Stopwatch.StartNew();
wrapper.MyKernel(args);
cuda.DeviceSynchronize();    // Include sync in timing!
sw.Stop();

Console.WriteLine($"Kernel: {sw.ElapsedMilliseconds} ms");
```

---

## Frequently Asked Questions

### Can I use Hybridizer without a GPU?

Yes! Use the OMP backend: `HybRunner.OMP()`. It runs generated code on CPU with OpenMP threads.

### What C# features are supported?

Most of C# works: classes (as host types), structs, arrays, generics, interfaces, static methods. See [Known Limitations](../overview/what-is-hybridizer#known-limitations) for what's not supported in kernel code.

### How does Hybridizer compare to writing CUDA C++ directly?

Hybridizer typically achieves 90-98% of hand-written CUDA performance. The main advantages are:
- Single-source C# (no separate .cu files)
- Automatic data marshalling
- Same code runs on CPU and GPU
- Debug with standard .NET tools

### Can I mix Hybridizer with existing CUDA code?

Yes. Use `[IntrinsicFunction]` and `[IntrinsicInclude]` to call existing CUDA functions from Hybridizer kernels.

### Does Hybridizer support multi-GPU?

Use `cuda.SetDevice(deviceId)` before creating `HybRunner`. Each HybRunner instance targets one GPU.
