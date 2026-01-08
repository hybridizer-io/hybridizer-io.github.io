---
id: use-libraries
title: Use Existing Libraries
description: How to call GPU libraries like cuBLAS from Hybridizer code.
keywords: [Hybridizer, cuBLAS, cuDNN, library, interop, BLAS]
---

# Use Existing Libraries

Hybridizer-generated code can interoperate with existing GPU libraries like cuBLAS, cuDNN, and cuFFT for specialized operations.

## Integration Strategies

### Strategy 1: Host Orchestration

The host orchestrates library calls alongside Hybridizer kernels:

```csharp
// Initialize cuBLAS
cublasHandle_t handle;
cublasCreate(ref handle);

// Step 1: Hybridizer kernel for preprocessing
wrapper.Preprocess(d_input, d_temp, N);

// Step 2: cuBLAS for matrix multiplication
cublasDgemm(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    ref alpha, d_A, M, d_B, K,
    ref beta, d_C, M);

// Step 3: Hybridizer kernel for postprocessing
wrapper.Postprocess(d_temp, d_output, N);

// Cleanup
cublasDestroy(handle);
```

### Strategy 2: Direct Library Calls

Use P/Invoke to call CUDA libraries directly:

```csharp
public static class CuBLAS
{
    [DllImport("cublas64_12.dll")]
    public static extern int cublasDgemm(
        IntPtr handle,
        int transa, int transb,
        int m, int n, int k,
        ref double alpha,
        IntPtr A, int lda,
        IntPtr B, int ldb,
        ref double beta,
        IntPtr C, int ldc);
}
```

## Common Libraries

### cuBLAS (Linear Algebra)

```csharp
// Matrix multiplication
cublasSgemm(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    ref alpha, d_A, M, d_B, K,
    ref beta, d_C, M);

// Vector operations
cublasSaxpy(handle, N, ref alpha, d_x, 1, d_y, 1);
```

### cuFFT (FFT)

```csharp
// Create FFT plan
cufftHandle plan;
cufftPlan1d(ref plan, N, CUFFT_C2C, 1);

// Execute FFT
cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

// Cleanup
cufftDestroy(plan);
```

### cuDNN (Deep Learning)

```csharp
// Convolution forward
cudnnConvolutionForward(
    handle,
    ref alpha, xDesc, x,
    wDesc, w,
    convDesc, algo, workspace, workspaceSize,
    ref beta, yDesc, y);
```

## Data Layout Considerations

Libraries often expect specific data layouts:

| Library | Expected Layout |
|---------|----------------|
| cuBLAS | Column-major (Fortran) |
| cuDNN | NCHW or NHWC |
| cuFFT | Contiguous complex |

### Converting Layouts

```csharp
[EntryPoint]
public static void TransposeForCuBLAS(
    float[] input, float[] output, 
    int rows, int cols)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < rows && j < cols)
    {
        // Row-major to column-major
        output[j * rows + i] = input[i * cols + j];
    }
}
```

## Zero-Copy Integration

Share memory between Hybridizer and libraries:

```csharp
// Allocate once
IntPtr d_data;
cuda.Malloc(out d_data, size);

// Use with Hybridizer
wrapper.PreProcess(d_data, N);

// Use same memory with cuBLAS
cublasDgemv(handle, ..., d_data, ...);

// Use result with Hybridizer
wrapper.PostProcess(d_data, N);
```

## Stream Synchronization

Ensure proper ordering when mixing libraries:

```csharp
cudaStream_t stream;
cuda.StreamCreate(out stream);

// Set stream for cuBLAS
cublasSetStream(handle, stream);

// Set stream for Hybridizer
wrapper.SetStream(stream);

// Operations execute in order on same stream
wrapper.Preprocess(d_data, N);
cublasSgemm(...);  // waits for Preprocess
wrapper.Postprocess(d_result, N);  // waits for SGEMM

cuda.StreamSynchronize(stream);
```

## Versioning Considerations

| Consideration | Recommendation |
|---------------|----------------|
| CUDA version | Match library and toolkit |
| Library version | Pin in build config |
| ABI compatibility | Test with upgrades |

## Next Steps

- [CI/CD Integration](./ci-cd-integration) — Reproducible builds
- [Invoke Generated Code](../guide/invoke-generated-code) — Kernel invocation
- [Data Marshalling](../guide/data-marshalling) — Memory management
