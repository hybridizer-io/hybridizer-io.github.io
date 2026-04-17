---
id: black-scholes
title: "Black-Scholes Option Pricing"
description: "A real-world finance application — pricing millions of European options on GPU."
keywords: [Hybridizer, Black-Scholes, finance, option pricing, GPU]
---

# Black-Scholes Option Pricing

> **Sample source**: [`4.Finance/BlackScholes`](https://github.com/hybridizer-io/hybridizer-basic-samples/tree/master/src/4.Finance/BlackScholes)

This example implements the Black-Scholes formula for pricing European call and put options. It demonstrates:
- Intrinsic function mapping (`[IntrinsicFunction]`)
- `[Kernel]` helper functions
- `[In]` / `[Out]` marshalling optimization
- Numerical validation between CPU and GPU

## Intrinsic Function Mapping

C# Math functions run on CPU. For GPU performance, we map them to CUDA fast-math intrinsics:

```csharp
[IntrinsicFunction("expf")]
public static float Expf(float f)
{
    return (float)Math.Exp((double)f);
}

[IntrinsicFunction("sqrtf")]
public static float Sqrtf(float f)
{
    return (float)Math.Sqrt((double)f);
}

[IntrinsicFunction("logf")]
public static float Logf(float f)
{
    return (float)Math.Log((double)f);
}

[IntrinsicFunction("fabsf")]
public static float fabsf(float f)
{
    return Math.Abs(f);
}
```

| C# Code | CPU | GPU (CUDA) |
|----------|-----|------------|
| `Expf(x)` | `Math.Exp` | `expf` (hardware unit) |
| `Sqrtf(x)` | `Math.Sqrt` | `sqrtf` (hardware unit) |
| `Logf(x)` | `Math.Log` | `logf` (SFU) |

:::tip
The body of each intrinsic function serves as the **CPU fallback**. On GPU, it is replaced by the native CUDA instruction. This keeps code portable.
:::

## The Kernel

### Cumulative Normal Distribution

```csharp
[Kernel]
static float CND(float f)
{
    const float A1 = 0.31938153f;
    const float A2 = -0.356563782f;
    const float A3 = 1.781477937f;
    const float A4 = -1.821255978f;
    const float A5 = 1.330274429f;
    const float RSQRT2PI = 0.3989422804f;

    float K = 1.0f / (1.0f + 0.2316419f * fabsf(f));
    float cnd = RSQRT2PI * Expf(-0.5f * f * f) *
                (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (f > 0) cnd = 1.0f - cnd;
    return cnd;
}
```

### Black-Scholes Entry Point

```csharp
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

[EntryPoint]
public static void BlackScholes(
    [Out] float[] callResult,
    [Out] float[] putResult,
    [In]  float[] stockPrice,
    [In]  float[] optionStrike,
    [In]  float[] optionYears,
    int lineFrom, int lineTo)
{
    for (int i = lineFrom + blockDim.x * blockIdx.x + threadIdx.x;
         i < lineTo;
         i += blockDim.x * gridDim.x)
    {
        float sqrtT = Sqrtf(optionYears[i]);
        float f1 = (Logf(stockPrice[i] / optionStrike[i])
                    + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * optionYears[i])
                   / (VOLATILITY * sqrtT);
        float f2 = f1 - VOLATILITY * sqrtT;

        float CNDF1 = CND(f1);
        float CNDF2 = CND(f2);

        float expRT = Expf(-RISKFREE * optionYears[i]);
        callResult[i] = stockPrice[i] * CNDF1 - optionStrike[i] * expRT * CNDF2;
        putResult[i]  = optionStrike[i] * expRT * (1.0f - CNDF2)
                       - stockPrice[i] * (1.0f - CNDF1);
    }
}
```

### Marshalling Analysis

| Parameter | Attribute | Transfer | Why |
|-----------|-----------|----------|-----|
| `callResult` | `[Out]` | Device → Host | Written only |
| `putResult` | `[Out]` | Device → Host | Written only |
| `stockPrice` | `[In]` | Host → Device | Read only |
| `optionStrike` | `[In]` | Host → Device | Read only |
| `optionYears` | `[In]` | Host → Device | Read only |

With 1M options × 5 arrays × 4 bytes: proper `[In]`/`[Out]` usage **saves 10 MB of PCI-e transfer per call**.

## Launching and Validation

```csharp
int OPT_N = 1024 * 1024 * Environment.ProcessorCount;

cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);
HybRunner runner = SatelliteLoader.Load()
    .SetDistrib(8 * prop.multiProcessorCount, 256);
dynamic wrapper = runner.Wrap(new Program());

wrapper.BlackScholes(callResult_cuda, putResult_cuda,
                     stockPrice, optionStrike, optionYears,
                     0, OPT_N);
```

### Numerical Parity Check

GPU floating-point may differ slightly from CPU:

```csharp
float maxCallError = 0.0f;
for (int i = 0; i < OPT_N; ++i)
{
    float error = Math.Abs(callResult_net[i] - callResult_cuda[i]);
    maxCallError = Math.Max(maxCallError, error);
}
// Typically: L∞ < 1e-6 for single precision
```

:::warning
Floating-point arithmetic is **not associative**. GPU may execute operations in different order than CPU, leading to small differences. Always validate with tolerances, not exact equality.
:::

## Next Steps

- [Hello World](./hello-world) — Simpler starting point
- [Intrinsics & Builtins](../guide/intrinsics-builtins) — Full intrinsic reference
- [Data Marshalling](../guide/data-marshalling) — Transfer optimization
