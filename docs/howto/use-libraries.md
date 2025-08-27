---
id: use-libraries
title: Use Existing Libraries
---

- Call into cuBLAS/cuDNN or CPU libraries from generated code via interop
- Strategy: host orchestrates library calls + Hybridizer kernels
- Data layout agreements and zero-copy possibilities
- Versioning and ABI considerations
