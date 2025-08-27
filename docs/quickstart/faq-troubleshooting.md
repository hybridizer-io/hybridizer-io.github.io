---
id: faq-troubleshooting
title: FAQ & Troubleshooting
---

- Build fails finding CUDA toolkit → ensure correct version and PATH.
- Kernel launches but returns wrong results → check memory transfers and thread/block sizes.
- Performance lower than expected → profile transfers, coalescing, divergence; see CUDA Basics.
- Unsupported IL pattern → refactor and see Reference → Attributes & Limitations.
