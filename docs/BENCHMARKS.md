# Benchmark Results

> **Environment**: M3 MacBook Air 8GB Unified Memory
> **Last Updated**: 2026-03-28

---

## Current Performance (Full MLX Pipeline)

### Real-Time Factor by Quality Mode

| Quality Mode | Codebooks | RTF (Direct MCP) | Audio Quality |
|-------------|-----------|-------------------|---------------|
| **high** | **15** | **0.4-0.7x** | Required for 0.6B Japanese |
| balanced | 11 | 0.8-1.0x | Japanese degrades |
| fast | 7 | 1.0-1.4x | Not recommended |
| ultra_fast | 3 | 1.4-2.0x | Unintelligible |

### E2E Generation (high mode, Direct MCP)

| Text | Audio Duration | Generation Time | RTF |
|------|---------------|-----------------|-----|
| Short (5 chars) | 4-5s | 10-13s | 0.4-0.5x |
| Medium (15 chars) | 6-8s | 12-15s | 0.5-0.7x |
| Long (30+ chars) | 8-10s | 15-20s | 0.5-0.6x |

### Streaming TTFA (Time To First Audio)

| Metric | Value |
|--------|-------|
| TTFA (10-step first chunk) | 1.2-1.7s |
| First chunk audio | 0.8s |
| Max per generation | ~12.5s (Metal 4GB limit) |

---

## Optimization History

### Phase 1: Quick Wins (Jan 2026)

| Change | Before | After | Improvement |
|--------|--------|-------|-------------|
| 1.7B -> 0.6B | 367s/5chars | 16s | 23x |
| fp32 -> bf16 | 16s | ~10s | 1.6x |
| OMP threads=8 | ~10s | ~8s | 1.25x |

### Phase 2: MLX Decoder/Quantizer (Jan 2026)

| Component | PyTorch CPU | MLX | Speedup |
|-----------|-------------|-----|---------|
| Audio Decoder | 93.85s | 2.07s | **45x** |
| Quantizer | 47.56s | 13.55s | **3.5x** |

### Phase 3: MLX Generate Loop (Jan 2026)

| Component | PyTorch CPU | MLX | Speedup |
|-----------|-------------|-----|---------|
| Talker Forward | 469ms | 43ms | **10.8x** |
| Generate Loop | 17,000ms | 102ms | **166x** |

### Phase 4: KV Cache + Attention (Mar 2026)

| Change | Before | After |
|--------|--------|-------|
| KVCache concat | O(n) per step | O(1) per step |
| Custom SDPA | Python matmul | mx.fast (Metal) |
| GQA expansion | mx.repeat 8->16 | Native GQA |

### Phase 5: TextEncoder MLX (Mar 2026)

| Metric | Before (PyTorch) | After (MLX) |
|--------|-------------------|-------------|
| Text processing | ~2-3s | ~0.1s |
| Numerical diff | - | max 0.007 |

### Phase 6: PreDecoder MLX (Mar 2026)

| Metric | Before | After |
|--------|--------|-------|
| Pipeline load time | ~25s | ~16.5s |
| PyTorch dependency | Required | **Eliminated** |
| Memory (PyTorch model) | ~1GB | 0 |

### Phase 7: PolarQuant KV Cache (Mar 2026)

| Metric | Standard | PolarQuant 4-bit |
|--------|----------|-----------------|
| KV memory (28L, 100 steps) | 21.9 MB | 4.2 MB |
| Compression | 1x | **5.2x** |
| Cosine similarity | 1.0 | 0.997 |
| Status | Production | MVP (not yet integrated) |

---

## Failed Approaches

| Approach | Result | Notes |
|----------|--------|-------|
| mlx-community 4-bit | Incompatible | MLX quantized format |
| PyTorch dynamic quantization | Error | Pickle error |
| torch.compile() | 2.37x slower | Overhead > benefit |
| MPS (Apple GPU) | Failed | conv1d channel limit |
| mx.compile() | No benefit | mx.fast SDPA already Metal-optimized |

---

## Memory Profile

```yaml
Model Weights (MLX):
  Talker (28L):         ~3.0 GB
  CodePredictor (5L):   ~0.5 GB
  PreDecoder (8L):      ~0.2 GB
  Decoder:              ~0.3 GB
  Quantizer:            ~0.03 GB
  TextEncoder bias:     ~0.01 GB
  Total:                ~4.0 GB

Runtime:
  KV Cache (150 steps): ~3 MB (or ~0.6 MB with PolarQuant)
  Audio Buffer:         ~0.3 GB
  Peak:                 ~4.3 GB (safe for 8GB, Metal 4GB limit)

Constraint:
  Metal buffer limit:   4 GB
  Max audio/generation: ~12.5s (150 steps at 12Hz)
  Long text strategy:   Streaming (sentence-level split)
```

---

*Benchmarked by Eris — 2026-03-28*
