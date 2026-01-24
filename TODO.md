# Optimization TODO - M3 MacBook Air 8GB

> **Goal**: Practical TTS speed on 8GB M3 MacBook Air
> **Status**: 98s → 31s achieved (14.8x speedup) ✅
> **Last Updated**: 2026-01-24

---

## Current Performance

| Text | Before | After | Speedup |
|------|--------|-------|---------|
| 5 chars | 16s | ~3s | ~5x |
| 28 chars | 98s | ~8s | ~12x |
| 97 chars | 462s | 31s | **14.8x** ✅ |

---

## Completed

### Phase 1: Quick Wins

- [x] **0.6B Model** - 1.7B → 0.6B (2.5x faster)
- [x] **bfloat16** - fp32 → bf16 (stable, ~1.5x faster)
- [x] **Thread optimization** - OMP_NUM_THREADS=8

### Phase 2: MLX Migration ✅

- [x] **Audio Decoder → MLX** - 45x speedup
  - CausalConv1d, CausalTransConv1d
  - SnakeBeta activation (log-scale alpha/beta)
  - DecoderBlock with residual units

- [x] **Quantizer → MLX** - 3.5x speedup
  - SplitResidualVectorQuantizer
  - EuclideanCodebook with EMA embedding

- [x] **Weight Converter** - PyTorch → MLX .npz

- [x] **Hybrid Pipeline** - 14.8x overall speedup

### Code Quality

- [x] Type hints throughout
- [x] Proper error handling (WeightLoadError)
- [x] Unit tests (test_decoder.py, test_quantizer.py)
- [x] Clean project structure

---

## Remaining (Low Priority)

### Pre-Transformer

- [ ] Port to MLX
  - Only 0.24s (<0.5% of total time)
  - Diminishing returns

### Future Optimization

- [ ] **mlx-audio Integration**
  - Waiting for Qwen3-TTS support
  - Expected: additional 2-3x speedup

- [ ] **Streaming Output**
  - Generate audio in chunks
  - Lower latency for long text

- [ ] **CoreML Conversion**
  - Apple Neural Engine
  - Potential for further acceleration

---

## Architecture Notes

```
Current Pipeline (Hybrid):

PyTorch                          MLX
   │                              │
   ├── LLM (text → codes)         │
   ├── Quantizer decode ──────────┼── 3.5x faster
   ├── Pre-conv + upsample        │
   └── Decoder blocks ────────────┼── 45x faster
                                  │
                              Audio output
```

---

## Memory Usage (8GB Environment)

```yaml
0.6B bf16:
  Model: ~1.2GB
  KV Cache: ~0.5GB
  Audio Buffer: ~0.3GB
  MLX Weights: ~0.3GB
  Total: ~2.3GB ✅ (Safe margin)
```

---

*Updated by Eris 😈 - 2026-01-24*
