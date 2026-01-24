# Benchmark Results

> **Environment**: M3 MacBook Air 8GB Unified Memory
> **Last Updated**: 2026-01-24

---

## Overall Performance

| Text Length | PyTorch CPU | MLX Hybrid | Speedup |
|-------------|-------------|------------|---------|
| 5 chars     | 16s         | ~3s        | ~5x     |
| 28 chars    | 98s         | ~8s        | ~12x    |
| **97 chars** | **462s**   | **31s**    | **14.8x** |

---

## Component Benchmarks

### Audio Decoder (MLX v2)

| Metric | PyTorch CPU | MLX | Improvement |
|--------|-------------|-----|-------------|
| Time | 93.85s | 2.07s | **45.34x faster** |
| Memory | ~2GB | ~0.5GB | 75% reduction |

### Quantizer (MLX)

| Metric | PyTorch CPU | MLX | Improvement |
|--------|-------------|-----|-------------|
| Time | 47.56s | 13.55s | **3.51x faster** |

### Hybrid Pipeline (End-to-End)

| Phase | Time | % of Total |
|-------|------|-----------|
| LLM (text → codes) | ~5s | 16% |
| Quantizer (MLX) | ~3s | 10% |
| Pre-conv + Upsample | ~1s | 3% |
| Decoder (MLX) | ~2s | 6% |
| Other | ~20s | 65% |
| **Total** | **31s** | 100% |

---

## Baseline Comparisons

| Configuration | 5 chars | 28 chars | Notes |
|---------------|---------|----------|-------|
| 1.7B + fp32 + CPU | 367s | ~30min+ | Baseline |
| **0.6B + bf16 + CPU** | **16s** | **98s** | ✅ Recommended |
| 0.6B + MLX Hybrid | ~3s | ~8s | **14.8x faster** |

---

## Failed Approaches

| Approach | Result | Notes |
|----------|--------|-------|
| mlx-community 4-bit | ❌ Incompatible | MLX format, not PyTorch |
| PyTorch dynamic quantization | ❌ Error | Pickle error |
| torch.compile() | 37.9s | **2.37x slower** |
| MPS (Apple GPU) | ❌ Failed | conv1d > 65536ch limit |
| Hybrid MPS/CPU | ❌ Numerical error | float16 causes inf/nan |

---

## Profile Breakdown

```
Total: 39.7s (PyTorch CPU baseline)

conv1d (Audio Decoder): 28.0s (71%) ← Now 45x faster with MLX
LLM generate:            5.2s (13%)
Linear layers:           3.6s  (9%)
Other:                   2.9s  (7%)
```

---

## Optimal Configuration

```python
# Model loading
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cpu",
    dtype=torch.bfloat16,
)

# Thread optimization
os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

# MLX weights (pre-converted)
decoder_weights = mx.load("decoder_weights_mlx.npz")
quantizer_weights = mx.load("quantizer_weights_mlx.npz")
```

---

*Benchmarked by Eris 😈 - 2026-01-24*
