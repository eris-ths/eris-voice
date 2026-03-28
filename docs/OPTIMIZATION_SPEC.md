# Qwen3-TTS MLX Optimization Spec

> **Goal**: Real-time TTS on Apple Silicon (M3 8GB)
> **Status**: Full MLX pipeline achieved. PyTorch eliminated.
> **Created**: 2026-01-23
> **Last Updated**: 2026-03-28

---

## Result Summary

| Metric | Start (Jan 23) | End (Mar 28) | Improvement |
|--------|----------------|-------------|-------------|
| RTF | 0.003x (367s/1.18s) | 0.4-0.7x (high mode) | **130-230x** |
| PyTorch dependency | 100% | **0%** | Eliminated |
| Pipeline load time | ~40s | ~16.5s | 2.4x |
| Model | 1.7B fp32 | 0.6B, MLX native | - |

---

## Optimization Layers

```
Layer 1: Model selection
  1.7B fp32 -> 0.6B bf16 weights -> MLX native
  Impact: 23x (367s -> 16s)

Layer 2: Component-level MLX migration
  Audio Decoder:    PyTorch -> MLX conv1d (45x)
  Quantizer:        PyTorch -> MLX (3.5x)
  Generate Loop:    PyTorch -> MLX autoregressive (166x)
  TextEncoder:      PyTorch -> MLX embedding + projection
  PreDecoder:       PyTorch -> MLX (8L transformer + ConvTranspose + ConvNeXt)
  Impact: 12x overall (cumulative)

Layer 3: Algorithmic improvements
  KV Cache:         O(n) concat -> O(1) pre-allocated (MLX-LM based)
  Attention:        Custom SDPA -> mx.fast.scaled_dot_product_attention (Metal)
  GQA:              mx.repeat -> native GQA in mx.fast
  Codebook:         @property -> pre-computed at load time
  Impact: ~20-40% at long sequences

Layer 4: Compression (research)
  PolarQuant:       KV cache 4-bit (5.2x compression, cosine sim 0.997)
  Status:           MVP complete, not yet integrated into pipeline
```

---

## Architectural Constraints

The Qwen3-TTS architecture imposes sequential codebook prediction:

```
1 step = Talker(1x) + CodePredictor(up to 15x) + Decoder(1x)
```

This cannot be parallelized without model retraining.
Optimization value comes from making each sequential call cheaper.

See README.md "Design Philosophy" for detailed analysis.

---

## Quality Constraints (0.6B model)

- `high` mode (15 codebooks) required for intelligible Japanese
- `balanced` and below: insufficient acoustic information for 0.6B
- temperature 0.5 reduces long vowel artifacts
- Max ~12.5s per generation (Metal 4GB buffer limit)

---

*Eris — 2026-03-28*
