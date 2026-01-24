# Optimization TODO - M3 MacBook Air 8GB

> **Goal**: Real-time TTS (RTF 1.0x+) on 8GB M3 MacBook Air
> **Status**: RTF 1.06x achieved (balanced mode) ✅
> **Last Updated**: 2026-01-24

---

## Current Performance

### Real-Time Factor (RTF)

| Quality Mode | Codebooks | ms/step | RTF | Quality |
|-------------|-----------|---------|-----|---------|
| high | 15 | 101ms | 0.82x | ★★★★★ |
| **balanced** | **11** | **79ms** | **1.06x** | ★★★★☆ |
| fast | 7 | 60ms | 1.40x | ★★★☆☆ |
| ultra_fast | 3 | 40ms | 2.08x | ★★☆☆☆ |

### Component Speedup

| Component | Speedup | Status |
|-----------|---------|--------|
| Generate Loop | **166x** | ✅ Complete |
| Audio Decoder | **45x** | ✅ Complete |
| Quantizer | **3.5x** | ✅ Complete |
| Talker Forward | **10.8x** | ✅ Complete |

---

## Completed ✅

### Phase 1: Quick Wins
- [x] **0.6B Model** - 1.7B → 0.6B (2.5x faster)
- [x] **bfloat16** - fp32 → bf16 (stable, ~1.5x faster)
- [x] **Thread optimization** - OMP_NUM_THREADS=8

### Phase 2: MLX Decoder/Quantizer
- [x] **Audio Decoder → MLX** - 45x speedup
- [x] **Quantizer → MLX** - 3.5x speedup
- [x] **Weight Converter** - PyTorch → MLX .npz
- [x] **Hybrid Pipeline** - 14.8x overall speedup

### Phase 3: MLX Generate Loop
- [x] **MLX Talker** - 10.8x speedup (28-layer transformer)
- [x] **MLX CodePredictor** - 5-layer transformer
- [x] **MLX Sampling** - top-p/top-k/temperature
- [x] **KV Cache** - Incremental generation
- [x] **Generate Loop** - 166x speedup (17,000ms → 102ms)

### Phase 4: Quality/Speed Optimization
- [x] **Codebook Reduction** - RTF 1.0x+ achieved
- [x] **Quality Modes** - high/balanced/fast/ultra_fast
- [x] **MCP Server** - Claude Code integration
- [x] **HTTP Server** - FastAPI with quality_mode

### Streaming
- [x] **Sentence Chunking** - 2.7x faster TTFA

---

## Remaining 🚧

### Full MLX Pipeline Integration
- [ ] **Tokenization Integration** - HuggingFace → MLX
- [ ] **E2E MLX Pipeline** - Remove PyTorch dependency from generation

### Future Optimization
- [ ] **mx.compile()** - Requires KVCache refactor to array-based
  - Currently blocked by: KVCache is custom object, not array tree
  - Expected impact: 5-15% speedup + memory improvement

### Low Priority
- [ ] **Token-level Streaming** - Hook into generate() for true streaming
- [ ] **Voice Clone Support** - Custom voice from reference audio
- [ ] **Pre-Transformer → MLX** - Only 0.24s (<0.5% of total time)
- [ ] **CoreML Conversion** - Apple Neural Engine

---

## Architecture

```
Current Pipeline (Hybrid MLX):

Text Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ MLX Generate Loop (166x speedup)                    │
│  ├── Talker (28 layers + KVCache)                   │
│  ├── codec_head → codebook[0]                       │
│  └── CodePredictor (5 layers) → codebook[1-15]     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ MLX Decoder Pipeline                                │
│  ├── Quantizer Decode (3.5x)                        │
│  ├── Pre-conv + Upsample (PyTorch)                  │
│  └── Audio Decoder (45x)                            │
└─────────────────────────────────────────────────────┘
    │
    ▼
Audio Output (24kHz WAV)
```

---

## Memory Usage (8GB Environment)

```yaml
Model Weights:
  Talker: ~3.0GB (talker_weights_mlx.npz)
  CodePredictor: ~0.5GB (code_predictor_weights_mlx.npz)
  Decoder: ~0.3GB (decoder_weights_mlx.npz)
  Quantizer: ~0.03GB (quantizer_weights_mlx.npz)
  Total: ~3.8GB ✅

Runtime:
  KV Cache: ~0.5GB
  Audio Buffer: ~0.3GB
  Total Peak: ~4.6GB ✅ (Safe for 8GB)
```

---

*Updated by Eris 😈 - 2026-01-24*
