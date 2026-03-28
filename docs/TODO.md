# Optimization TODO - M3 MacBook Air 8GB

> **Goal**: Real-time TTS (RTF 1.0x+) on 8GB M3 MacBook Air
> **Status**: RTF 1.0x achieved (high mode, Direct MCP) ✅
> **Last Updated**: 2026-03-28

---

## Current Performance

### Real-Time Factor (RTF)

| Quality Mode | Codebooks | RTF (Direct) | Quality | Notes |
|-------------|-----------|--------------|---------|-------|
| **high** | **15** | **0.4-0.7x** | ★★★★★ | **Required for 0.6B Japanese** |
| balanced | 11 | 0.8-1.0x | ★★★☆☆ | Japanese quality degrades |
| fast | 7 | 1.0-1.4x | ★★☆☆☆ | Not recommended |
| ultra_fast | 3 | 1.4-2.0x | ★☆☆☆☆ | Unintelligible |

### Component Speedup

| Component | Speedup | Status |
|-----------|---------|--------|
| Generate Loop | **166x** | ✅ Complete |
| Audio Decoder | **45x** | ✅ Complete |
| Quantizer | **3.5x** | ✅ Complete |
| Talker Forward | **10.8x** | ✅ Complete |
| KV Cache | **O(1)/step** | ✅ Complete (03/28) |
| Attention | **mx.fast SDPA** | ✅ Complete (03/28) |
| TextEncoder | **MLX native** | ✅ Complete (03/28) |

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

### Phase 5: KV Cache + Attention Optimization (03/28)
- [x] **Pre-allocated KV Cache** - concat O(n²) → slice O(1) (MLX-LM based)
- [x] **mx.fast SDPA** - Metal-accelerated attention, native GQA
- [x] **Dead code cleanup** - Removed duplicate KVCache, custom SDPA
- [x] **mx.compile evaluation** - No benefit (mx.fast already Metal-optimized)

### Phase 6: TextEncoder MLX Migration (03/28)
- [x] **MLX TextEncoder** - Tokenizer + embedding + projection + token assembly
- [x] **text_projection bias** - Extracted and applied (fc1_bias, fc2_bias)
- [x] **Numerical validation** - max diff 0.007 vs PyTorch
- [x] **Codebook embedding pre-computation** - @property → load_weights()

### Phase 7: MCP + Voice Customization (03/28)
- [x] **Direct MCP** - In-process pipeline (no HTTP overhead)
- [x] **voice_presets.yaml** - 6 mood presets, hot-reloadable
- [x] **instruct parameter** - Direct voice style control
- [x] **max_steps 150** - Safe for M3 8GB Metal limit

---

### Phase 8: Audio Decoder Full MLX (03/28)
- [x] **Pre-conv/Upsample → MLX** - PyTorch completely eliminated
  - `mlx_pre_decoder.py`: CausalConv + 8-layer Transformer + 2-stage ConvTranspose/ConvNeXt
  - Load time: 25s → 16.5s
  - 117 weight tensors extracted (202MB pre_decoder_weights_mlx.npz)

---

## Remaining 🚧

### Future Research
- [ ] **TurboQuant/PolarQuant** - KV cache quantization (3.5-bit, 9x compression)
  - Papers: arXiv:2504.19874, 2502.02617
  - Impact: Enable 1.7B on 8GB or longer context on 0.6B
- [ ] **1.7B Model** - Better quality at balanced mode (requires 16GB+ RAM)

### Low Priority
- [ ] **Token-level Streaming** - Hook into generate() for true streaming
- [ ] **Voice Clone Support** - Custom voice from reference audio
- [ ] **CoreML Conversion** - Apple Neural Engine

---

## Constraints (M3 8GB)

```yaml
Metal buffer limit: 4GB
Max audio per generation: ~12.5s (150 steps at 12Hz)
Long text strategy: streaming (sentence-level split)
Quality requirement: high (15 codebooks) for Japanese
temperature: 0.5 (reduces long vowel stretching)
```

---

## Architecture

```
Text Input
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ MLX TextEncoder (PyTorch-free)                               │
│  ├── Tokenizer (transformers)                                │
│  ├── text_embedding → text_projection (with bias)            │
│  └── Token assembly (instruct + role + codec + text)         │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ MLX Generate Loop (166x speedup)                             │
│  ├── Talker (28 layers + Pre-allocated KVCache)              │
│  ├── mx.fast SDPA (Metal + native GQA)                       │
│  ├── codec_head → codebook[0]                                │
│  └── CodePredictor (5 layers) → codebook[1-15]               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Audio Decoder Pipeline                                       │
│  ├── Quantizer Decode (3.5x, pre-computed codebook)          │
│  ├── PreDecoder (pre-conv + 8L transformer + 2x upsample)   │
│  └── Audio Decoder (45x, SnakeBeta + conv1d)                 │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Audio Output (24kHz WAV)
```

---

*Updated by Eris 😈 - 2026-03-28*
