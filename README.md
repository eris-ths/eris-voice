# Eris Voice - Qwen3-TTS for Apple Silicon

> MLX-accelerated Text-to-Speech for Apple Silicon 😈

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Qwen3-TTS optimized for Apple Silicon without CUDA.
Achieves **real-time speech synthesis (RTF 1.0x+)** through MLX-native pipeline.

> ⚠️ **Experimental Software**: This is a research project. See [Disclaimer](#disclaimer) before use.

---

## Highlights

| Achievement | Description |
|-------------|-------------|
| **RTF 1.0x+** | Real-time speech synthesis achieved |
| **166x Generate Loop** | MLX-native autoregressive generation |
| **45x Audio Decoder** | MLX conv1d acceleration |
| **Pre-allocated KV Cache** | O(1) per-step updates (MLX-LM based) |
| **mx.fast SDPA** | Metal-accelerated attention with native GQA |
| **MLX TextEncoder** | PyTorch-free text processing |
| **Voice Presets** | YAML-based voice mood customization |
| **Direct MCP** | In-process pipeline, no HTTP overhead |

---

## Benchmark

### Real-Time Factor (RTF)

| Quality Mode | Codebooks | RTF (Direct) | Quality | Notes |
|-------------|-----------|--------------|---------|-------|
| **high** | **15** | **0.4-0.7x** | ★★★★★ | **Recommended for 0.6B** |
| balanced | 11 | 0.8-1.0x | ★★★☆☆ | Japanese quality degrades |
| fast | 7 | 1.0-1.4x | ★★☆☆☆ | Not recommended for Japanese |
| ultra_fast | 3 | 1.4-2.0x | ★☆☆☆☆ | Unintelligible |

> Environment: M3 MacBook Air 8GB Unified Memory
> Note: 0.6B model requires `high` mode for intelligible Japanese speech.

### Speedup vs PyTorch CPU

| Component | PyTorch (CPU) | MLX | Speedup |
|-----------|---------------|-----|---------|
| RTF | 0.08x | 0.97x | **12x** |
| Audio Decoder | baseline | 45x | **45x** |
| Quantizer | baseline | 3.5x | **3.5x** |
| Talker Forward | baseline | 10.8x | **10.8x** |

---

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| **macOS** | 13.0+ | Apple Silicon (M1/M2/M3/M4) |
| **Python** | 3.10+ | 3.11+ recommended |
| **Memory** | 8GB+ | Model weights ~4GB |

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/eris-ths/eris-voice.git
cd eris-voice

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Convert Weights

```bash
python src/weight_converter.py
python src/convert_talker_weights.py
```

### 3. Use via MCP (Recommended)

Configure in `~/.claude.json`:
```json
{
  "mcpServers": {
    "eris-voice": {
      "command": "python3",
      "args": ["/path/to/eris-voice/src/eris_voice_mcp_direct.py"]
    }
  }
}
```

The Direct MCP server loads the pipeline in-process — no separate HTTP server needed.
First call triggers model loading (~40s), subsequent calls are fast.

### 4. Alternative: HTTP Server

```bash
python src/eris_voice_server.py
```

Then use `eris_voice_mcp.py` (HTTP client version) instead.

---

## Voice Customization

### Voice Mood Presets

Defined in `voice_presets.yaml` (hot-reloaded, no restart needed):

| Mood | Description | Age Feel |
|------|-------------|----------|
| `default` | Standard anime voice | 20 |
| `playful` | Bright, cheerful | 15 |
| `calm` | Composed, explanatory | 25 |
| `gentle` | Soft, warm | 20 |
| `serious` | Clear, focused | 20 |
| `whisper` | Soft whisper | 20 |

### Usage

```python
# Use preset
eris_speak(text="...", voice_mood="playful")

# Direct instruct (overrides voice_mood)
eris_speak(text="...", instruct="A cute anime girl speaking cheerfully")
```

---

## API

### MCP Tools

```python
# Generate and play speech (high quality, default)
eris_speak(text="こんにちは", speaker="ono_anna")

# With voice mood preset
eris_speak(text="今日は良い天気ね", voice_mood="gentle")

# Direct voice style instruction
eris_speak(text="確実に進化してるわ", instruct="A 20-year-old cute Japanese anime girl voice")

# Streaming (sentence-by-sentence)
eris_speak_streaming(text="こんにちは。私はエリスよ。よろしくね。")

# List voice moods
eris_list_voice_moods()

# Check status
eris_status()
```

### HTTP API

```bash
# Generate speech
curl -X POST http://localhost:8765/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは", "speaker": "ono_anna", "quality_mode": "high"}'

# Status
curl http://localhost:8765/status
```

---

## Architecture

```
Text Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ MLX TextEncoder (PyTorch-free)                      │
│  ├── Tokenizer (transformers AutoTokenizer)         │
│  ├── text_embedding → text_projection (with bias)   │
│  ├── codec_embedding (speaker/language tokens)      │
│  └── Token assembly (instruct + role + text)        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ MLX Generate Loop (12x speedup vs PyTorch CPU)      │
│                                                     │
│  Prefill:                                           │
│    Talker(initial_embeds) → past_hidden, code_0     │
│                                                     │
│  Generate Loop:                                     │
│    CodePredictor([past_hidden, embed(code_0)])       │
│      → codebook[1-15]                               │
│    Talker(sum_embeds + trailing)                     │
│      → new_past_hidden, new_code_0                  │
│                                                     │
│  Optimizations:                                     │
│    ├── Pre-allocated KV Cache (O(1) per step)       │
│    ├── mx.fast.scaled_dot_product_attention          │
│    ├── Native GQA (no mx.repeat overhead)           │
│    └── Quality modes: high/balanced/fast/ultra_fast  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Audio Decoder Pipeline                              │
│  ├── Quantizer Decode (3.5x, pre-computed codebook) │
│  ├── MLX PreDecoder (pre-conv + 8L transformer + upsample) │
│  └── Audio Decoder (45x MLX)                        │
└─────────────────────────────────────────────────────┘
    │
    ▼
Audio Output (24kHz WAV)
```

---

## Project Structure

```
.
├── src/
│   ├── eris_voice_mcp_direct.py  # MCP server (Direct, recommended)
│   ├── eris_voice_mcp.py         # MCP server (HTTP client)
│   ├── eris_voice_server.py      # HTTP server (FastAPI)
│   ├── mlx_text_encoder.py       # MLX TextEncoder (PyTorch-free)
│   ├── mlx_pipeline.py           # Full pipeline orchestration
│   ├── mlx_generate.py           # Generate loop (166x)
│   ├── mlx_talker.py             # MLX Talker (28 layers)
│   ├── mlx_code_predictor.py     # MLX CodePredictor (5 layers)
│   ├── mlx_pre_decoder.py        # MLX PreDecoder (pre-conv + 8L transformer + upsample)
│   ├── mlx_decoder_v2.py         # MLX Audio Decoder (45x)
│   ├── mlx_quantizer.py          # MLX Quantizer (3.5x)
│   ├── mlx_kv_cache.py           # Pre-allocated KV Cache
│   └── mlx_sampling.py           # Sampling utilities
├── voice_presets.yaml             # Voice mood presets (YAML)
├── text_projection_bias.npz       # TextEncoder bias weights
├── docs/
├── tests/
├── archive/
└── *.npz                          # MLX weights (generated)
```

---

## Available Speakers

| Speaker | Language | Description |
|---------|----------|-------------|
| ono_anna ⭐ | Japanese | Playful female |
| vivian | Chinese | Bright, edgy female |
| serena | Chinese | Warm, gentle female |
| ryan | English | Dynamic male |
| aiden | English | Sunny American male |
| sohee | Korean | Warm female |

---

## Roadmap

| Feature | Status |
|---------|--------|
| RTF 1.0x (Real-time) | ✅ Complete |
| MLX Generate Loop | ✅ Complete (166x) |
| Pre-allocated KV Cache | ✅ Complete (O(1)) |
| mx.fast SDPA | ✅ Complete (Metal) |
| MLX TextEncoder | ✅ Complete |
| Codebook Pre-computation | ✅ Complete |
| Voice Mood Presets | ✅ Complete |
| Direct MCP Server | ✅ Complete |
| Quality Modes | ✅ Complete |
| Audio Decoder full MLX | ✅ Complete (PyTorch eliminated) |
| TurboQuant KV Cache | 📋 Research (PolarQuant paper) |
| 1.7B Model Support | 📋 Requires 16GB+ RAM |

---

## Limitations

- **Apple Silicon Only**: MLX requires M1/M2/M3/M4
- **0.6B Model**: Requires `high` quality mode (15 codebooks) for Japanese
- **Max ~12.5s per generation**: M3 8GB Metal buffer limit (use streaming for longer text)
- **PyTorch-free**: Entire pipeline runs on MLX. `transformers` (tokenizer only) is the sole non-MLX dependency

---

## Disclaimer

### Experimental Software

This is **experimental research software** provided "as is" without warranty.

### Voice Synthesis Ethics

- Do not clone voices without consent
- Label AI-generated audio as synthetic
- Follow local laws regarding synthetic media
- Do not use for fraud or impersonation

---

## Acknowledgments

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by Alibaba (Apache 2.0)
- **[MLX](https://github.com/ml-explore/mlx)** by Apple
- **[MLX-LM](https://github.com/ml-explore/mlx-examples)** KV Cache reference implementation

## Authors

- **[@eris-ths](https://github.com/eris-ths)** - Development & MLX optimization
- **[@nao-amj](https://github.com/nao-amj)** - Project direction

## License

MIT License - see [LICENSE](LICENSE) for details.
