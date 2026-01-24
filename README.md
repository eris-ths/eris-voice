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
| **Quality Modes** | Speed/quality tradeoff: high/balanced/fast/ultra_fast |
| **MCP Server** | Claude Code integration ready |

---

## Benchmark

### Real-Time Factor (RTF)

| Quality Mode | Codebooks | ms/step | RTF | Quality |
|-------------|-----------|---------|-----|---------|
| high | 15 | ~100ms | 0.83x | ★★★★★ |
| **balanced** | **11** | **~80ms** | **0.93-0.97x** | ★★★★☆ |
| fast | 7 | ~60ms | 1.4x | ★★★☆☆ |
| ultra_fast | 3 | ~40ms | 2.0x | ★★☆☆☆ |

> Environment: M3 MacBook Air 8GB Unified Memory

### Test Results (balanced mode)

| Text | Audio Duration | Generation Time | RTF |
|------|---------------|-----------------|-----|
| こんにちは。 | 2.48s | 2.56s | **0.97x** |
| 私はエリス... | 5.68s | 5.84s | **0.97x** |
| 今日はとても... | 3.84s | 4.15s | **0.93x** |

### Speedup vs PyTorch CPU

| Metric | PyTorch (CPU) | MLX | Speedup |
|--------|---------------|-----|---------|
| RTF | 0.08x | 0.97x | **12x** |
| Audio Decoder | baseline | 45x | **45x** |
| Quantizer | baseline | 3.5x | **3.5x** |
| Talker Forward | baseline | 10.8x | **10.8x** |

---

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| **macOS** | 13.0+ | Apple Silicon (M1/M2/M3/M4) |
| **Python** | 3.10+ | 3.11 recommended |
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

### 3. Start Server

```bash
python src/eris_voice_server.py
```

### 4. Use via MCP

Configure in `~/.claude.json`:
```json
{
  "mcpServers": {
    "eris-voice": {
      "command": "python",
      "args": ["/path/to/eris-voice/src/eris_voice_mcp.py"]
    }
  }
}
```

---

## API

### MCP Tools

```python
# Generate and play speech
eris_speak(text="こんにちは", speaker="ono_anna")

# With quality mode (balanced = realtime)
eris_speak(text="速く話して", quality_mode="fast")

# Streaming (sentence-by-sentence)
eris_speak_streaming(text="こんにちは。私はエリスよ。よろしくね。")

# Check status
eris_status()
```

### HTTP API

```bash
# Generate speech
curl -X POST http://localhost:8765/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは", "speaker": "ono_anna", "quality_mode": "balanced"}'

# Status
curl http://localhost:8765/status
```

---

## Quality Modes

| Mode | Use Case | RTF |
|------|----------|-----|
| `high` | Best quality, offline generation | 0.82x |
| `balanced` | **Default**, realtime playback | 1.06x |
| `fast` | Quick previews, acceptable quality | 1.40x |
| `ultra_fast` | Speed priority, reduced quality | 2.08x |

---

## Architecture

```
Text Input (PyTorch tokenization)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ MLX Generate Loop (12x speedup vs PyTorch CPU)      │
│                                                     │
│  Prefill:                                           │
│    Talker(initial_embeds) → past_hidden, code_0    │
│                                                     │
│  Generate Loop:                                     │
│    CodePredictor([past_hidden, embed(code_0)])     │
│      → codebook[1-15]                               │
│    Talker(sum_embeds + trailing)                    │
│      → new_past_hidden, new_code_0                  │
│                                                     │
│  Features:                                          │
│    ├── Talker: 28-layer Transformer + KVCache      │
│    ├── CodePredictor: 5-layer Transformer + KVCache│
│    ├── suppress_tokens: 2048-3071 (EOS=2150 OK)    │
│    └── Quality modes: high/balanced/fast/ultra_fast │
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

## Project Structure

```
.
├── src/
│   ├── eris_voice_mcp.py      # MCP server (Claude Code)
│   ├── eris_voice_server.py   # HTTP server (FastAPI)
│   ├── mlx_generate.py        # Generate loop (166x)
│   ├── mlx_talker.py          # MLX Talker model
│   ├── mlx_code_predictor.py  # MLX CodePredictor
│   ├── mlx_decoder_v2.py      # MLX Audio Decoder (45x)
│   ├── mlx_quantizer.py       # MLX Quantizer (3.5x)
│   ├── mlx_kv_cache.py        # KV Cache for inference
│   ├── mlx_sampling.py        # Sampling utilities
│   └── streaming_prototype.py # Sentence-level streaming
├── docs/
│   ├── generate_loop_mlx_tasks.md  # Progress report
│   └── BENCHMARKS.md               # Detailed benchmarks
├── tests/
├── archive/                   # Experimental code
└── *.npz                      # MLX weights (generated)
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
| Quality Modes | ✅ Complete |
| MCP Server | ✅ Complete |
| HTTP Server | ✅ Complete |
| Full MLX Pipeline | 🚧 In Progress |
| mx.compile() | 📋 Planned |

---

## Limitations

- **Apple Silicon Only**: MLX requires M1/M2/M3/M4
- **Hybrid Pipeline**: Still uses PyTorch for tokenization and some pre-processing
- **Memory**: Model weights require ~4GB RAM

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

## Authors

- **[@eris-ths](https://github.com/eris-ths)** - Development & MLX optimization
- **[@nao-amj](https://github.com/nao-amj)** - Project direction

## License

MIT License - see [LICENSE](LICENSE) for details.
