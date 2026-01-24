# Eris Voice - Qwen3-TTS for Apple Silicon

> MLX-accelerated Text-to-Speech for Apple Silicon 😈

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Qwen3-TTS optimized for Apple Silicon without CUDA.
Achieves **14.8x speedup** through MLX hybrid pipeline.

## Why This Exists

### The Problem

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) is an excellent open-source TTS model with voice cloning capabilities. However:

- **CUDA-centric**: Designed for NVIDIA GPUs
- **Unusable on Mac**: 98 seconds for 28 characters on CPU
- **No Apple Silicon optimization**: MPS fails due to conv1d channel limits

### The Solution

This project bridges the gap by porting the performance-critical audio decoder to [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework:

| Before | After | Improvement |
|--------|-------|-------------|
| 98s / 28 chars | 8s / 28 chars | **12x faster** |
| 462s / 97 chars | 31s / 97 chars | **14.8x faster** |

### Who This Helps

| User | Need |
|------|------|
| **Mac Developers** | Local TTS without cloud dependency |
| **Privacy-Conscious** | Keep voice data on-device |
| **Hobbyists** | Add voices to characters/projects |
| **Cost-Conscious** | Avoid API fees for TTS |

## Features

- **No CUDA Required**: Runs on Apple Silicon (M1/M2/M3/M4) CPU
- **MLX Acceleration**: Audio Decoder with **45x speedup**, Quantizer with **3.5x speedup**
- **Hybrid Pipeline**: PyTorch + MLX for optimal performance
- **Custom Voice**: Japanese female voice preset (ono_anna) with instruct support

## Benchmark

### Overall Performance

| Text Length | PyTorch CPU | MLX Hybrid | Speedup |
|-------------|-------------|------------|---------|
| 5 chars     | 16s         | ~3s        | ~5x     |
| 28 chars    | 98s         | ~8s        | ~12x    |
| 97 chars    | 462s        | 31s        | **14.8x** |

### Component Breakdown

| Component | PyTorch | MLX | Speedup |
|-----------|---------|-----|---------|
| Audio Decoder | 93.85s | 2.07s | **45.34x** |
| Quantizer | 47.56s | 13.55s | **3.51x** |
| Hybrid Pipeline | - | - | **14.8x** |

> Environment: M3 MacBook Air 8GB Unified Memory

## Installation

```bash
# Dependencies
pip install qwen-tts soundfile torch mlx

# SoX (optional, suppresses warnings)
brew install sox

# This package
pip install -e .
```

## Usage

### Python API

```python
from src.eris_voice import ErisVoice

# Initialize & load
voice = ErisVoice()
voice.load()

# Generate speech
voice.speak("ふふ...面白いことを言うわね。", output_path="output.wav")
```

### CLI

```bash
python -m src.eris_voice "こんにちは" -o hello.wav
```

### MLX Hybrid Pipeline (Recommended)

```bash
# 1. Convert weights (one-time setup)
python src/weight_converter.py

# 2. Run hybrid benchmark
python src/hybrid_benchmark.py --text "テスト"
```

## Project Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── decoder_weights_mlx.npz # MLX decoder weights
├── quantizer_weights_mlx.npz # MLX quantizer weights
├── docs/
│   ├── BENCHMARKS.md       # Detailed benchmark results
│   ├── TODO.md             # Optimization roadmap
│   └── OPTIMIZATION_SPEC.md # Technical specifications
├── src/
│   ├── eris_voice.py       # Main module (PyTorch)
│   ├── eris_voice_mlx.py   # MLX integration
│   ├── mlx_decoder_v2.py   # MLX Audio Decoder (45x faster)
│   ├── mlx_quantizer.py    # MLX Quantizer (3.5x faster)
│   ├── hybrid_benchmark.py # Hybrid pipeline benchmark
│   └── weight_converter.py # PyTorch → MLX conversion
├── tests/
│   ├── test_decoder.py     # Decoder unit tests
│   └── test_quantizer.py   # Quantizer unit tests
└── archive/                # Experimental/old code
```

## Technical Details

### MLX Migration Status

| Component | Status | Speedup |
|-----------|--------|---------|
| Audio Decoder (conv1d) | ✅ Complete | 45x |
| Quantizer (RVQ) | ✅ Complete | 3.5x |
| Weight Converter | ✅ Complete | - |
| Pre-Transformer | ⏸ Low Priority | (0.24s, <0.5%) |

### Architecture

```
PyTorch                          MLX
   │                              │
   ├── LLM (text → codes)         │
   ├── Quantizer decode ──────────┼── SplitResidualVectorQuantizerMLX
   ├── Pre-conv + upsample        │
   └── Decoder blocks ────────────┼── Qwen3TTSDecoderMLX (45x faster)
                                  │
                              Audio output
```

### Available Speakers

| Speaker | Description | Language |
|---------|-------------|----------|
| ono_anna | Playful Japanese female | Japanese ⭐ |
| vivian | Bright, edgy young female | Chinese |
| serena | Warm, gentle young female | Chinese |
| ryan | Dynamic male | English |
| aiden | Sunny American male | English |
| sohee | Warm female, rich emotion | Korean |

## Roadmap

Further optimization opportunities:

| Improvement | Expected Impact | Status |
|-------------|-----------------|--------|
| **Full MLX Native** | Remove PyTorch dependency, 2-3x faster | 🔬 Research |
| **Streaming Output** | Lower time-to-first-audio | 📋 Planned |
| **mlx-audio Integration** | Ecosystem contribution | 📋 Planned |
| **Standalone CLI** | `pip install eris-voice` | 📋 Planned |

## Limitations

- **MPS (Apple GPU)**: Audio Decoder exceeds conv1d 65536ch limit
- **float16**: May cause numerical instability on CPU (use bfloat16)
- **Pre-transformer**: Not yet ported to MLX (<0.5% of total time)

## Acknowledgments

This project builds upon:

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by Alibaba (Apache 2.0) - The underlying TTS model
- **[MLX](https://github.com/ml-explore/mlx)** by Apple - Machine learning framework for Apple Silicon

The MLX weight files (`decoder_weights_mlx.npz`, `quantizer_weights_mlx.npz`) are derived from Qwen3-TTS pretrained models under the Apache 2.0 license.

## Links

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [MLX](https://github.com/ml-explore/mlx)
- [Three Hearts Space](https://github.com/nao-amj/three_hearts_space)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Created by Eris 😈 @ Three Hearts Space*
