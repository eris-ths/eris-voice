"""
Hybrid Execution v2: Quantizer + Decoder (without pre_transformer)

PyTorch で codes を生成し、MLX で Quantizer + Decoder を実行。
NOTE: pre_transformer はスキップ（次のステップで移植）
"""

import torch
import mlx.core as mx
import numpy as np
import time
import os
import soundfile as sf

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

print("=== Hybrid Execution v2 Test ===")
print("PyTorch (LLM) → MLX (Quantizer + Decoder)\n")
print("NOTE: pre_transformer is SKIPPED in this test\n")

# Load MLX components
print("1. Loading MLX components...")
from mlx_decoder_v2 import Qwen3TTSDecoderMLX
from mlx_quantizer import SplitResidualVectorQuantizerMLX

mlx_quantizer = SplitResidualVectorQuantizerMLX()
mlx_quantizer.load_weights(dict(mx.load("quantizer_weights_mlx.npz")))

mlx_decoder = Qwen3TTSDecoderMLX()
mlx_decoder.load_weights(dict(mx.load("decoder_weights_mlx.npz")))
print("   MLX components ready!")

# Load PyTorch model
print("\n2. Loading PyTorch model...")
from qwen_tts import Qwen3TTSModel

pytorch_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cpu",
    dtype=torch.bfloat16,
)
print("   PyTorch model ready!")


def profile_pytorch_full(model, text: str, speaker: str = "ono_anna"):
    """Profile full PyTorch pipeline."""
    start = time.time()
    with torch.no_grad():
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="Japanese",
            speaker=speaker,
        )
    total_time = time.time() - start
    return wavs[0], sr, total_time


def profile_pytorch_components(model, codes):
    """Profile PyTorch Quantizer + Decoder separately."""
    decoder = model.model.speech_tokenizer.model.decoder

    with torch.no_grad():
        # Quantizer
        start = time.time()
        hidden = decoder.quantizer.decode(codes)
        quantizer_time = time.time() - start

        # Pre-conv
        start = time.time()
        hidden = decoder.pre_conv(hidden).transpose(1, 2)
        preconv_time = time.time() - start

        # Pre-transformer (skip profiling, just execute)
        start = time.time()
        hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
        pretrans_time = time.time() - start

        # Upsample + Decoder
        hidden = hidden.permute(0, 2, 1)
        start = time.time()
        for blocks in decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in decoder.decoder:
            wav = block(wav)
        decoder_time = time.time() - start

    return {
        "quantizer": quantizer_time,
        "preconv": preconv_time,
        "pretransformer": pretrans_time,
        "decoder": decoder_time,
    }


def profile_mlx_components(quantizer, decoder, codes_mlx):
    """Profile MLX Quantizer + Decoder."""

    # Quantizer
    start = time.time()
    hidden = quantizer.decode(codes_mlx)
    mx.eval(hidden)
    quantizer_time = time.time() - start

    # Decoder (includes pre_conv and upsample)
    start = time.time()
    wav = decoder(hidden)
    mx.eval(wav)
    decoder_time = time.time() - start

    return {
        "quantizer": quantizer_time,
        "decoder": decoder_time,
        "total": quantizer_time + decoder_time,
    }


# Test with synthetic codes
print("\n3. Testing with synthetic codes...")
batch, n_q, length = 1, 16, 50  # 50 tokens ≈ short utterance
codes_torch = torch.randint(0, 2048, (batch, n_q, length))
codes_mlx = mx.array(codes_torch.numpy())
print(f"   Input codes shape: {codes_torch.shape}")

# Warmup
print("\n4. Warmup runs...")
_ = profile_pytorch_components(pytorch_model, codes_torch)
_ = profile_mlx_components(mlx_quantizer, mlx_decoder, codes_mlx)

# Benchmark
print("\n5. Benchmarking...")

# PyTorch components
pytorch_times = profile_pytorch_components(pytorch_model, codes_torch)
print("\n--- PyTorch Component Times ---")
for name, t in pytorch_times.items():
    print(f"   {name}: {t:.4f}s")
pytorch_total = sum(pytorch_times.values())
print(f"   TOTAL: {pytorch_total:.4f}s")

# MLX components
mlx_times = profile_mlx_components(mlx_quantizer, mlx_decoder, codes_mlx)
print("\n--- MLX Component Times ---")
for name, t in mlx_times.items():
    print(f"   {name}: {t:.4f}s")

# Comparison (excluding pre_transformer)
print("\n" + "=" * 50)
print("COMPARISON (Quantizer + Decoder only)")
print("=" * 50)
pytorch_qd = pytorch_times["quantizer"] + pytorch_times["preconv"] + pytorch_times["decoder"]
mlx_qd = mlx_times["total"]
print(f"PyTorch (q + preconv + d): {pytorch_qd:.4f}s")
print(f"MLX (q + d):               {mlx_qd:.4f}s")
if mlx_qd > 0:
    speedup = pytorch_qd / mlx_qd
    print(f"Speedup:                   {speedup:.2f}x")

print("\n" + "=" * 50)
print("NOTE: pre_transformer not yet ported to MLX")
print(f"PyTorch pre_transformer:   {pytorch_times['pretransformer']:.4f}s")
print("=" * 50)

# Full pipeline test
print("\n6. Full PyTorch generation test...")
text = "こんにちは"
wav, sr, total_time = profile_pytorch_full(pytorch_model, text)
duration = len(wav) / sr
print(f"   Text: '{text}'")
print(f"   Total time: {total_time:.2f}s")
print(f"   Audio duration: {duration:.2f}s")
print(f"   Efficiency: {total_time/duration:.2f}s per second of audio")

sf.write("test_v2_pytorch.wav", wav, sr)
print("   Saved: test_v2_pytorch.wav")
