"""
Hybrid Execution: PyTorch (LLM + Quantizer) → MLX (Decoder)

PyTorch で codes を生成し、MLX で音声にデコードする。
"""

import torch
import mlx.core as mx
import numpy as np
import time
import os
import soundfile as sf

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

print("=== Hybrid Execution Test ===")
print("PyTorch (LLM + Quantizer) → MLX (Decoder)\n")

# Load MLX decoder with weights
print("1. Loading MLX decoder...")
from mlx_decoder_v2 import Qwen3TTSDecoderMLX

mlx_decoder = Qwen3TTSDecoderMLX()
weights = dict(mx.load("decoder_weights_mlx.npz"))
mlx_decoder.load_weights(weights)
print("   MLX decoder ready!")

# Load PyTorch model
print("\n2. Loading PyTorch model...")
from qwen_tts import Qwen3TTSModel

pytorch_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cpu",
    dtype=torch.bfloat16,
)
print("   PyTorch model ready!")


def pytorch_generate_codes(model, text: str, speaker: str = "ono_anna"):
    """Generate audio codes using PyTorch model."""
    # Use the internal generate method to get codes before decoding
    # This is a simplified version - actual implementation may vary

    with torch.no_grad():
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="Japanese",
            speaker=speaker,
        )
    return wavs, sr


def decode_with_pytorch(model, codes):
    """Decode codes to audio using PyTorch."""
    with torch.no_grad():
        start = time.time()
        wav = model.model.speech_tokenizer.decode(codes)
        return time.time() - start, wav


def decode_with_mlx(decoder, hidden):
    """Decode hidden states to audio using MLX."""
    # Convert PyTorch tensor to MLX
    hidden_mlx = mx.array(hidden.detach().cpu().float().numpy())

    start = time.time()
    wav = decoder(hidden_mlx)
    mx.eval(wav)
    return time.time() - start, wav


# Test text
text = "こんにちは"
print(f"\n3. Testing with text: '{text}'")

# Full PyTorch execution (baseline)
print("\n--- Full PyTorch Execution (baseline) ---")
start_total = time.time()
wavs_pytorch, sr = pytorch_generate_codes(pytorch_model, text)
pytorch_total_time = time.time() - start_total

duration = len(wavs_pytorch[0]) / sr
print(f"Time: {pytorch_total_time:.1f}s")
print(f"Audio: {duration:.2f}s")
print(f"Efficiency: {pytorch_total_time/duration:.1f}s per second of audio")

# Save PyTorch result
sf.write("test_pytorch_baseline.wav", wavs_pytorch[0], sr)
print("Saved: test_pytorch_baseline.wav")

# Profiling PyTorch decoder separately
print("\n--- Profiling PyTorch Decoder ---")

# Get the internal decoder
decoder_pytorch = pytorch_model.model.speech_tokenizer.model.decoder

# Create synthetic codes for testing
# codes shape: (batch, num_quantizers, length)
# After quantizer.decode: hidden shape (batch, codebook_dim, length)
synthetic_hidden = torch.randn(1, 512, 100, dtype=torch.float32)

# Time PyTorch decoder
start = time.time()
with torch.no_grad():
    # We need to pass through the full decoder pipeline
    hidden = synthetic_hidden.to(torch.bfloat16)
    hidden = decoder_pytorch.pre_conv(hidden).transpose(1, 2)
    hidden = decoder_pytorch.pre_transformer(inputs_embeds=hidden).last_hidden_state
    hidden = hidden.permute(0, 2, 1)
    for blocks in decoder_pytorch.upsample:
        for block in blocks:
            hidden = block(hidden)
    wav = hidden
    for block in decoder_pytorch.decoder:
        wav = block(wav)
    wav = wav.clamp(min=-1, max=1)
pytorch_decoder_time = time.time() - start
print(f"PyTorch decoder time: {pytorch_decoder_time:.3f}s")

# Time MLX decoder with same input
print("\n--- Profiling MLX Decoder ---")
hidden_mlx = mx.array(synthetic_hidden.numpy())

start = time.time()
wav_mlx = mlx_decoder(hidden_mlx)
mx.eval(wav_mlx)
mlx_decoder_time = time.time() - start
print(f"MLX decoder time: {mlx_decoder_time:.3f}s")

# Comparison
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"PyTorch decoder: {pytorch_decoder_time:.3f}s")
print(f"MLX decoder:     {mlx_decoder_time:.3f}s")
if pytorch_decoder_time > 0:
    speedup = pytorch_decoder_time / mlx_decoder_time
    print(f"Speedup:         {speedup:.2f}x")
print("="*50)

print("\nNote: This test uses synthetic data for the decoder-only comparison.")
print("Full hybrid execution requires proper codes extraction from PyTorch.")
