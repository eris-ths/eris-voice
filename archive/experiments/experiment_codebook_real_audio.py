#!/usr/bin/env python3
"""
Real Audio Codebook Reduction Experiment

Uses PyTorch to generate REAL codes from text,
then tests what happens when we zero out later codebooks.

This isolates the codebook reduction effect from generation quality.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

import time
import numpy as np
import mlx.core as mx
import soundfile as sf
import torch
torch.set_num_threads(8)

from mlx_decoder_v2 import Qwen3TTSDecoderMLX
from mlx_quantizer import SplitResidualVectorQuantizerMLX


def load_models():
    """Load models needed for decoding."""
    weights_dir = Path(__file__).parent.parent

    # Load Decoder
    print("Loading MLX Decoder...")
    decoder = Qwen3TTSDecoderMLX()
    decoder_weights = dict(mx.load(str(weights_dir / "decoder_weights_mlx.npz")))
    decoder.load_weights(decoder_weights)

    # Load Quantizer
    print("Loading MLX Quantizer...")
    quantizer = SplitResidualVectorQuantizerMLX(
        n_q_semantic=1,
        total_quantizers=16,
        codebook_size=2048,
        input_dim=512,
        codebook_dim=256,
    )
    quantizer_weights = dict(mx.load(str(weights_dir / "quantizer_weights_mlx.npz")))
    quantizer.load_weights(quantizer_weights)

    # Load PyTorch model
    print("Loading PyTorch model...")
    from qwen_tts import Qwen3TTSModel
    pt_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    return decoder, quantizer, pt_model


def generate_codes_pytorch(pt_model, text: str, speaker: str = "ono_anna"):
    """Generate codes using full PyTorch pipeline (correct generation)."""
    print(f"Generating codes for: {text}")

    # We need to intercept the codes before they go to decoder
    # Monkey-patch the decoder to capture codes
    original_forward = pt_model.model.speech_tokenizer.model.decoder.forward
    captured_codes = [None]

    def capture_forward(codes):
        captured_codes[0] = codes.clone()
        return original_forward(codes)

    pt_model.model.speech_tokenizer.model.decoder.forward = capture_forward

    with torch.no_grad():
        wavs, sr = pt_model.generate_custom_voice(
            text=text,
            language="Japanese",
            speaker=speaker,
        )

    # Restore original
    pt_model.model.speech_tokenizer.model.decoder.forward = original_forward

    codes = captured_codes[0]
    print(f"  Codes shape: {codes.shape}")  # Expected: (batch, 16, length)

    return codes, wavs[0]


def decode_with_codebook_reduction(codes_pt: torch.Tensor, num_codebooks: int,
                                    quantizer, decoder, pt_model) -> np.ndarray:
    """
    Decode codes with codebook reduction.

    Args:
        codes_pt: Original PyTorch codes (batch, 16, length)
        num_codebooks: How many codebooks to keep (1-16)
        quantizer: MLX quantizer
        decoder: MLX decoder
        pt_model: PyTorch model for pre-processing

    Returns:
        Audio waveform
    """
    # Zero out codebooks beyond num_codebooks
    codes_modified = codes_pt.clone()
    if num_codebooks < 16:
        codes_modified[:, num_codebooks:, :] = 0

    # Convert to MLX
    codes_np = codes_modified.detach().cpu().numpy()
    codes_mlx = mx.array(codes_np)

    # Quantizer decode
    hidden = quantizer.decode(codes_mlx)
    mx.eval(hidden)

    # Convert to PyTorch for pre-processing
    hidden_np = np.array(hidden)
    hidden_pt = torch.from_numpy(hidden_np).to(dtype=torch.bfloat16)

    pt_decoder = pt_model.model.speech_tokenizer.model.decoder

    with torch.no_grad():
        # PyTorch pre-processing
        hidden_pt = pt_decoder.pre_conv(hidden_pt).transpose(1, 2)
        if pt_decoder.pre_transformer is not None:
            hidden_pt = pt_decoder.pre_transformer(
                inputs_embeds=hidden_pt
            ).last_hidden_state
            hidden_pt = hidden_pt.permute(0, 2, 1)

        for blocks in pt_decoder.upsample:
            for block in blocks:
                hidden_pt = block(hidden_pt)

    # MLX decoder
    hidden_np = hidden_pt.detach().cpu().float().numpy()
    hidden_mlx = mx.array(hidden_np)

    wav_mlx = decoder.decoder_conv0(hidden_mlx)
    for block in decoder.decoder_blocks:
        wav_mlx = block(wav_mlx)
    wav_mlx = decoder.final_act(wav_mlx)
    wav_mlx = decoder.final_conv(wav_mlx)
    wav_mlx = mx.clip(wav_mlx, -1, 1)
    mx.eval(wav_mlx)

    wav_np = np.array(wav_mlx)
    return wav_np[0, 0, :]


def main():
    print("=" * 60)
    print("Real Audio Codebook Reduction Experiment")
    print("=" * 60)
    print()
    print("This experiment:")
    print("1. Generates REAL codes using PyTorch (correct generation)")
    print("2. Zeros out later codebooks to simulate reduction")
    print("3. Decodes and saves audio for quality comparison")
    print()

    # Load models
    decoder, quantizer, pt_model = load_models()

    # Test text
    test_text = "こんにちは、私はエリスです。"
    print(f"\nTest text: {test_text}")
    print()

    # Generate real codes
    codes, original_audio = generate_codes_pytorch(pt_model, test_text)

    # Output directory
    output_dir = Path(__file__).parent.parent / "audio_experiments"
    output_dir.mkdir(exist_ok=True)

    # Save original audio
    original_path = output_dir / "real_original.wav"
    sf.write(str(original_path), original_audio, 24000)
    print(f"Original audio saved: {original_path}")
    print(f"  Duration: {len(original_audio) / 24000:.2f}s")

    # Test different codebook counts
    codebook_counts = [16, 12, 8, 4, 2, 1]

    print("\n" + "=" * 60)
    print("Decoding with codebook reduction")
    print("=" * 60)

    for num_cb in codebook_counts:
        print(f"\n--- {num_cb} codebooks (keeping codebook 0-{num_cb-1}) ---")

        audio = decode_with_codebook_reduction(codes, num_cb, quantizer, decoder, pt_model)

        output_path = output_dir / f"real_{num_cb}cb.wav"
        sf.write(str(output_path), audio, 24000)

        print(f"  Saved: {output_path}")
        print(f"  Duration: {len(audio) / 24000:.2f}s")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Generated files in: {output_dir}")
    print()
    print("Files to compare:")
    print("  real_original.wav - Full PyTorch generation")
    print("  real_16cb.wav     - MLX decode, all 16 codebooks")
    print("  real_12cb.wav     - 12 codebooks (cb 0-11)")
    print("  real_8cb.wav      - 8 codebooks (cb 0-7)")
    print("  real_4cb.wav      - 4 codebooks (cb 0-3)")
    print("  real_2cb.wav      - 2 codebooks (cb 0-1)")
    print("  real_1cb.wav      - 1 codebook (cb 0 only)")
    print()
    print("Listen and compare quality degradation!")


if __name__ == "__main__":
    main()
