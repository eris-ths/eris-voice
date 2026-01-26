#!/usr/bin/env python3
"""
Streaming Buffer Size Optimization Test

Test different buffer sizes to find optimal trade-off between:
- Latency (time to first audio)
- Efficiency (overhead per decode call)
"""

import time
import numpy as np
import mlx.core as mx
import torch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mlx_pipeline import MLXFullPipeline
from mlx_generate import GenerateConfig


def decode_chunk(pipeline, codes_chunk: mx.array) -> np.ndarray:
    """
    Decode a chunk of codes to audio.

    Args:
        pipeline: MLXFullPipeline instance
        codes_chunk: (1, num_steps, 16) codec codes

    Returns:
        Audio waveform as numpy array
    """
    codes_np = np.array(codes_chunk)

    if codes_np.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)

    # Reshape: (1, num_steps, 16) → (1, 16, num_steps)
    codes = mx.array(np.transpose(codes_np, (0, 2, 1)))

    # MLX Quantizer decode
    hidden_mlx = pipeline.mlx_quantizer.decode(codes)
    mx.eval(hidden_mlx)
    hidden_np = np.array(hidden_mlx)

    # PyTorch pre-processing
    with torch.no_grad():
        hidden = torch.from_numpy(hidden_np).to(dtype=torch.bfloat16)

        # Pre-conv
        hidden = pipeline.pt_decoder.pre_conv(hidden).transpose(1, 2)

        # Pre-transformer (if exists)
        if pipeline.pt_decoder.pre_transformer is not None:
            hidden = pipeline.pt_decoder.pre_transformer(
                inputs_embeds=hidden
            ).last_hidden_state
            hidden = hidden.permute(0, 2, 1)

        # Upsample
        for blocks in pipeline.pt_decoder.upsample:
            for block in blocks:
                hidden = block(hidden)

        hidden_np = hidden.detach().cpu().float().numpy()

    # MLX final decoding
    hidden_mlx = mx.array(hidden_np)
    wav_mlx = pipeline.mlx_decoder.decoder_conv0(hidden_mlx)
    for block in pipeline.mlx_decoder.decoder_blocks:
        wav_mlx = block(wav_mlx)
    wav_mlx = pipeline.mlx_decoder.final_act(wav_mlx)
    wav_mlx = pipeline.mlx_decoder.final_conv(wav_mlx)
    mx.eval(wav_mlx)

    return np.array(wav_mlx[0, 0, :])


def test_buffer_sizes():
    """Test different buffer sizes for streaming decode."""
    print("Loading pipeline...")
    pipeline = MLXFullPipeline()
    pipeline.load()

    # Warmup
    print("Warming up...")
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="balanced")
    pipeline.generate("こんにちは", speaker="ono_anna", quality_mode="balanced")
    print()

    # Generate codes for testing (longer text for more steps)
    text = "こんにちは。私はエリスよ。今日もあなたと話せて嬉しいわ。"
    print(f"Generating codes for: {text}")

    # Get full generation to know total steps
    start = time.time()
    audio_full, gen_time = pipeline.generate(text, speaker="ono_anna", quality_mode="balanced")
    total_duration = len(audio_full) / 24000
    print(f"Full generation: {len(audio_full)} samples, {total_duration:.2f}s audio, {gen_time:.2f}s gen time")
    print()

    # Generate codes once for testing
    inputs_embeds, trailing_hidden, tts_pad_embed = pipeline._extract_pytorch_inputs(
        text, speaker="ono_anna"
    )

    config = GenerateConfig(
        max_new_tokens=500,
        temperature=0.7,
        top_p=1.0,
        do_sample=True,
        quality_mode="balanced",
    )
    codes = pipeline.generator.generate(
        input_embeds=inputs_embeds,
        config=config,
        trailing_text_hidden=trailing_hidden,
        tts_pad_embed=tts_pad_embed,
    )

    # Find EOS and truncate
    codes_np = np.array(codes)
    codebook_0 = codes_np[0, :, 0]
    eos_positions = np.where(codebook_0 == 2150)[0]
    if len(eos_positions) > 0:
        codes_np = codes_np[:, :eos_positions[0], :]
    codes = mx.array(codes_np)

    total_steps = codes.shape[1]
    print(f"Generated {total_steps} steps")
    print()

    # Test buffer sizes
    buffer_sizes = [1, 3, 5, 10, 15, 20, total_steps]

    print("=" * 70)
    print("Testing Buffer Sizes")
    print("=" * 70)
    print(f"{'Buffer':<10} {'Decode Calls':<15} {'Total Time':<12} {'TTFA':<12} {'Overhead':<12}")
    print("-" * 70)

    results = []
    for buffer_size in buffer_sizes:
        if buffer_size > total_steps:
            continue

        num_decodes = (total_steps + buffer_size - 1) // buffer_size

        # Decode in chunks and measure time
        start = time.time()
        all_audio = []
        ttfa = None

        for i in range(0, total_steps, buffer_size):
            end_idx = min(i + buffer_size, total_steps)
            chunk_codes = codes[:, i:end_idx, :]

            # Decode chunk
            audio_chunk = decode_chunk(pipeline, chunk_codes)

            if ttfa is None:
                ttfa = time.time() - start

            all_audio.append(audio_chunk)

        total_time = time.time() - start
        combined = np.concatenate(all_audio) if all_audio else np.zeros(0)
        audio_duration = len(combined) / 24000
        overhead = total_time - audio_duration

        results.append({
            'buffer': buffer_size,
            'decodes': num_decodes,
            'total_time': total_time,
            'ttfa': ttfa,
            'overhead': overhead,
            'audio_duration': audio_duration,
        })

        print(f"{buffer_size:<10} {num_decodes:<15} {total_time:<12.3f} {ttfa:<12.3f} {overhead:<12.3f}")

    print()
    print("=" * 70)
    print("Analysis")
    print("=" * 70)

    # Calculate metrics
    print(f"\nTotal steps: {total_steps}")
    print(f"Expected audio: {total_steps / 12:.2f}s (at 12Hz)")

    if results:
        # Find optimal
        print("\nRecommendations:")

        # Best TTFA
        best_ttfa = min(results, key=lambda x: x['ttfa'])
        print(f"  Lowest TTFA: buffer={best_ttfa['buffer']} ({best_ttfa['ttfa']:.3f}s)")

        # Best efficiency (lowest overhead)
        best_efficiency = min(results, key=lambda x: x['overhead'])
        print(f"  Lowest overhead: buffer={best_efficiency['buffer']} ({best_efficiency['overhead']:.3f}s)")

        # Find sweet spot: TTFA < 1s and reasonable overhead
        sweet_spots = [r for r in results if r['ttfa'] < 1.0 and r['overhead'] < 2.0]
        if sweet_spots:
            best_sweet = min(sweet_spots, key=lambda x: x['ttfa'] + x['overhead'])
            print(f"  Sweet spot: buffer={best_sweet['buffer']} (TTFA={best_sweet['ttfa']:.3f}s, overhead={best_sweet['overhead']:.3f}s)")


if __name__ == "__main__":
    test_buffer_sizes()
