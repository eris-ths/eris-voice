#!/usr/bin/env python3
"""
Test chunk boundary issues in streaming decode.

Compare:
1. Full decode (baseline)
2. Chunk decode (current streaming)
3. Overlap decode (potential fix)
"""

import time
import numpy as np
import mlx.core as mx
import torch
import subprocess
import tempfile
import soundfile as sf
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mlx_pipeline import MLXFullPipeline
from mlx_generate import GenerateConfig


def decode_full(pipeline, codes: mx.array) -> np.ndarray:
    """Decode all codes at once (baseline)."""
    return pipeline._decode_codes(codes)


def decode_chunked(pipeline, codes: mx.array, chunk_size: int = 5) -> np.ndarray:
    """Decode in chunks (current streaming method)."""
    codes_np = np.array(codes)
    total_steps = codes_np.shape[1]

    all_audio = []
    for i in range(0, total_steps, chunk_size):
        end_idx = min(i + chunk_size, total_steps)
        chunk = mx.array(codes_np[:, i:end_idx, :])
        audio_chunk = pipeline._decode_codes_chunk(chunk)
        all_audio.append(audio_chunk)

    return np.concatenate(all_audio)


def decode_overlapped(pipeline, codes: mx.array, chunk_size: int = 5, overlap: int = 2) -> np.ndarray:
    """
    Decode with overlap and crossfade to smooth boundaries.

    Strategy:
    - Decode chunk_size + overlap steps
    - Only keep chunk_size steps worth of audio
    - Crossfade at boundaries
    """
    codes_np = np.array(codes)
    total_steps = codes_np.shape[1]

    # Samples per step (at 12Hz, 24kHz: 24000/12 = 2000 samples/step)
    samples_per_step = 2000
    fade_samples = overlap * samples_per_step

    all_audio = []
    prev_tail = None

    for i in range(0, total_steps, chunk_size):
        # Decode with overlap
        start_idx = max(0, i - overlap) if i > 0 else 0
        end_idx = min(i + chunk_size + overlap, total_steps)

        chunk = mx.array(codes_np[:, start_idx:end_idx, :])
        audio_chunk = pipeline._decode_codes_chunk(chunk)

        # Calculate where our actual chunk starts in the decoded audio
        if i > 0:
            # Skip the overlap prefix
            prefix_steps = i - start_idx
            audio_start = prefix_steps * samples_per_step
        else:
            audio_start = 0

        # Calculate where our actual chunk ends
        actual_chunk_steps = min(chunk_size, total_steps - i)
        audio_end = audio_start + actual_chunk_steps * samples_per_step

        # Extract our portion
        audio_portion = audio_chunk[audio_start:audio_end]

        # Crossfade with previous chunk's tail
        if prev_tail is not None and len(prev_tail) > 0:
            fade_len = min(len(prev_tail), len(audio_portion), fade_samples)
            if fade_len > 0:
                # Create crossfade
                fade_out = np.linspace(1, 0, fade_len)
                fade_in = np.linspace(0, 1, fade_len)

                # Apply crossfade
                audio_portion[:fade_len] = (
                    prev_tail[-fade_len:] * fade_out +
                    audio_portion[:fade_len] * fade_in
                )

        all_audio.append(audio_portion)

        # Save tail for next iteration (if we have overlap)
        if end_idx < total_steps:
            suffix_steps = end_idx - (i + chunk_size)
            if suffix_steps > 0:
                prev_tail = audio_chunk[-suffix_steps * samples_per_step:]
            else:
                prev_tail = None
        else:
            prev_tail = None

    return np.concatenate(all_audio) if all_audio else np.zeros(0, dtype=np.float32)


def test_boundary():
    """Compare different decode methods."""
    print("Loading pipeline...")
    pipeline = MLXFullPipeline()
    pipeline.load()

    # Warmup
    print("Warming up...")
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="balanced")
    print()

    # Generate codes
    text = "こんにちは。私はエリスよ。"
    print(f"Text: {text}")
    print()

    # Get codes via generate
    initial_embeds, trailing_text_hidden, tts_pad_embed = pipeline._extract_pytorch_inputs(
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
        input_embeds=initial_embeds,
        config=config,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
    )

    # Find EOS and truncate
    codes_np = np.array(codes)
    codebook_0 = codes_np[0, :, 0]
    eos_positions = np.where(codebook_0 == 2150)[0]
    if len(eos_positions) > 0:
        codes_np = codes_np[:, :eos_positions[0], :]
    codes = mx.array(codes_np)

    print(f"Generated {codes.shape[1]} steps")
    print()

    # Compare decode methods
    print("=" * 60)
    print("Comparing decode methods")
    print("=" * 60)

    # 1. Full decode (baseline)
    print("\n1. Full decode (baseline):")
    audio_full = decode_full(pipeline, codes)
    print(f"   Samples: {len(audio_full)}, Duration: {len(audio_full)/24000:.2f}s")

    # Save
    fd, path_full = tempfile.mkstemp(suffix="_full.wav")
    os.close(fd)
    sf.write(path_full, audio_full, 24000)
    print(f"   Saved: {path_full}")

    # 2. Chunked decode (current)
    print("\n2. Chunked decode (current streaming):")
    audio_chunked = decode_chunked(pipeline, codes, chunk_size=5)
    print(f"   Samples: {len(audio_chunked)}, Duration: {len(audio_chunked)/24000:.2f}s")

    # Save
    fd, path_chunked = tempfile.mkstemp(suffix="_chunked.wav")
    os.close(fd)
    sf.write(path_chunked, audio_chunked, 24000)
    print(f"   Saved: {path_chunked}")

    # 3. Overlapped decode
    print("\n3. Overlapped decode (with crossfade):")
    audio_overlap = decode_overlapped(pipeline, codes, chunk_size=5, overlap=2)
    print(f"   Samples: {len(audio_overlap)}, Duration: {len(audio_overlap)/24000:.2f}s")

    # Save
    fd, path_overlap = tempfile.mkstemp(suffix="_overlap.wav")
    os.close(fd)
    sf.write(path_overlap, audio_overlap, 24000)
    print(f"   Saved: {path_overlap}")

    # Calculate difference
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    min_len = min(len(audio_full), len(audio_chunked), len(audio_overlap))

    diff_chunked = np.abs(audio_full[:min_len] - audio_chunked[:min_len])
    diff_overlap = np.abs(audio_full[:min_len] - audio_overlap[:min_len])

    print(f"\nDifference from baseline (lower is better):")
    print(f"  Chunked: mean={diff_chunked.mean():.6f}, max={diff_chunked.max():.6f}")
    print(f"  Overlap: mean={diff_overlap.mean():.6f}, max={diff_overlap.max():.6f}")

    # Play comparison
    print("\n" + "=" * 60)
    print("Playing comparison")
    print("=" * 60)

    print("\nPlaying FULL (baseline)...")
    subprocess.run(["afplay", path_full])

    print("\nPlaying CHUNKED (current - may have artifacts)...")
    subprocess.run(["afplay", path_chunked])

    print("\nPlaying OVERLAP (with crossfade)...")
    subprocess.run(["afplay", path_overlap])


if __name__ == "__main__":
    test_boundary()
