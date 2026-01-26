#!/usr/bin/env python3
"""
Hybrid Streaming: Generate streaming + Cumulative decode

Strategy:
- Generate codes step-by-step (streaming)
- Decode ALL accumulated codes each time
- Only play the NEW portion (delta)

This maintains audio quality while providing streaming TTFA.
"""

import time
import numpy as np
import mlx.core as mx
import subprocess
import tempfile
import soundfile as sf
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mlx_pipeline import MLXFullPipeline
from mlx_generate import GenerateConfig


def generate_hybrid_streaming(
    pipeline,
    text: str,
    speaker: str = "ono_anna",
    quality_mode: str = "balanced",
    buffer_size: int = 5,
):
    """
    Generate with hybrid streaming: generate streaming + cumulative decode.

    Yields (new_audio_chunk, info) where new_audio_chunk is only the NEW audio
    since the last yield (not redundantly decoded audio).
    """
    if not pipeline._loaded:
        pipeline.load()

    if not pipeline._warmed_up:
        pipeline.warmup()

    start = time.time()

    # Extract embeddings
    initial_embeds, trailing_text_hidden, tts_pad_embed = pipeline._extract_pytorch_inputs(
        text, speaker
    )

    config = GenerateConfig(
        max_new_tokens=500,
        temperature=0.7,
        top_p=1.0,
        do_sample=True,
        quality_mode=quality_mode,
    )

    # Accumulate all codes
    all_codes = []
    prev_audio_len = 0

    for codes_chunk, is_final in pipeline.generator.generate_streaming(
        input_embeds=initial_embeds,
        config=config,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        buffer_size=buffer_size,
    ):
        # Accumulate codes
        chunk_np = np.array(codes_chunk)
        all_codes.append(chunk_np)

        # Stack all accumulated codes
        accumulated = np.concatenate(all_codes, axis=1)
        accumulated_mx = mx.array(accumulated)

        # Decode ALL accumulated codes
        full_audio = pipeline._decode_codes(accumulated_mx)

        # Extract only the NEW portion
        new_audio = full_audio[prev_audio_len:]
        prev_audio_len = len(full_audio)

        elapsed = time.time() - start
        total_steps = accumulated.shape[1]

        yield new_audio, {
            'chunk_steps': codes_chunk.shape[1],
            'total_steps': total_steps,
            'elapsed': elapsed,
            'total_audio_len': len(full_audio),
            'is_final': is_final,
        }


def test_hybrid():
    """Test hybrid streaming."""
    print("Loading pipeline...")
    pipeline = MLXFullPipeline()
    pipeline.load()

    # Warmup
    print("Warming up...")
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="balanced")
    pipeline.generate("こんにちは", speaker="ono_anna", quality_mode="balanced")
    print()

    text = "こんにちは。私はエリスよ。"
    print(f"Text: {text}")
    print()

    buffer_size = 5
    print(f"Buffer size: {buffer_size} steps")
    print()

    all_audio = []
    start = time.time()
    ttfa = None

    print("Hybrid streaming chunks:")
    for audio_chunk, info in generate_hybrid_streaming(
        pipeline, text, buffer_size=buffer_size
    ):
        audio_duration = len(audio_chunk) / 24000
        all_audio.append(audio_chunk)

        if ttfa is None:
            ttfa = info['elapsed']

        status = "FINAL" if info['is_final'] else ""
        print(f"  Steps +{info['chunk_steps']:2d} = {info['total_steps']:3d}: "
              f"new_audio={audio_duration:.3f}s, "
              f"total={info['total_audio_len']/24000:.2f}s, "
              f"elapsed={info['elapsed']:.3f}s {status}")

    total_time = time.time() - start
    combined = np.concatenate(all_audio)
    total_duration = len(combined) / 24000

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total audio: {total_duration:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"TTFA: {ttfa:.3f}s")
    print(f"RTF: {total_duration / total_time:.2f}x")

    # Save and play
    fd, path = tempfile.mkstemp(suffix="_hybrid.wav")
    os.close(fd)
    sf.write(path, combined, 24000)
    print(f"\nSaved: {path}")
    print("Playing...")
    subprocess.run(["afplay", path])


def test_comparison():
    """Compare hybrid vs chunked vs non-streaming."""
    print("Loading pipeline...")
    pipeline = MLXFullPipeline()
    pipeline.load()

    # Warmup
    print("Warming up...")
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="balanced")
    pipeline.generate("こんにちは", speaker="ono_anna", quality_mode="balanced")
    print()

    text = "こんにちは。私はエリスよ。今日もあなたと話せて嬉しいわ。"
    print(f"Text: {text}")
    print()

    # 1. Non-streaming (baseline)
    print("=" * 60)
    print("1. Non-streaming (baseline)")
    print("=" * 60)
    start = time.time()
    audio_baseline, _ = pipeline.generate(text, speaker="ono_anna", quality_mode="balanced")
    time_baseline = time.time() - start
    print(f"Audio: {len(audio_baseline)/24000:.2f}s")
    print(f"Time: {time_baseline:.2f}s")
    print(f"TTFA: {time_baseline:.2f}s")
    print(f"RTF: {len(audio_baseline)/24000/time_baseline:.2f}x")

    fd, path_baseline = tempfile.mkstemp(suffix="_baseline.wav")
    os.close(fd)
    sf.write(path_baseline, audio_baseline, 24000)
    print()

    # 2. Chunked streaming (has artifacts)
    print("=" * 60)
    print("2. Chunked streaming (has boundary artifacts)")
    print("=" * 60)
    all_audio = []
    start = time.time()
    ttfa = None

    for audio_chunk, info in pipeline.generate_streaming_steps(
        text, speaker="ono_anna", quality_mode="balanced", buffer_size=5
    ):
        all_audio.append(audio_chunk)
        if ttfa is None:
            ttfa = info['elapsed']

    time_chunked = time.time() - start
    audio_chunked = np.concatenate(all_audio)
    print(f"Audio: {len(audio_chunked)/24000:.2f}s")
    print(f"Time: {time_chunked:.2f}s")
    print(f"TTFA: {ttfa:.3f}s")
    print(f"RTF: {len(audio_chunked)/24000/time_chunked:.2f}x")

    fd, path_chunked = tempfile.mkstemp(suffix="_chunked.wav")
    os.close(fd)
    sf.write(path_chunked, audio_chunked, 24000)
    print()

    # 3. Hybrid streaming (no artifacts)
    print("=" * 60)
    print("3. Hybrid streaming (cumulative decode - no artifacts)")
    print("=" * 60)
    all_audio = []
    start = time.time()
    ttfa = None

    for audio_chunk, info in generate_hybrid_streaming(
        pipeline, text, buffer_size=5
    ):
        all_audio.append(audio_chunk)
        if ttfa is None:
            ttfa = info['elapsed']

    time_hybrid = time.time() - start
    audio_hybrid = np.concatenate(all_audio)
    print(f"Audio: {len(audio_hybrid)/24000:.2f}s")
    print(f"Time: {time_hybrid:.2f}s")
    print(f"TTFA: {ttfa:.3f}s")
    print(f"RTF: {len(audio_hybrid)/24000/time_hybrid:.2f}x")

    fd, path_hybrid = tempfile.mkstemp(suffix="_hybrid.wav")
    os.close(fd)
    sf.write(path_hybrid, audio_hybrid, 24000)
    print()

    # Diff analysis
    print("=" * 60)
    print("Audio Quality Analysis")
    print("=" * 60)
    min_len = min(len(audio_baseline), len(audio_chunked), len(audio_hybrid))
    diff_chunked = np.abs(audio_baseline[:min_len] - audio_chunked[:min_len])
    diff_hybrid = np.abs(audio_baseline[:min_len] - audio_hybrid[:min_len])

    print(f"Difference from baseline:")
    print(f"  Chunked: mean={diff_chunked.mean():.6f}, max={diff_chunked.max():.6f}")
    print(f"  Hybrid:  mean={diff_hybrid.mean():.6f}, max={diff_hybrid.max():.6f}")
    print()

    # Play comparison
    print("=" * 60)
    print("Playing comparison")
    print("=" * 60)

    print("\n1. BASELINE (non-streaming)...")
    subprocess.run(["afplay", path_baseline])

    print("\n2. CHUNKED (has artifacts)...")
    subprocess.run(["afplay", path_chunked])

    print("\n3. HYBRID (should be clean)...")
    subprocess.run(["afplay", path_hybrid])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        test_comparison()
    else:
        test_hybrid()
