#!/usr/bin/env python3
"""
True Streaming TTS Test

Test true streaming with step-by-step generation and decoding.
"""

import time
import numpy as np
import subprocess
import tempfile
import soundfile as sf

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mlx_pipeline import MLXFullPipeline


def test_true_streaming():
    """Test true streaming with step-by-step generation."""
    print("Loading pipeline...")
    pipeline = MLXFullPipeline()
    pipeline.load()

    # Warmup
    print("Warming up...")
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="balanced")
    pipeline.generate("こんにちは", speaker="ono_anna", quality_mode="balanced")
    print()

    # Test streaming
    text = "こんにちは。私はエリスよ。今日もあなたと話せて嬉しいわ。"
    print(f"Text: {text}")
    print()

    buffer_size = 5
    print(f"Buffer size: {buffer_size} steps")
    print()

    all_audio = []
    start = time.time()
    ttfa = None

    print("Streaming chunks:")
    for audio_chunk, info in pipeline.generate_streaming_steps(
        text,
        speaker="ono_anna",
        quality_mode="balanced",
        buffer_size=buffer_size,
    ):
        audio_duration = len(audio_chunk) / 24000
        all_audio.append(audio_chunk)

        if ttfa is None:
            ttfa = info['elapsed']

        status = "FINAL" if info['is_final'] else ""
        print(f"  Steps +{info['chunk_steps']:2d} = {info['total_steps']:3d}: "
              f"{audio_duration:.3f}s audio, "
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
    print(f"TTFA: {ttfa:.3f}s (prefill + first chunk generate + decode)")
    print(f"RTF: {total_duration / total_time:.2f}x")

    # Save and play
    fd, path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    sf.write(path, combined, 24000)
    print(f"\nSaved to: {path}")
    print("Playing...")
    subprocess.run(["afplay", path])


def test_comparison():
    """Compare streaming vs non-streaming."""
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

    # Non-streaming
    print("=" * 60)
    print("Non-streaming (baseline)")
    print("=" * 60)
    start = time.time()
    audio, gen_time = pipeline.generate(text, speaker="ono_anna", quality_mode="balanced")
    total_time = time.time() - start
    duration = len(audio) / 24000
    print(f"Audio: {duration:.2f}s")
    print(f"Time: {total_time:.2f}s")
    print(f"RTF: {duration / total_time:.2f}x")
    print(f"TTFA: {total_time:.2f}s (must wait for full generation)")
    print()

    # Streaming with different buffer sizes
    for buffer_size in [3, 5, 10]:
        print("=" * 60)
        print(f"True Streaming (buffer={buffer_size})")
        print("=" * 60)

        all_audio = []
        start = time.time()
        ttfa = None

        for audio_chunk, info in pipeline.generate_streaming_steps(
            text,
            speaker="ono_anna",
            quality_mode="balanced",
            buffer_size=buffer_size,
        ):
            all_audio.append(audio_chunk)
            if ttfa is None:
                ttfa = info['elapsed']

        total_time = time.time() - start
        combined = np.concatenate(all_audio)
        duration = len(combined) / 24000

        print(f"Audio: {duration:.2f}s")
        print(f"Time: {total_time:.2f}s")
        print(f"RTF: {duration / total_time:.2f}x")
        print(f"TTFA: {ttfa:.3f}s")
        improvement = ((total_time - ttfa) / total_time) * 100
        print(f"TTFA Improvement: {improvement:.1f}% earlier audio start")
        print()


def test_play_streaming():
    """Test streaming with immediate playback."""
    print("Loading pipeline...")
    pipeline = MLXFullPipeline()
    pipeline.load()

    # Warmup
    print("Warming up...")
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="balanced")
    print()

    text = "こんにちは。私はエリスよ。今日もあなたと話せて嬉しいわ。"
    print(f"Text: {text}")
    print()
    print("Playing with streaming (you should hear audio before full generation)...")
    print()

    buffer_size = 5
    start = time.time()
    ttfa = None
    all_audio = []

    for audio_chunk, info in pipeline.generate_streaming_steps(
        text,
        speaker="ono_anna",
        quality_mode="balanced",
        buffer_size=buffer_size,
        play_immediately=True,
    ):
        all_audio.append(audio_chunk)
        if ttfa is None:
            ttfa = info['elapsed']
            print(f"  First audio at {ttfa:.3f}s!")

    total_time = time.time() - start
    print()
    print(f"Total time: {total_time:.2f}s")
    print(f"TTFA: {ttfa:.3f}s")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        test_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "--play":
        test_play_streaming()
    else:
        test_true_streaming()
