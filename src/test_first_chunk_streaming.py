#!/usr/bin/env python3
"""
First-Chunk Streaming: Best of both worlds

Strategy:
1. Generate first N steps → decode → play immediately (TTFA improvement)
2. Continue generating remaining steps
3. Decode ALL codes at once (no boundary artifacts)
4. Play remaining audio

This gives:
- Fast TTFA (first chunk plays quickly)
- High quality (bulk decode has no artifacts)
- Acceptable efficiency (only 2 decode calls)
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


def generate_first_chunk_streaming(
    pipeline,
    text: str,
    speaker: str = "ono_anna",
    quality_mode: str = "balanced",
    first_chunk_steps: int = 10,
):
    """
    Generate with first-chunk streaming.

    1. Yields first chunk immediately for fast TTFA
    2. Continues generating
    3. Yields complete audio (decoded all at once) for quality

    Yields:
        (audio_chunk, info_dict)
        - First yield: first chunk audio
        - Second yield: complete audio (all steps decoded together)
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

    # Collect all codes while yielding first chunk
    all_codes = []
    first_chunk_yielded = False

    for codes_chunk, is_final in pipeline.generator.generate_streaming(
        input_embeds=initial_embeds,
        config=config,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        buffer_size=first_chunk_steps,
    ):
        chunk_np = np.array(codes_chunk)
        all_codes.append(chunk_np)

        # Yield first chunk immediately for TTFA
        if not first_chunk_yielded:
            first_chunk_yielded = True

            # Decode just the first chunk
            first_audio = pipeline._decode_codes(codes_chunk)
            ttfa = time.time() - start

            yield first_audio, {
                'type': 'first_chunk',
                'steps': codes_chunk.shape[1],
                'ttfa': ttfa,
                'audio_duration': len(first_audio) / 24000,
            }

    # Now decode ALL codes together for quality
    all_codes_np = np.concatenate(all_codes, axis=1)
    all_codes_mx = mx.array(all_codes_np)

    # Full decode (no boundary artifacts)
    full_audio = pipeline._decode_codes(all_codes_mx)
    total_time = time.time() - start

    yield full_audio, {
        'type': 'complete',
        'total_steps': all_codes_np.shape[1],
        'total_time': total_time,
        'audio_duration': len(full_audio) / 24000,
    }


def test_first_chunk():
    """Test first-chunk streaming."""
    print("Loading pipeline...")
    pipeline = MLXFullPipeline()
    pipeline.load()

    # Warmup
    print("Warmup...")
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="balanced")
    print()

    text = "こんにちは。私はエリスよ。"
    print(f"Text: {text}")
    print()

    first_chunk_steps = 10
    print(f"First chunk: {first_chunk_steps} steps")
    print()

    for audio, info in generate_first_chunk_streaming(
        pipeline, text, first_chunk_steps=first_chunk_steps
    ):
        if info['type'] == 'first_chunk':
            print(f"First chunk ready!")
            print(f"  TTFA: {info['ttfa']:.3f}s")
            print(f"  Audio: {info['audio_duration']:.2f}s ({info['steps']} steps)")
            print()

            # Save and play first chunk
            fd, path = tempfile.mkstemp(suffix="_first.wav")
            os.close(fd)
            sf.write(path, audio, 24000)
            print("Playing first chunk...")
            subprocess.run(["afplay", path])

        elif info['type'] == 'complete':
            print(f"\nComplete audio ready!")
            print(f"  Total time: {info['total_time']:.2f}s")
            print(f"  Audio: {info['audio_duration']:.2f}s ({info['total_steps']} steps)")
            print(f"  RTF: {info['audio_duration']/info['total_time']:.2f}x")
            print()

            # Save complete audio
            fd, path = tempfile.mkstemp(suffix="_complete.wav")
            os.close(fd)
            sf.write(path, audio, 24000)
            print("Playing complete (high quality)...")
            subprocess.run(["afplay", path])


def test_quality_comparison():
    """Compare first chunk vs complete audio quality."""
    print("Loading pipeline...")
    pipeline = MLXFullPipeline()
    pipeline.load()

    print("Warmup...")
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="balanced")
    print()

    text = "こんにちは。私はエリスよ。"
    print(f"Text: {text}")
    print("=" * 60)

    first_audio = None
    complete_audio = None

    for audio, info in generate_first_chunk_streaming(
        pipeline, text, first_chunk_steps=10
    ):
        if info['type'] == 'first_chunk':
            first_audio = audio
            print(f"First chunk: {info['audio_duration']:.2f}s, TTFA={info['ttfa']:.3f}s")
        elif info['type'] == 'complete':
            complete_audio = audio
            print(f"Complete: {info['audio_duration']:.2f}s")

    # Compare first N samples
    if first_audio is not None and complete_audio is not None:
        min_len = min(len(first_audio), len(complete_audio))
        diff = np.abs(first_audio[:min_len] - complete_audio[:min_len])

        print()
        print("Quality comparison (first chunk vs complete):")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Max diff: {diff.max():.6f}")

        # Save both
        fd1, path1 = tempfile.mkstemp(suffix="_first_chunk.wav")
        os.close(fd1)
        sf.write(path1, first_audio, 24000)

        fd2, path2 = tempfile.mkstemp(suffix="_complete.wav")
        os.close(fd2)
        sf.write(path2, complete_audio, 24000)

        print()
        print(f"First chunk saved: {path1}")
        print(f"Complete saved: {path2}")

        print()
        print("Playing first chunk (may have slight boundary artifact at end)...")
        subprocess.run(["afplay", path1])

        print()
        print("Playing complete (high quality, no artifacts)...")
        subprocess.run(["afplay", path2])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quality":
        test_quality_comparison()
    else:
        test_first_chunk()
