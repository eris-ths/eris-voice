#!/usr/bin/env python3
"""
Generate reference audio using PyTorch for comparison.
"""

import os
import sys
from pathlib import Path
import time

os.environ["OMP_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)
import soundfile as sf


def main():
    from qwen_tts import Qwen3TTSModel

    print("Loading PyTorch model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    text = "こんにちは。"
    speaker = "ono_anna"

    print(f"\nGenerating: '{text}'")
    print(f"Speaker: {speaker}")

    start = time.time()
    with torch.no_grad():
        result = model.generate_custom_voice(
            text=text,
            language="Japanese",
            speaker=speaker,
        )
    gen_time = time.time() - start

    # Handle tuple return
    if isinstance(result, tuple):
        audio = result[0]
    else:
        audio = result

    # Save audio
    import numpy as np
    output_path = Path(__file__).parent.parent / "test_pytorch_reference.wav"

    print(f"  Audio type: {type(audio)}")
    if hasattr(audio, 'shape'):
        print(f"  Audio shape: {audio.shape}")

    if hasattr(audio, 'cpu'):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = np.array(audio)

    # Ensure 1D
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()

    # Ensure float32
    audio_np = audio_np.astype(np.float32)

    print(f"  Audio np shape: {audio_np.shape}, dtype: {audio_np.dtype}")

    sf.write(str(output_path), audio_np, 24000)

    duration = len(audio_np) / 24000
    rtf = duration / gen_time if gen_time > 0 else 0

    print(f"\n  Audio duration: {duration:.2f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  RTF: {rtf:.2f}x")
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
