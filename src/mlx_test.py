"""
MLX-Audio Qwen3-TTS Test

Test if mlx-audio can run Qwen3-TTS natively on Apple Silicon.
"""

import time

print("=== MLX-Audio Qwen3-TTS テスト ===\n")

# Check MLX availability
try:
    import mlx.core as mx
    print(f"MLX version: {mx.__version__}")
    print(f"Metal available: True")
except ImportError as e:
    print(f"MLX not available: {e}")
    exit(1)

# Load model
print("\nLoading model...")
start_load = time.time()

try:
    from mlx_audio.tts.utils import load_model

    # Try CustomVoice model for Japanese speaker support
    model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16")
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    # Generate speech
    text = "こんにちは"
    print(f"\nGenerating: {text}")
    start_gen = time.time()

    # Try with ono_anna speaker (Japanese)
    results = list(model.generate_custom_voice(
        text=text,
        language="Japanese",
        speaker="ono_anna",
        instruct="Playful and mischievous tone",
    ))

    gen_time = time.time() - start_gen

    if results:
        audio = results[-1]  # Get final audio
        duration = len(audio) / 24000  # Assume 24kHz sample rate
        print(f"\n=== 結果 ===")
        print(f"Audio: {duration:.2f}s")
        print(f"Time: {gen_time:.1f}s")
        print(f"vs PyTorch CPU baseline (16s): {16/gen_time:.2f}x faster" if gen_time < 16 else f"vs baseline: {gen_time/16:.2f}x slower")

        # Save audio
        import soundfile as sf
        output_path = "/Users/hirohashi/Develop/three_hearts_space/_nao/work/qwen3_tts_eris_voice/test_mlx.wav"
        sf.write(output_path, audio, 24000)
        print(f"Saved: {output_path}")
    else:
        print("No audio generated")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
