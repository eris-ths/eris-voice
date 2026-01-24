"""
Hybrid MPS/CPU Execution Test

LLM inference on MPS (Apple GPU), Audio Decoder on CPU.
"""

import torch
import soundfile as sf
import os
import time

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print("=== ハイブリッド MPS/CPU テスト ===\n")
print(f"MPS available: {torch.backends.mps.is_available()}")

from qwen_tts import Qwen3TTSModel

# Load model on MPS first
print("\nLoading model on MPS...")
start_load = time.time()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="mps",
    dtype=torch.float16,  # MPS prefers float16
)
load_time = time.time() - start_load
print(f"Model loaded in {load_time:.1f}s")

# The model structure:
# - model.model.text_encoder (LLM) - can use MPS
# - model.model.talker (code predictor) - can use MPS
# - model.model.speech_tokenizer (audio decoder) - needs CPU due to conv1d > 65536ch

# Generate - the MPS fallback should handle the decoder automatically
text = "こんにちは"
print(f"\nGenerating: {text}")
start_gen = time.time()

try:
    wavs, sr = model.generate_custom_voice(
        text=text,
        language="Japanese",
        speaker="ono_anna",
    )
    gen_time = time.time() - start_gen

    duration = len(wavs[0]) / sr
    print(f"\n=== 結果 ===")
    print(f"Audio: {duration:.2f}s")
    print(f"Time: {gen_time:.1f}s")
    print(f"vs CPU baseline (16s): {16/gen_time:.2f}x faster" if gen_time < 16 else f"vs CPU baseline (16s): slower")

    output_path = "/Users/hirohashi/Develop/three_hearts_space/_nao/work/qwen3_tts_eris_voice/test_hybrid.wav"
    sf.write(output_path, wavs[0], sr)
    print(f"Saved: {output_path}")

except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
