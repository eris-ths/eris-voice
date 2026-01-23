"""
Hybrid PyTorch + MLX Execution Test

LLM (text -> codes) を PyTorch で、Audio Decoder (codes -> wav) を MLX で実行。
"""

import torch
import mlx.core as mx
import numpy as np
import time
import os
import soundfile as sf

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

print("=== Hybrid PyTorch + MLX Test ===\n")


def measure_pytorch_decoder(model, codes):
    """Measure PyTorch decoder time."""
    start = time.time()
    with torch.no_grad():
        wav = model.model.speech_tokenizer.decode(codes)
    return time.time() - start, wav


def pytorch_codes_to_mlx(codes: torch.Tensor) -> mx.array:
    """Convert PyTorch codes tensor to MLX array."""
    return mx.array(codes.detach().cpu().numpy())


def mlx_wav_to_numpy(wav: mx.array) -> np.ndarray:
    """Convert MLX wav to numpy for saving."""
    return np.array(wav)


# Load PyTorch model
print("Loading PyTorch model...")
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cpu",
    dtype=torch.bfloat16,
)

# Generate codes (text -> codes) with PyTorch
text = "こんにちは"
print(f"\n1. Generating codes for: '{text}'")

start_total = time.time()
start_codes = time.time()

with torch.no_grad():
    # Get codes without decoding
    result = model.model.generate(
        input_ids=model.tokenizer(
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt"
        ).input_ids,
        max_new_tokens=1200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

codes_time = time.time() - start_codes
print(f"   Codes generation: {codes_time:.1f}s")

# 実際の音声生成（参照用）
print("\n2. Full generation for reference...")
start_full = time.time()
wavs, sr = model.generate_custom_voice(
    text=text,
    language="Japanese",
    speaker="ono_anna",
)
full_time = time.time() - start_full

# 音声生成から decoder 部分を分離
print(f"\n3. Profiling decoder...")

# speech_tokenizer の decode 時間を測定
# Note: 実際の codes を取得するには内部処理が必要
# ここでは全体の処理時間を参考値として使用

duration = len(wavs[0]) / sr
print(f"\n=== 結果 ===")
print(f"テキスト: {text}")
print(f"音声長: {duration:.2f}s")
print(f"全体時間: {full_time:.1f}s")
print(f"Real-time factor: {duration/full_time:.3f}x")

# 音声保存
output_path = "/Users/hirohashi/Develop/three_hearts_space/_nao/work/qwen3_tts_eris_voice/test_hybrid_mlx.wav"
sf.write(output_path, wavs[0], sr)
print(f"Saved: {output_path}")

print("\n" + "="*50)
print("Note: Full MLX decoder integration requires:")
print("  1. Weight loading implementation")
print("  2. Quantizer integration")
print("  3. Transformer integration")
print("="*50)
