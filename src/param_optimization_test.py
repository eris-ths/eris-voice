"""
生成パラメータ最適化テスト

max_new_tokens, temperature, top_p などを調整して速度改善を測定
"""

import torch
import time
import os

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

print("=== 生成パラメータ最適化テスト ===\n")

from qwen_tts import Qwen3TTSModel

# ベースライン設定
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cpu",
    dtype=torch.bfloat16,
)

text = "こんにちは"
baseline_params = {
    "language": "Japanese",
    "speaker": "ono_anna",
}

# テストケース
test_cases = [
    # ベースライン（デフォルト）
    {"name": "baseline", "params": {}},

    # max_new_tokens 削減
    {"name": "max_tokens_800", "params": {"max_new_tokens": 800}},
    {"name": "max_tokens_600", "params": {"max_new_tokens": 600}},

    # temperature 調整
    {"name": "temp_0.5", "params": {"temperature": 0.5}},
    {"name": "temp_0.3", "params": {"temperature": 0.3}},

    # top_p 調整
    {"name": "top_p_0.7", "params": {"top_p": 0.7}},
    {"name": "top_p_0.5", "params": {"top_p": 0.5}},

    # 組み合わせ
    {"name": "optimized", "params": {
        "max_new_tokens": 600,
        "temperature": 0.5,
        "top_p": 0.7,
    }},
]

results = []

for case in test_cases:
    name = case["name"]
    params = {**baseline_params, **case["params"]}

    print(f"\nテスト: {name}")
    print(f"  パラメータ: {case['params']}")

    start = time.time()
    try:
        wavs, sr = model.generate_custom_voice(text=text, **params)
        gen_time = time.time() - start
        duration = len(wavs[0]) / sr

        results.append({
            "name": name,
            "time": gen_time,
            "duration": duration,
            "rtf": duration / gen_time,
        })
        print(f"  時間: {gen_time:.1f}s, 音声: {duration:.2f}s")

    except Exception as e:
        print(f"  エラー: {e}")
        results.append({"name": name, "time": None, "error": str(e)})

# 結果サマリー
print("\n" + "=" * 50)
print("結果サマリー")
print("=" * 50)

baseline_time = next((r["time"] for r in results if r["name"] == "baseline"), None)

for r in results:
    if r.get("time"):
        speedup = baseline_time / r["time"] if baseline_time else 1
        print(f"{r['name']:20} | {r['time']:5.1f}s | {speedup:.2f}x")
    else:
        print(f"{r['name']:20} | ERROR: {r.get('error', 'unknown')}")
