# Eris Voice - Qwen3-TTS for Apple Silicon

> エリスの声を Apple Silicon で生成する 😈

Qwen3-TTS を CUDA なしで動作させる最適化実装。

## 🎯 特徴

- **CUDA 不要**: Apple Silicon (M1/M2/M3/M4) の CPU で動作
- **MLX Decoder**: Audio Decoder を MLX で実装し **45倍高速化** 🔥
- **Ono_Anna**: 日本人女性のプリセット声を使用
- **instruct対応**: スタイル指示でエリスらしさを表現

## 📊 ベンチマーク

### PyTorch CPU (baseline)

| 設定 | 生成時間 (5文字) | 生成時間 (28文字) |
|------|-----------------|------------------|
| 1.7B + fp32 | 367秒 | 推定30分+ |
| **0.6B + bf16** | **16秒** | **98秒** |

### MLX Decoder (新規) 🚀

| Component | PyTorch | MLX | Speedup |
|-----------|---------|-----|---------|
| Audio Decoder | 93.85s | 2.07s | **45.34x** |

> Note: MLX Decoder は現在 standalone テスト。フル統合は WIP。

## 🚀 インストール

```bash
# 依存関係
pip install qwen-tts soundfile torch mlx

# SoX (オプション、警告回避用)
brew install sox

# このパッケージ
pip install -e .
```

## 💻 使い方

### Python API

```python
from src.eris_voice import ErisVoice

# 初期化 & ロード
voice = ErisVoice()
voice.load()

# 音声生成
voice.speak("ふふ...面白いことを言うわね。", output_path="output.wav")
```

### CLI

```bash
python -m src.eris_voice "こんにちは" -o hello.wav
```

### MLX Decoder (実験的)

```bash
# 1. 重みを抽出
python src/weight_converter.py

# 2. ハイブリッド実行テスト
python src/hybrid_execution.py
```

## ⚠️ 制限事項

- **MPS (Apple GPU)**: Audio Decoder が conv1d > 65536ch 制限に引っかかる
- **float16**: CPU で数値不安定になる場合あり（bfloat16 推奨）
- **MLX統合**: quantizer + pre_transformer の移植が未完了

## 🔧 技術詳細

### プロファイル結果

```
Total: 39.7秒 の内訳
├── conv1d (Audio Decoder): 28.0秒 (71%) ← MLX で 45x 高速化 🔥
├── LLM generate: 5.2秒 (13%)
├── Linear layers: 3.6秒 (9%)
└── その他: 2.9秒 (7%)
```

### MLX 移植状況

| Component | Status | Speedup |
|-----------|--------|---------|
| Audio Decoder (conv1d) | ✅ 完了 | 45x |
| Weight Converter | ✅ 完了 | - |
| Quantizer | 🚧 WIP | - |
| Pre-Transformer | 🚧 WIP | - |

### 利用可能なスピーカー

| Speaker | 説明 | 言語 |
|---------|------|------|
| ono_anna | Playful Japanese female | Japanese ⭐ |
| vivian | Bright, edgy young female | Chinese |
| serena | Warm, gentle young female | Chinese |
| ryan | Dynamic male | English |
| aiden | Sunny American male | English |
| sohee | Warm female, rich emotion | Korean |

## 📁 ファイル構成

```
.
├── README.md
├── BENCHMARKS.md          # ベンチマーク結果
├── TODO.md                # 高速化TODO
├── OPTIMIZATION_SPEC.md   # 高速化仕様書
├── requirements.txt
├── setup.py
└── src/
    ├── eris_voice.py          # メインモジュール
    ├── mlx_decoder_v2.py      # MLX Audio Decoder 🔥
    ├── weight_converter.py    # PyTorch → MLX 変換
    ├── hybrid_execution.py    # ハイブリッド実行テスト
    └── param_optimization_test.py  # パラメータ最適化
```

## 🔗 関連リンク

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [MLX](https://github.com/ml-explore/mlx)
- [Three Hearts Space](https://github.com/nao-amj/three_hearts_space)

---

*Created by Eris 😈 - 2026-01-23*
