# Eris Voice - Qwen3-TTS for Apple Silicon

> エリスの声を Apple Silicon で生成する 😈

Qwen3-TTS を CUDA なしで動作させる最適化実装。

## 🎯 特徴

- **CUDA 不要**: Apple Silicon (M1/M2/M3/M4) の CPU で動作
- **最適化済み**: 0.6B + bfloat16 で 20倍以上高速化
- **Ono_Anna**: 日本人女性のプリセット声を使用
- **instruct対応**: スタイル指示でエリスらしさを表現

## 📊 ベンチマーク

| 設定 | 生成時間 (5文字) | 生成時間 (28文字) |
|------|-----------------|------------------|
| 1.7B + fp32 | 367秒 | 推定30分+ |
| **0.6B + bf16** | **16秒** | **98秒** |

## 🚀 インストール

```bash
# 依存関係
pip install qwen-tts soundfile torch

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

## ⚠️ 制限事項

- **MPS (Apple GPU)**: Audio Decoder が conv1d > 65536ch 制限に引っかかる
- **float16**: CPU で数値不安定になる場合あり（bfloat16 推奨）
- **リアルタイム**: 0.077x（約13倍遅い）なのでリアルタイム用途には不向き

## 🔧 技術詳細

### プロファイル結果

- **conv1d (オーディオデコード)**: 71% ← ボトルネック
- **LLM生成**: 13%
- **Linear層**: 9%

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
├── OPTIMIZATION_SPEC.md    # 高速化仕様書
├── requirements.txt
├── setup.py
└── src/
    └── eris_voice.py       # メインモジュール
```

## 🔗 関連リンク

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Three Hearts Space](https://github.com/nao-amj/three_hearts_space)

---

*Created by Eris 😈 - 2026-01-23*
