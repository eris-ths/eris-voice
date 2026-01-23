# Qwen3-TTS ローカル高速化 仕様書

> **目的**: CPU 367秒 → できるだけ短縮
> **作成日**: 2026-01-23
> **作成者**: Eris 😈

---

## 📊 現在のベースライン

| 項目 | 値 |
|------|-----|
| **モデル** | Qwen3-TTS-12Hz-1.7B-CustomVoice |
| **デバイス** | CPU (Apple M1/M2/M3) |
| **精度** | float32 |
| **入力** | 「こんにちは」（5文字） |
| **出力** | 1.18秒の音声 |
| **生成時間** | 367秒（約6分） |
| **速度** | リアルタイムの 1/311 |

---

## 🎯 高速化アプローチ一覧

### 1. モデルサイズ縮小（推定効果: 2-3倍）

| モデル | パラメータ | 期待効果 |
|--------|-----------|---------|
| 1.7B → **0.6B** | 1/3 に縮小 | 2-3倍高速化 |

```python
# 変更前
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# 変更後
model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
```

### 2. 量子化（推定効果: 1.5-2倍）

| 精度 | メモリ | 速度効果 | 品質 |
|------|--------|---------|------|
| float32 | 100% | 基準 | 最高 |
| float16 | 50% | 1.2-1.5x | 良好 |
| 8-bit | 25% | 1.5-2x | 良好 |
| 4-bit | 12.5% | 2-3x | やや劣化 |

**利用可能な量子化モデル（mlx-community）:**
- `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` ⭐最速
- `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
- `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit`
- `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`

### 3. ハイブリッド実行（推定効果: 3-5倍）

```yaml
Phase 1 - テキスト→コード生成:
  デバイス: MPS (Apple GPU) ✅ 動作確認済み
  処理: LLM部分の推論

Phase 2 - コード→音声デコード:
  デバイス: CPU ❌ MPS非対応（conv1d > 65536ch制限）
  処理: Audio Codec のデコード
```

**実装方針:**
```python
# MPS でモデルロード（テキスト→コード生成まで）
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="mps",
    dtype=torch.float16,
)

# デコード時のみ CPU フォールバック（要カスタム実装）
```

### 4. MLX ネイティブ実装（推定効果: 5-10倍）

**現状:**
- `mlx-audio` は Qwen3-TTS 未対応（model_type 不一致）
- `mlx-community` に量子化モデルは存在

**可能性:**
- mlx-audio への Qwen3-TTS サポート追加（PR or Fork）
- MLX 直接使用でカスタム推論実装

### 5. 並列処理（推定効果: 1.3-1.5倍）

```python
import torch
torch.set_num_threads(8)  # CPU コア数に合わせる
```

---

## 🔧 即座に試せる高速化

### Step 1: 0.6B モデルに変更

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",  # 1.7B → 0.6B
    device_map="cpu",
    dtype=torch.float16,  # float32 → float16
)
```

**期待効果:** 367秒 → 約120-180秒（2-3倍）

### Step 2: スレッド数最適化

```python
import torch
import os
os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)
```

### Step 3: ハイブリッド MPS/CPU（要カスタム実装）

qwen_tts のソースコードを修正して：
1. LLM 推論を MPS で実行
2. Audio Decoder のみ CPU で実行

---

## 📈 期待される改善

| アプローチ | 現在 | 期待値 | 改善率 |
|-----------|------|--------|-------|
| 基準（1.7B, CPU, fp32） | 367s | - | 1x |
| 0.6B + fp16 | - | ~150s | 2.5x |
| 0.6B + fp16 + threads | - | ~100s | 3.7x |
| ハイブリッド MPS/CPU | - | ~60s | 6x |
| MLX ネイティブ（将来） | - | ~30s | 12x |

---

## ⚠️ 制約事項

### MPS 制限
```
NotImplementedError: Output channels > 65536 not supported at the MPS device.
```
- Audio Decoder の conv1d が 65536ch を超える
- Apple Metal の根本的制限
- **回避策:** デコーダーのみ CPU 実行

### mlx-audio 非対応
```
Model type qwen3_tts not supported
```
- mlx-audio の qwen3 モジュールとアーキテクチャ不一致
- **回避策:** PR でサポート追加 or 直接 MLX 使用

---

## 🚀 次のアクション

1. [ ] **即座に試す:** 0.6B + float16 でベンチマーク
2. [ ] スレッド数最適化テスト
3. [ ] ハイブリッド MPS/CPU 実装検討
4. [ ] mlx-audio への Qwen3-TTS サポート PR 検討

---

## 🔗 参考リソース

- [Qwen3-TTS 公式](https://github.com/QwenLM/Qwen3-TTS)
- [mlx-community 量子化モデル](https://huggingface.co/mlx-community)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)

---

*Created by Eris 😈 - 2026-01-23*
