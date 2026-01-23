# ベンチマーク結果

> **環境**: M3 MacBook Air 8GB Unified Memory
> **日付**: 2026-01-23

---

## 📊 ベースライン比較

| 設定 | 5文字 | 28文字 | 備考 |
|------|-------|--------|------|
| 1.7B + fp32 + CPU | 367秒 | 推定30分+ | 基準 |
| **0.6B + bf16 + CPU** | **16秒** | **98秒** | ✅ 推奨 |

**改善率**: 22.9倍高速化

---

## 🧪 高速化アプローチ検証結果

### 量子化

| アプローチ | 結果 | 備考 |
|-----------|------|------|
| mlx-community 4-bit | ❌ 非互換 | MLX形式、PyTorch非対応 |
| mlx-community 8-bit | ❌ 非互換 | 同上 |
| PyTorch 動的量子化 | ❌ エラー | pickle エラー |

### コンパイル最適化

| アプローチ | 結果 | 備考 |
|-----------|------|------|
| torch.compile() | 37.9秒 | 2.37倍**遅い** ❌ |

### デバイス最適化

| アプローチ | 結果 | 備考 |
|-----------|------|------|
| MPS (Apple GPU) | ❌ デコーダーで失敗 | conv1d > 65536ch 制限 |
| Hybrid MPS/CPU | ❌ 数値エラー | float16 で inf/nan |
| CPU only + bf16 | ✅ 16秒 | **現状最速** |

---

## 📈 プロファイル結果

```
Total: 39.7秒 の内訳

conv1d (Audio Decoder): 28.0秒 (71%) ← ボトルネック
LLM generate: 5.2秒 (13%)
Linear layers: 3.6秒 (9%)
その他: 2.9秒 (7%)
```

---

## ✅ 結論

**現時点での最適設定:**

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cpu",
    dtype=torch.bfloat16,
)
```

**さらなる高速化には:**
- mlx-audio への Qwen3-TTS サポート追加が必要
- または Alibaba Cloud API ($0.1/10K文字) を使用

---

*Benchmarked by Eris 😈 - 2026-01-23*
