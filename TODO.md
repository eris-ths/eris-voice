# 高速化 TODO - M3 MacBook Air 8GB 向け

> **目標**: 8GB Unified Memory の M3 MacBook Air で実用的な速度を実現
> **現状**: 98秒/28文字 → 目標: 30秒以下

---

## 📊 現状ベンチマーク

| 設定 | 5文字 | 28文字 | メモリ |
|------|-------|--------|--------|
| 1.7B + fp32 + CPU | 367s | 推定30分+ | ~7GB |
| **0.6B + bf16 + CPU** | **16s** | **98s** | **~2GB** ✅ |

---

## 🎯 高速化アプローチ TODO

### Phase 1: 即座に試せる（難易度: 低）

- [ ] **量子化モデル試行**
  - `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit`
  - 期待: メモリ 1/4、速度 1.5-2x
  - 方法: モデルID変更のみ

- [ ] **生成パラメータ最適化**
  - `max_new_tokens` 削減
  - `temperature` 調整
  - `top_p` / `top_k` 調整

- [ ] **メモリマップドロード**
  - `torch.load(..., mmap=True)`
  - 8GB 環境でのスワップ回避

### Phase 2: コード修正必要（難易度: 中）

- [ ] **ハイブリッド MPS/CPU 実行**
  - LLM推論: MPS (GPU) - 現状動作確認済み
  - Audio Decoder: CPU only (65536ch制限回避)
  - 期待: 3-5x 高速化
  - 実装: qwen_tts のソース修正 or monkey patch

- [ ] **Audio Decoder 分割処理**
  - conv1d を小さいチャンクに分割
  - MPS の 65536ch 制限を回避
  - 期待: MPS でデコード可能に

- [ ] **KV Cache 最適化**
  - 生成中のキャッシュサイズ削減
  - 8GB でのメモリ効率改善

### Phase 3: 外部貢献必要（難易度: 高）

- [ ] **mlx-audio への Qwen3-TTS サポート追加**
  - Issue/PR: https://github.com/Blaizzy/mlx-audio
  - 期待: 5-10x 高速化（MLX ネイティブ）
  - 作業: アーキテクチャ実装

- [ ] **GGUF 変換 & llama.cpp 対応**
  - Audio Codec + LLM の特殊構造が障壁
  - 現時点では不可能

- [ ] **CoreML 変換**
  - Apple Neural Engine 活用
  - coremltools でのエクスポート検証

---

## 🔬 プロファイル詳細

```
ボトルネック分析 (39.7秒の内訳):
├── conv1d (Audio Decoder): 28.0s (71%) ← 最大のボトルネック
├── LLM generate: 5.2s (13%)
├── Linear layers: 3.6s (9%)
└── その他: 2.9s (7%)
```

**結論**: Audio Decoder の conv1d が 71% を占める
→ ここを MPS で動かせれば大幅改善

---

## 💡 8GB 環境での注意点

```yaml
メモリ使用量:
  0.6B bf16: ~1.2GB (モデル)
  KV Cache: ~0.5GB (生成中)
  Audio Buffer: ~0.3GB
  合計: ~2GB (余裕あり ✅)

  1.7B bf16: ~3.4GB (モデル)
  KV Cache: ~1.5GB
  合計: ~5GB (ギリギリ ⚠️)

推奨:
  - 0.6B モデルを使用
  - 他のアプリを閉じる
  - Activity Monitor でメモリ監視
```

---

## 📈 期待される改善ロードマップ

| Phase | 改善 | 28文字の予想時間 |
|-------|------|-----------------|
| 現状 | - | 98秒 |
| Phase 1 | 量子化 | ~60秒 |
| Phase 2 | ハイブリッド | ~30秒 |
| Phase 3 | MLX ネイティブ | ~10秒 |

---

## 🔗 参考リソース

- [mlx-community モデル一覧](https://huggingface.co/mlx-community?search=qwen3-tts)
- [mlx-audio GitHub](https://github.com/Blaizzy/mlx-audio)
- [PyTorch MPS ドキュメント](https://pytorch.org/docs/stable/notes/mps.html)

---

*Created by Eris 😈 - 2026-01-23*
