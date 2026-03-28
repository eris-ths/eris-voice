# TurboQuant / PolarQuant — KV Cache Quantization Research

> **Status**: Research Phase (2026-03-28)
> **Goal**: 3.5-bit KV cache quantization for eris-voice (9x memory compression)
> **Papers**: [TurboQuant](https://arxiv.org/abs/2504.19874), [PolarQuant](https://arxiv.org/abs/2502.02617), [QJL](https://arxiv.org/abs/2406.03482)

---

## Background

eris-voice の Talker (28-layer Transformer) は autoregressive generation で KV cache を蓄積する。
M3 8GB 環境では Metal buffer 上限 (4GB) により ~150 steps (~12.5秒) が上限。

KV cache を圧縮すれば:
- **同じメモリでより長い音声生成** (12.5秒 → 30秒+)
- **1.7B モデルの 8GB 動作** (現在 OOM)
- **将来的に RAG の Vector Search にも応用可能**

---

## TurboQuant の技術 (Two-Stage)

### Stage 1: PolarQuant

1. **ランダム回転**: KV ベクトルに直交行列 Q を適用 → outlier が均一化
2. **極座標変換**: 回転後のベクトルを (r, θ₁, θ₂, ..., θ_{d-1}) に変換
3. **角度の量子化**: 回転後の角度成分が **Beta分布** に集中 → 少ない bit で高精度量子化

```
v → Q @ v → (r, θ₁...θₙ) → quantize(θ) → compressed
```

### Stage 2: QJL (Quantized Johnson-Lindenstrauss)

- Stage 1 の残差を 1-bit (±1) に圧縮
- Johnson-Lindenstrauss 変換により内積推定が保存される
- メモリオーバーヘッドほぼゼロ

### なぜ極座標が効くのか

通常の per-channel min-max 量子化は outlier に弱い。PolarQuant は:
1. ランダム回転で座標系を均一化（outlier 分散）
2. 極座標変換で角度成分が **解析的に既知の Beta 分布** に収束
3. 分布が集中してるから **少ない bit 数で MSE 最小** の量子化が可能

---

## 達成性能 (論文報告)

| bit数 | 結果 |
|-------|------|
| **3.5 bit/channel** | 品質完全ニュートラル（劣化なし） |
| 2.5 bit/channel | わずかな品質低下のみ |
| 4-bit | H100 で最大 8x 性能向上 |
| KV cache 圧縮 | **6x 以上**のメモリ削減 |

---

## eris-voice への適用見積もり

### メモリ削減

| 手法 | KV Cache サイズ (2048 steps) | 圧縮率 |
|------|------|------|
| 現行 (float32) | 224 MB | 1.0x |
| MLX quantize (4-bit) | 28 MB | 7.9x |
| **PolarQuant (3.5-bit)** | **25 MB** | **9.0x** |
| PolarQuant (2.5-bit) | 18 MB | 12.5x |

### 実装コンポーネント

| コンポーネント | 実装量 | MLX対応 |
|---|---|---|
| ランダム直交行列 | 2h | `mx.linalg.qr()` ✅ |
| 極座標変換 | 3h | sin/cos/arctan ✅ (atan2 なし → workaround) |
| 角度量子化 | 3h | `mx.quantize()` ベース |
| 逆変換 (dequantize) | 3h | 対称実装 |
| Attention 統合 | 4h | KVCache subclass |
| テスト・検証 | 3h | 品質回帰テスト |
| **合計** | **~18h** | |

### 既知の制約

- `mx.atan2` が MLX に存在しない → `mx.arctan(y/x)` + 符号補正で代替
- QR 分解は CPU stream 必須 → 初回のみ、< 1ms
- コード未公開 → 論文からの再実装

---

## RAG Vector Search への応用

TurboQuant の論文自体が nearest neighbor search でも成果を報告:
- Product Quantization を recall で上回る
- indexing time ≈ 0 (codebook 学習不要)

**KV cache と Vector Search は本質的に同じ操作**:
- 大量のベクトルを圧縮保持して query との内積を高精度に推定する

Gemini Embedding 2 (3072次元) 等の高次元ベクトル圧縮にも直接応用可能。

---

## 実装計画

### Phase 1: PolarQuant MVP (Week 1)
1. `PolarQuantKVCache` クラス作成
2. ランダム回転 + 極座標変換
3. 3.5-bit 量子化
4. 品質検証 (max abs diff)

### Phase 2: Attention 統合 (Week 2)
1. `mlx_talker.py` の Attention に hook
2. KV 書き込み時に quantize、読み出し時に dequantize
3. メモリ・レイテンシ ベンチマーク

### Phase 3: 評価・Go/No-Go (Week 2)
1. 音声品質の聴取テスト
2. RTF 影響測定
3. 1.7B モデルでのテスト (16GB 環境)

---

## 参考

- [Google Research Blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [HuggingFace MLX prototype: flovflo/turboquant-mlx-qwen35-kv](https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv)

---

*Written by Eris 😈 - 2026-03-28*
