# TurboQuant を Apple Silicon で実装して分かったこと

> 極座標変換による KV キャッシュ量子化を MLX で実装した記録。
> 論文の数学、実装の現実、そして正直な評価。

---

## 論文が言っていること

### 問題：KV キャッシュがメモリを食う

LLM の autoregressive generation では、過去の Key/Value ベクトルをキャッシュに保持する。
シーケンスが長くなるほどキャッシュは膨らみ、メモリがボトルネックになる。

従来の量子化（min-max スケーリング等）はブロックごとに scale と zero-point を
保存する必要があり、ブロックサイズによっては 1-2 bit/要素のオーバーヘッドが生じる。

### PolarQuant の着眼点（arXiv:2502.02617）

ランダムな直交行列で回転すると、高次元ベクトルの各座標は
**ほぼ独立な Beta 分布** に従い、**高度に集中する**。

この性質を使えば：
1. ブロックごとの scale/zero-point が **不要** になる
2. 角度成分の分布が解析的に既知だから、**最適な量子化器** を設計できる
3. 結果として、従来手法より少ない bit 数で同等の精度を達成する

### TurboQuant の二段構成（arXiv:2504.19874）

PolarQuant を Stage 1 として、残差に QJL（Quantized Johnson-Lindenstrauss）を
適用する Stage 2 を追加。

- **Stage 1（PolarQuant）**: MSE 最適な量子化。ただしバイアスが残る
- **Stage 2（QJL）**: 残差の符号ビット（+1/-1）だけを保存。内積推定のバイアスを消す

QJL（arXiv:2406.03482）は JL 変換 + sign-bit 量子化で、
scale/zero-point の保存オーバーヘッドを**完全にゼロ**にする。

論文報告値：
- **3.5 bit/channel で品質ニュートラル**（劣化なし）
- **2.5 bit/channel でわずかな劣化**
- KV キャッシュ 4.2x 以上の圧縮

---

## 実装して分かったこと

### 環境

- Apple M3 MacBook Air 8GB
- MLX 0.28.0
- 対象モデル：Qwen3-TTS Talker（28層、GQA、n_kv_heads=8、head_dim=128）

### 実装した範囲

論文の完全な再現ではなく、コアアイデアの MVP を実装した。

```
実装したもの:
  - ランダム直交回転（QR 分解）
  - 正規化（unit vector + radius 分離）
  - 正規化ベクトルの 4-bit 量子化（mx.quantize）
  - radius の float16 保存
  - 逆変換（dequantize → rescale → inverse rotate）

実装していないもの:
  - 角度空間での最適量子化（Beta 分布に基づく非一様量子化）
  - QJL 残差補正（Stage 2）
  - pre-allocated buffer（MVP は concat 方式）
```

### 結果

| 指標 | 値 |
|------|-----|
| 圧縮率 | **5.2x**（4-bit） |
| Cosine similarity | **0.997** |
| Mean abs error | **0.0065**（attention-like values） |
| Max abs error | **0.27-0.30**（ランダム値） |
| KV メモリ（28層, 100 steps） | 21.9 MB → **4.2 MB** |

### 何が簡単で、何が難しかったか

**簡単だったこと:**

ランダム直交行列の生成。MLX の `mx.linalg.qr` が CPU stream で動く。
直交性の検証（Q^T @ Q ≈ I）も max diff 6e-7 で問題なし。

正規化と量子化。radius を分離して unit vector を量子化する、
というコアロジックは 10 行程度で書ける。
`mx.quantize` / `mx.dequantize` がそのまま使える。

逆変換。Q が直交行列だから、逆回転は Q^T を掛けるだけ。

**見送ったこと:**

角度空間での最適量子化。論文の本質的な貢献はここにある。
ランダム回転後の座標が Beta 分布に従うという数学的性質を使って、
一様量子化より効率的な非一様量子化器を設計する。

これを実装するには Beta 分布の CDF の逆関数が必要で、
MLX にはネイティブの Beta 分布関数がない。
scipy で事前計算してテーブル化することは可能だが、MVP の範囲を超える。

私の実装は unit vector の各成分を [-1, 1] 範囲で一様に量子化している。
これは論文の手法より劣るが、cosine similarity 0.997 で実用水準。

QJL（Stage 2）。内積のバイアス補正。TTS の KV キャッシュでは
attention の softmax が相対的な大小関係を使うため、
バイアスが一様にかかる限り影響は軽微。
LLM の長文コンテキストではより重要になる。

---

## 正直な評価

### この手法が効く場面

1. **KV キャッシュが GB 級になる長文 LLM 推論**。
   ここが論文の本来のターゲット。2048+ トークンの長文で
   メモリが支配的になる場面で、5-9x の圧縮は実用的。

2. **RAG の Vector Search**。
   論文自体が nearest neighbor search でも成果を報告している。
   KV キャッシュと Vector Search は本質的に同じ操作
   ——大量のベクトルを保持して query との内積を推定する。
   codebook 学習不要（data-oblivious）だから、リアルタイムインデキシングが可能。

3. **メモリ制約の厳しいエッジデバイス**。
   Apple Silicon の統合メモリ環境で、モデルウェイトと KV キャッシュが
   同じメモリプールを奪い合う場面。

### 今回の実装：TTS の KV キャッシュで試した理由

eris-voice（Qwen3-TTS 0.6B）では KV キャッシュは 100 ステップで ~22 MB。
メモリのボトルネックは KV ではなく audio decoder 側にある。
つまり、この環境では KV 圧縮の直接的な恩恵は小さい。

それでも TTS の KV で試したのは、**本命の RAG Vector Search に持っていく前の
実験台として**。KV キャッシュと Vector Search は同じ数学だから、
ここで実装パターンと精度特性を掴んでおけば、
Embedding 圧縮への応用がスムーズになる。

### 論文の数学 vs 実装の現実

論文の最も重要な貢献は「回転後の座標が Beta 分布に従う」という
情報理論的な洞察。これにより最適量子化器の設計が可能になる。

私の実装はこの部分を省略して一様量子化で代替した。
それでも cosine similarity 0.997 が出るのは、
ランダム回転による座標の均一化 が単独でかなり強力だから。

言い換えると：
- **回転だけで大部分の効果が得られる**（outlier 分散 + 座標均一化）
- **Beta 分布最適量子化は追加の 10-20% 改善**（推定）
- **QJL は内積バイアス補正**（長文 LLM で重要、TTS では軽微）

---

## コード

```python
# Core: 10 行で書ける PolarQuant
def quantize(v, Q, bits=4, group_size=32):
    v_rot = v @ Q                                          # 回転
    radius = mx.sqrt((v_rot * v_rot).sum(axis=-1, keepdims=True))  # 半径
    v_unit = v_rot / (radius + 1e-8)                       # 正規化
    q, s, b = mx.quantize(v_unit, group_size=group_size, bits=bits)
    return q, s, b, radius.astype(mx.float16)

def dequantize(q, s, b, radius, Q, group_size=32, bits=4):
    v_unit = mx.dequantize(q, s, b, group_size=group_size, bits=bits)
    v_rot = v_unit * radius.astype(mx.float32)             # 復元
    return v_rot @ Q.T                                     # 逆回転
```

MLX のプリミティブだけで完結する。外部依存なし。

---

## 次にやるなら

1. **Beta 分布最適量子化の実装** — scipy で量子化テーブルを事前計算し、
   MLX の lookup で適用。圧縮率を 5.2x → 8-9x に改善できる見込み

2. **Attention 統合** — 現在は fetch 時に全キャッシュを dequantize している。
   MLX-LM の QuantizedKVCache パターンを参考に、
   `mx.fast.scaled_dot_product_attention` 内で直接 quantized KV を扱えれば
   dequantize オーバーヘッドが消える

3. **RAG Vector Search への応用** — Embedding ベクトル (3072次元) の圧縮。
   PQ (Product Quantization) との recall 比較が次の検証ポイント

---

## 参考文献

- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) (Han et al., AISTATS 2026)
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization](https://arxiv.org/abs/2406.03482) (Zandieh et al., NeurIPS 2024)
- [MLX Framework](https://github.com/ml-explore/mlx) (Apple)
- [MLX-LM QuantizedKVCache](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)

---

*Written based on hands-on implementation experience.*
*eris-voice: [github.com/eris-ths/eris-voice](https://github.com/eris-ths/eris-voice)*
