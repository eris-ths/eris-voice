#!/usr/bin/env python3
"""
MLX CodePredictor for Qwen3-TTS

5-layer Transformer that predicts remaining 15 codebook tokens
after Talker predicts the first codebook token.

Architecture:
    Talker → hidden_states → CodePredictor
                            ├── codec_embedding[0-14] (15 embeddings)
                            ├── 5 Transformer layers
                            ├── norm
                            └── lm_head[0-14] (15 heads)
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_talker import (
    TransformerBlock,
    create_attention_mask,
    TalkerConfig,
)


@dataclass
class CodePredictorConfig:
    """Configuration for CodePredictor."""
    hidden_size: int = 1024
    num_hidden_layers: int = 5
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    vocab_size: int = 2048  # codec vocabulary per codebook
    rope_theta: float = 1000000.0  # Same as Talker
    head_dim: int = 128  # Same as Talker (q_norm weight is (128,))
    num_codebooks: int = 15  # Predicts codebooks 1-15 (Talker does 0)
    max_position_embeddings: int = 32768


class Qwen3TTSCodePredictorMLX(nn.Module):
    """
    MLX implementation of Qwen3-TTS CodePredictor.

    Predicts 15 codebook tokens (indices 1-15) given:
    - Hidden states from Talker
    - Previous codec tokens for each codebook

    The Talker handles codebook 0, this handles 1-15.
    """

    def __init__(self, config: CodePredictorConfig):
        super().__init__()
        self.config = config

        # Codec embeddings for each codebook (15 total)
        self.codec_embedding = [
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_codebooks)
        ]

        # 5 Transformer layers (same structure as Talker but different config)
        # We reuse TransformerBlock but need to adjust for different head_dim
        # For now, create a compatible config
        talker_compatible_config = TalkerConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.layers = [TransformerBlock(talker_compatible_config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM heads for each codebook (15 total)
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_codebooks)
        ]

    def __call__(
        self,
        input_embeds: mx.array,
        cache: Optional[List] = None,
    ) -> mx.array:
        """
        Forward pass through transformer layers.

        Args:
            input_embeds: (batch, seq_len, hidden_size)
            cache: Optional KV cache

        Returns:
            Hidden states (batch, seq_len, hidden_size)
        """
        h = input_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0] if cache else None)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)

    def get_logits(self, hidden: mx.array, codebook_idx: int) -> mx.array:
        """
        Get logits for a specific codebook.

        Args:
            hidden: Hidden states (batch, seq_len, hidden_size)
            codebook_idx: Which codebook (0-14, corresponding to original 1-15)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        return self.lm_head[codebook_idx](hidden)

    def get_all_logits(self, hidden: mx.array) -> List[mx.array]:
        """
        Get logits for all codebooks.

        Args:
            hidden: Hidden states (batch, seq_len, hidden_size)

        Returns:
            List of logits, one per codebook
        """
        return [head(hidden) for head in self.lm_head]

    def embed_codes(self, codes: mx.array, codebook_idx: int) -> mx.array:
        """
        Embed codes for a specific codebook.

        Args:
            codes: Token indices (batch, seq_len)
            codebook_idx: Which codebook (0-14)

        Returns:
            Embeddings (batch, seq_len, hidden_size)
        """
        return self.codec_embedding[codebook_idx](codes)


def load_code_predictor_weights(
    model: Qwen3TTSCodePredictorMLX,
    weights_path: str,
) -> Qwen3TTSCodePredictorMLX:
    """
    Load weights from npz file into CodePredictor.

    Args:
        model: MLX CodePredictor model
        weights_path: Path to code_predictor_weights_mlx.npz

    Returns:
        Model with loaded weights
    """
    print(f"Loading CodePredictor weights from {weights_path}...")

    weights_np = dict(np.load(weights_path))
    weights_mlx = {k: mx.array(v) for k, v in weights_np.items()}

    # Load codec embeddings (15)
    for i in range(model.config.num_codebooks):
        key = f'codec_embedding.{i}.weight'
        if key in weights_mlx:
            model.codec_embedding[i].weight = weights_mlx[key]

    # Load transformer layers (5)
    for i, layer in enumerate(model.layers):
        prefix = f'layers.{i}'

        # Attention
        layer.self_attn.q_proj.weight = weights_mlx[f'{prefix}.self_attn.q_proj.weight']
        layer.self_attn.k_proj.weight = weights_mlx[f'{prefix}.self_attn.k_proj.weight']
        layer.self_attn.v_proj.weight = weights_mlx[f'{prefix}.self_attn.v_proj.weight']
        layer.self_attn.o_proj.weight = weights_mlx[f'{prefix}.self_attn.o_proj.weight']
        layer.self_attn.q_norm.weight = weights_mlx[f'{prefix}.self_attn.q_norm.weight']
        layer.self_attn.k_norm.weight = weights_mlx[f'{prefix}.self_attn.k_norm.weight']

        # MLP
        layer.mlp.gate_proj.weight = weights_mlx[f'{prefix}.mlp.gate_proj.weight']
        layer.mlp.up_proj.weight = weights_mlx[f'{prefix}.mlp.up_proj.weight']
        layer.mlp.down_proj.weight = weights_mlx[f'{prefix}.mlp.down_proj.weight']

        # Layer norms
        layer.input_layernorm.weight = weights_mlx[f'{prefix}.input_layernorm.weight']
        layer.post_attention_layernorm.weight = weights_mlx[f'{prefix}.post_attention_layernorm.weight']

    # Final norm
    model.norm.weight = weights_mlx['norm.weight']

    # Load LM heads (15)
    for i in range(model.config.num_codebooks):
        key = f'lm_head.{i}.weight'
        if key in weights_mlx:
            model.lm_head[i].weight = weights_mlx[key]

    print(f"Loaded {len(weights_mlx)} weight tensors")
    return model


# ============================================================
# Tests
# ============================================================

def test_code_predictor():
    """Test CodePredictor forward pass."""
    print("=" * 60)
    print("CodePredictor Tests")
    print("=" * 60)
    print()

    # Create model
    config = CodePredictorConfig()
    print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    print(f"  num_codebooks={config.num_codebooks}, vocab_size={config.vocab_size}")

    model = Qwen3TTSCodePredictorMLX(config)

    # Test forward pass (without loading weights)
    print("\nTest 1: Forward pass")
    batch_size = 1
    seq_len = 8
    x = mx.random.normal((batch_size, seq_len, config.hidden_size))

    hidden = model(x)
    mx.eval(hidden)
    print(f"  Input: {x.shape}")
    print(f"  Output: {hidden.shape}")
    assert hidden.shape == (batch_size, seq_len, config.hidden_size)

    # Test logits
    print("\nTest 2: Get logits")
    logits_0 = model.get_logits(hidden, 0)
    mx.eval(logits_0)
    print(f"  Logits for codebook 0: {logits_0.shape}")
    assert logits_0.shape == (batch_size, seq_len, config.vocab_size)

    # Test all logits
    print("\nTest 3: Get all logits")
    all_logits = model.get_all_logits(hidden)
    print(f"  Number of codebook logits: {len(all_logits)}")
    assert len(all_logits) == config.num_codebooks

    # Test embedding
    print("\nTest 4: Embed codes")
    codes = mx.array([[1, 2, 3, 4]])
    emb = model.embed_codes(codes, 0)
    mx.eval(emb)
    print(f"  Codes: {codes.shape} -> Embeddings: {emb.shape}")
    assert emb.shape == (1, 4, config.hidden_size)

    print("\n✅ Basic tests passed!")
    return True


def test_code_predictor_with_weights():
    """Test CodePredictor with actual weights."""
    print("\n" + "=" * 60)
    print("CodePredictor with Weights Test")
    print("=" * 60)
    print()

    weights_path = Path(__file__).parent.parent / 'code_predictor_weights_mlx.npz'
    if not weights_path.exists():
        print(f"⚠️ Weights not found: {weights_path}")
        print("  Run convert_talker_weights.py first")
        return False

    # Create and load model
    config = CodePredictorConfig()
    model = Qwen3TTSCodePredictorMLX(config)
    model = load_code_predictor_weights(model, str(weights_path))

    # Test forward pass
    print("\nTest: Forward pass with weights")
    batch_size = 1
    seq_len = 16
    x = mx.random.normal((batch_size, seq_len, config.hidden_size))

    import time
    start = time.time()
    hidden = model(x)
    mx.eval(hidden)
    elapsed = time.time() - start

    print(f"  Input: {x.shape}")
    print(f"  Output: {hidden.shape}")
    print(f"  Time: {elapsed * 1000:.1f}ms")

    # Get logits for all codebooks
    start = time.time()
    all_logits = model.get_all_logits(hidden)
    for logits in all_logits:
        mx.eval(logits)
    elapsed = time.time() - start

    print(f"  All logits time: {elapsed * 1000:.1f}ms")

    print("\n✅ Weights test passed!")
    return True


if __name__ == "__main__":
    test_code_predictor()
    test_code_predictor_with_weights()
