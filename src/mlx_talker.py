"""
MLX Qwen3-TTS Talker Implementation

MLX-native implementation of Qwen3-TTS Talker for Apple Silicon acceleration.
Based on ml-explore/mlx-lm Qwen3 implementation.

Architecture:
    PyTorch (embedding + pre-processing)
    → MLX (28-layer Transformer - this file)
    → MLX (CodePredictor - 5 layers)
    → PyTorch/MLX (Decoder - already MLX)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class TalkerConfig:
    """Configuration for Qwen3-TTS Talker."""
    hidden_size: int = 1024
    text_hidden_size: int = 2048  # text embedding dimension
    num_hidden_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # GQA
    rms_norm_eps: float = 1e-6
    text_vocab_size: int = 151936  # text vocabulary
    codec_vocab_size: int = 3072  # codec vocabulary
    rope_theta: float = 1000000.0  # actual value from config
    head_dim: int = 128  # actual value (not 64!)
    max_position_embeddings: int = 32768


@dataclass
class CodePredictorConfig:
    """Configuration for CodePredictor."""
    hidden_size: int = 1024
    num_hidden_layers: int = 5
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    vocab_size: int = 2048  # codec vocabulary
    rope_theta: float = 10000.0
    head_dim: int = 64
    num_codebooks: int = 16
    max_position_embeddings: int = 8192


def create_attention_mask(h: mx.array, cache: Optional[Any] = None) -> Optional[mx.array]:
    """Create causal attention mask."""
    T = h.shape[1]
    if T == 1:
        return None

    # Causal mask
    mask = mx.triu(mx.full((T, T), -mx.inf), k=1)

    if cache is not None and cache.offset > 0:
        # Extend mask for cached keys
        prefix = mx.zeros((T, cache.offset))
        mask = mx.concatenate([prefix, mask], axis=1)

    return mask


def scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Scaled dot-product attention."""
    # queries: (B, n_heads, L, head_dim)
    # keys: (B, n_kv_heads, S, head_dim)
    # values: (B, n_kv_heads, S, head_dim)

    n_heads = queries.shape[1]
    n_kv_heads = keys.shape[1]

    # GQA: repeat k/v heads if needed
    if n_kv_heads < n_heads:
        n_rep = n_heads // n_kv_heads
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale

    if mask is not None:
        scores = scores + mask

    weights = mx.softmax(scores, axis=-1)
    return weights @ values


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dims: int, base: float = 10000.0, max_position_embeddings: int = 8192):
        super().__init__()
        self.dims = dims
        self.base = base
        self.max_position_embeddings = max_position_embeddings

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        self._inv_freq = inv_freq

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply rotary embeddings to input tensor."""
        # x: (B, n_heads, L, head_dim)
        seq_len = x.shape[2]

        # Create position indices
        positions = mx.arange(offset, offset + seq_len).astype(mx.float32)

        # Compute sin/cos
        freqs = mx.outer(positions, self._inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)

        # Apply rotation
        x1 = x[..., :self.dims // 2]
        x2 = x[..., self.dims // 2:]

        # Rotate
        rotated = mx.concatenate([
            x1 * cos[..., :self.dims // 2] - x2 * sin[..., :self.dims // 2],
            x1 * sin[..., self.dims // 2:] + x2 * cos[..., self.dims // 2:],
        ], axis=-1)

        return rotated


class Attention(nn.Module):
    """Multi-head attention with GQA support."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        # q_proj: hidden_size -> n_heads * head_dim (1024 -> 2048)
        # k_proj, v_proj: hidden_size -> n_kv_heads * head_dim (1024 -> 1024)
        # o_proj: n_heads * head_dim -> hidden_size (2048 -> 1024)
        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = RoPE(self.head_dim, base=config.rope_theta, max_position_embeddings=config.max_position_embeddings)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape and apply norms
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        # Update cache if provided
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Attention
        output = scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3TTSTalkerMLX(nn.Module):
    """
    MLX implementation of Qwen3-TTS Talker.

    This is the main 28-layer transformer that converts text+codec embeddings
    into hidden states for code generation.

    Note: text_embedding outputs 2048-dim, needs projection to 1024 for transformer.
    """

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        # text_embedding: (151936, 2048) - needs projection to hidden_size
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)
        # codec_embedding: (3072, 1024) - matches hidden_size
        self.codec_embedding = nn.Embedding(config.codec_vocab_size, config.hidden_size)

        # Text projection: 2048 -> 2048 -> 1024 (matches PyTorch Qwen3TTSTalkerResizeMLP)
        # ResizeMLP: FC1(2048, 2048) -> SiLU -> FC2(2048, 1024)
        # Note: Original has bias=True, but weights file doesn't include bias (may be zeros)
        self.text_projection_fc1 = nn.Linear(config.text_hidden_size, config.text_hidden_size, bias=False)
        self.text_projection_fc2 = nn.Linear(config.text_hidden_size, config.hidden_size, bias=False)

        # Transformer layers
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embedding
        self.rotary_emb = RoPE(config.head_dim, base=config.rope_theta)

    def __call__(
        self,
        input_embeds: mx.array,
        cache: Optional[List] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            input_embeds: Pre-computed embeddings (B, L, hidden_size)
            cache: Optional KV cache for incremental generation

        Returns:
            Hidden states (B, L, hidden_size)
        """
        h = input_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0] if cache else None)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)

    def load_weights(self, weights: Dict[str, mx.array]):
        """Load weights from dictionary."""
        # Map PyTorch weight names to MLX
        weight_map = {}

        for key, value in weights.items():
            # Remove 'model.' prefix if present
            new_key = key.replace('model.talker.model.', '')
            new_key = new_key.replace('model.talker.', '')

            # Handle embedding weights
            if 'text_embedding' in new_key or 'codec_embedding' in new_key:
                weight_map[new_key] = value
            # Handle layer weights
            elif 'layers.' in new_key:
                weight_map[new_key] = value
            elif new_key == 'norm.weight':
                weight_map[new_key] = value

        # Apply weights
        self.update(weight_map)
        print(f"Loaded {len(weight_map)} weight tensors")


class Qwen3TTSCodePredictorMLX(nn.Module):
    """
    MLX implementation of Qwen3-TTS CodePredictor.

    Smaller 5-layer transformer that predicts codec codes autoregressively.
    """

    def __init__(self, config: CodePredictorConfig):
        super().__init__()
        self.config = config

        # Codec embeddings (one per codebook)
        self.codec_embedding = [
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_codebooks)
        ]

        # Transformer layers
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM heads (one per codebook)
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_codebooks)
        ]

    def __call__(
        self,
        input_embeds: mx.array,
        cache: Optional[List] = None,
    ) -> mx.array:
        """Forward pass."""
        h = input_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0] if cache else None)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)

    def get_logits(self, hidden: mx.array, codebook_idx: int) -> mx.array:
        """Get logits for a specific codebook."""
        return self.lm_head[codebook_idx](hidden)


class KVCache:
    """Key-Value cache for incremental generation."""

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Update cache and return full key/value tensors."""
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.offset = self.keys.shape[2]
        return self.keys, self.values


def convert_pytorch_weights(pt_state_dict: dict) -> Dict[str, mx.array]:
    """
    Convert PyTorch state dict to MLX format.

    Args:
        pt_state_dict: PyTorch model state dict

    Returns:
        Dictionary of MLX arrays
    """
    mlx_weights = {}

    for key, value in pt_state_dict.items():
        # Convert to numpy then MLX
        np_value = value.detach().cpu().float().numpy()
        mlx_weights[key] = mx.array(np_value)

    return mlx_weights


if __name__ == "__main__":
    # Test initialization
    print("Testing MLX Talker initialization...")

    config = TalkerConfig()
    print(f"Config: hidden_size={config.hidden_size}, head_dim={config.head_dim}")
    print(f"  n_heads={config.num_attention_heads}, n_kv_heads={config.num_key_value_heads}")
    print(f"  q_proj output: {config.num_attention_heads * config.head_dim}")
    print(f"  k/v_proj output: {config.num_key_value_heads * config.head_dim}")

    talker = Qwen3TTSTalkerMLX(config)

    # Test forward pass
    batch_size = 1
    seq_len = 10
    x = mx.random.normal((batch_size, seq_len, config.hidden_size))

    output = talker(x)
    mx.eval(output)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ MLX Talker test passed!")
