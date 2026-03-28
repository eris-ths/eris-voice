#!/usr/bin/env python3
"""
MLX Pre-Decoder for Qwen3-TTS

Replaces the PyTorch pre-conv + pre-transformer + upsample pipeline.
This is the last PyTorch dependency in the audio decode path.

Architecture:
    Quantizer output (1, 512, seq)
      → pre_conv: CausalConv1d(512→1024, kernel=3)
      → pre_transformer: 8-layer Transformer (hidden=512) with LayerScale + RoPE
      → upsample: 2 stages of ConvTranspose1d + ConvNeXtBlock
    → MLX Audio Decoder (existing)
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ─── CausalConv1d ───

class CausalConv1d:
    """Causal Conv1d with left-padding. Input/output: (B, C, L)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        # MLX conv1d weight: (out_channels, kernel_size, in_channels)
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Args: x (B, C, L). Returns: (B, C_out, L)."""
        x = x.transpose(0, 2, 1)  # (B, L, C)
        x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        x = mx.conv1d(x, self.weight, stride=1, padding=0)
        x = x + self.bias
        return x.transpose(0, 2, 1)  # (B, C_out, L)


class CausalConv1dGrouped:
    """Causal Conv1d with groups (depthwise). Input/output: (B, C, L)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1):
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.groups = groups
        # For depthwise (groups=C): weight shape (C, K, 1) in MLX format
        c_per_group = in_channels // groups
        self.weight = mx.zeros((out_channels, kernel_size, c_per_group))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Args: x (B, C, L). Returns: (B, C_out, L)."""
        x = x.transpose(0, 2, 1)  # (B, L, C)
        x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        x = mx.conv1d(x, self.weight, stride=1, padding=0, groups=self.groups)
        x = x + self.bias
        return x.transpose(0, 2, 1)  # (B, C_out, L)


# ─── LayerScale ───

class LayerScale:
    """Per-channel scaling (learned)."""

    def __init__(self, dim: int):
        self.scale = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


# ─── RoPE for Decoder ───

class DecoderRoPE:
    """Rotary Position Embedding for decoder (simpler than Talker)."""

    def __init__(self, dim: int, base: float = 10000.0):
        self.dim = dim
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self._inv_freq = inv_freq

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """x: (B, n_heads, L, head_dim)"""
        seq_len = x.shape[2]
        positions = mx.arange(offset, offset + seq_len).astype(mx.float32)
        freqs = mx.outer(positions, self._inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)

        half = self.dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = mx.concatenate([
            x1 * cos[..., :half] - x2 * sin[..., :half],
            x1 * sin[..., half:] + x2 * cos[..., half:],
        ], axis=-1)
        return rotated


# ─── Decoder Attention ───

class DecoderAttention:
    """Attention for pre-decoder transformer. No QK norm (Identity)."""

    def __init__(self, hidden_size: int = 512, num_heads: int = 16):
        self.hidden_size = hidden_size
        self.n_heads = num_heads
        self.head_dim = hidden_size * 2 // num_heads  # q/k/v proj to 1024, then split
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.o_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        self.rope = DecoderRoPE(self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=None
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# ─── Decoder MLP ───

class DecoderMLP:
    """SwiGLU MLP for decoder."""

    def __init__(self, hidden_size: int = 512, intermediate_size: int = 1024):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ─── Decoder Transformer Block ───

class DecoderTransformerBlock:
    """Single transformer block with LayerScale."""

    def __init__(self, hidden_size: int = 512, intermediate_size: int = 1024, num_heads: int = 16):
        self.self_attn = DecoderAttention(hidden_size, num_heads)
        self.mlp = DecoderMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.self_attn_layer_scale = LayerScale(hidden_size)
        self.mlp_layer_scale = LayerScale(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        r = self.self_attn(self.input_layernorm(x))
        h = x + self.self_attn_layer_scale(r)
        r = self.mlp(self.post_attention_layernorm(h))
        return h + self.mlp_layer_scale(r)


# ─── CausalTransConvNet (Upsample) ───

class CausalTransConv1d:
    """Causal transposed convolution (upsample by 2)."""

    def __init__(self, channels: int = 1024, kernel_size: int = 2):
        # MLX conv_transpose1d weight: (in_channels, kernel_size, out_channels)
        self.weight = mx.zeros((channels, kernel_size, channels))
        self.bias = mx.zeros((channels,))
        self.stride = 2
        self.kernel_size = kernel_size

    def __call__(self, x: mx.array) -> mx.array:
        """x: (batch, channels, length) → (batch, channels, length*2)"""
        x = x.transpose(0, 2, 1)  # (B, L, C)
        x = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=0)
        x = x + self.bias
        return x.transpose(0, 2, 1)


# ─── ConvNeXtBlock ───

class ConvNeXtBlock:
    """
    ConvNeXt block matching PyTorch Qwen3TTSTokenizerV2ConvNeXtBlock:
        dwconv(B,C,L) → permute(0,2,1) → LayerNorm → pwconv1 → GELU → pwconv2 → gamma → permute → residual
    """

    def __init__(self, channels: int = 1024, kernel_size: int = 7):
        self.channels = channels
        self.kernel_size = kernel_size
        self.gamma = mx.ones((channels,))
        # Depthwise CausalConv: input (B, C, L), groups=C
        self.dwconv = CausalConv1dGrouped(channels, channels, kernel_size, groups=channels)
        self.norm_weight = mx.zeros((channels,))
        self.norm_bias = mx.zeros((channels,))
        self.pwconv1_weight = mx.zeros((channels * 4, channels))
        self.pwconv1_bias = mx.zeros((channels * 4,))
        self.pwconv2_weight = mx.zeros((channels, channels * 4))
        self.pwconv2_bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (batch, channels, length)"""
        residual = x

        # Depthwise causal conv: (B, C, L) → (B, C, L)
        x = self.dwconv(x)

        # Permute to (B, L, C) for LayerNorm and Linear
        x = x.transpose(0, 2, 1)

        # LayerNorm (eps=1e-6 matching PyTorch)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + 1e-6)
        x = x * self.norm_weight + self.norm_bias

        # Pointwise convs (linear layers)
        x = x @ self.pwconv1_weight.T + self.pwconv1_bias
        x = nn.gelu(x)
        x = x @ self.pwconv2_weight.T + self.pwconv2_bias

        # Scale
        x = self.gamma * x

        # Back to (B, C, L) + residual
        x = x.transpose(0, 2, 1)
        return residual + x


# ─── Full Pre-Decoder ───

class MLXPreDecoder:
    """
    MLX implementation of pre-conv + pre-transformer + upsample.

    Replaces PyTorch decoder for the intermediate processing step between
    quantizer decode and MLX audio decoder.
    """

    def __init__(self):
        # Pre-conv: CausalConv1d(512→1024, kernel=3)
        self.pre_conv = CausalConv1d(512, 1024, 3)

        # Pre-transformer: 8 layers, hidden=512, intermediate=1024
        self.input_proj = nn.Linear(1024, 512, bias=True)
        self.output_proj = nn.Linear(512, 1024, bias=True)
        self.layers = [DecoderTransformerBlock(512, 1024, 16) for _ in range(8)]
        self.norm = nn.RMSNorm(512, eps=1e-5)

        # Upsample: 2 stages
        self.upsample_stages = [
            (CausalTransConv1d(1024, 2), ConvNeXtBlock(1024, 7)),
            (CausalTransConv1d(1024, 2), ConvNeXtBlock(1024, 7)),
        ]

    def __call__(self, hidden: mx.array) -> mx.array:
        """
        Args: hidden: (batch, 512, seq_len) from quantizer
        Returns: (batch, 1024, seq_len * 4) ready for MLX audio decoder
        """
        # Pre-conv: (B, 512, L) → (B, 1024, L)
        hidden = self.pre_conv(hidden)

        # Pre-transformer: (B, 1024, L) → (B, L, 1024) → proj → transformer → proj → (B, 1024, L)
        hidden = hidden.transpose(0, 2, 1)  # (B, L, 1024)
        hidden = self.input_proj(hidden)  # (B, L, 512)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        hidden = self.output_proj(hidden)  # (B, L, 1024)
        hidden = hidden.transpose(0, 2, 1)  # (B, 1024, L)

        # Upsample: 2 stages × 2x = 4x total
        for trans_conv, conv_next in self.upsample_stages:
            hidden = trans_conv(hidden)
            hidden = conv_next(hidden)

        return hidden

    def load_weights(self, weights: Dict[str, mx.array]):
        """Load weights from extracted PyTorch model."""
        # Pre-conv
        # PyTorch Conv1d weight: (out, in, kernel) → MLX: (out, kernel, in)
        w = weights['pre_conv.conv.weight']
        self.pre_conv.weight = mx.array(w).transpose(0, 2, 1) if isinstance(w, np.ndarray) else w.transpose(0, 2, 1)
        self.pre_conv.bias = mx.array(weights['pre_conv.conv.bias']) if isinstance(weights['pre_conv.conv.bias'], np.ndarray) else weights['pre_conv.conv.bias']

        # Pre-transformer input/output proj
        self.input_proj.weight = mx.array(weights['pre_transformer.input_proj.weight'])
        self.input_proj.bias = mx.array(weights['pre_transformer.input_proj.bias'])
        self.output_proj.weight = mx.array(weights['pre_transformer.output_proj.weight'])
        self.output_proj.bias = mx.array(weights['pre_transformer.output_proj.bias'])
        self.norm.weight = mx.array(weights['pre_transformer.norm.weight'])

        # Transformer layers
        for i, layer in enumerate(self.layers):
            prefix = f'pre_transformer.layers.{i}'
            layer.self_attn.q_proj.weight = mx.array(weights[f'{prefix}.self_attn.q_proj.weight'])
            layer.self_attn.k_proj.weight = mx.array(weights[f'{prefix}.self_attn.k_proj.weight'])
            layer.self_attn.v_proj.weight = mx.array(weights[f'{prefix}.self_attn.v_proj.weight'])
            layer.self_attn.o_proj.weight = mx.array(weights[f'{prefix}.self_attn.o_proj.weight'])
            layer.mlp.gate_proj.weight = mx.array(weights[f'{prefix}.mlp.gate_proj.weight'])
            layer.mlp.up_proj.weight = mx.array(weights[f'{prefix}.mlp.up_proj.weight'])
            layer.mlp.down_proj.weight = mx.array(weights[f'{prefix}.mlp.down_proj.weight'])
            layer.input_layernorm.weight = mx.array(weights[f'{prefix}.input_layernorm.weight'])
            layer.post_attention_layernorm.weight = mx.array(weights[f'{prefix}.post_attention_layernorm.weight'])
            layer.self_attn_layer_scale.scale = mx.array(weights[f'{prefix}.self_attn_layer_scale.scale'])
            layer.mlp_layer_scale.scale = mx.array(weights[f'{prefix}.mlp_layer_scale.scale'])

        # Upsample stages
        for i, (trans_conv, conv_next) in enumerate(self.upsample_stages):
            prefix = f'upsample.{i}'
            # ConvTranspose weight: PyTorch (C_in, C_out, K) → MLX (C_out, K, C_in)
            w = weights[f'{prefix}.0.conv.weight']
            trans_conv.weight = mx.array(w).transpose(1, 2, 0) if isinstance(w, np.ndarray) else w.transpose(1, 2, 0)
            trans_conv.bias = mx.array(weights[f'{prefix}.0.conv.bias'])

            # ConvNeXt
            conv_next.gamma = mx.array(weights[f'{prefix}.1.gamma'])
            # Depthwise conv weight: PyTorch (C, 1, K) → MLX (C, K, 1)
            dw_w = weights[f'{prefix}.1.dwconv.conv.weight']
            conv_next.dwconv.weight = mx.array(dw_w).transpose(0, 2, 1) if isinstance(dw_w, np.ndarray) else dw_w.transpose(0, 2, 1)
            conv_next.dwconv.bias = mx.array(weights[f'{prefix}.1.dwconv.conv.bias'])
            conv_next.norm_weight = mx.array(weights[f'{prefix}.1.norm.weight'])
            conv_next.norm_bias = mx.array(weights[f'{prefix}.1.norm.bias'])
            conv_next.pwconv1_weight = mx.array(weights[f'{prefix}.1.pwconv1.weight'])
            conv_next.pwconv1_bias = mx.array(weights[f'{prefix}.1.pwconv1.bias'])
            conv_next.pwconv2_weight = mx.array(weights[f'{prefix}.1.pwconv2.weight'])
            conv_next.pwconv2_bias = mx.array(weights[f'{prefix}.1.pwconv2.bias'])

        print("Pre-decoder weights loaded successfully!")


# ─── Tests ───

def test_pre_decoder():
    """Test MLXPreDecoder shapes."""
    print("=" * 50)
    print("MLX Pre-Decoder Test")
    print("=" * 50)

    weights_dir = Path(__file__).parent.parent
    weights_path = weights_dir / 'pre_decoder_weights_mlx.npz'

    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        return

    pre_decoder = MLXPreDecoder()
    weights = dict(np.load(str(weights_path)))
    pre_decoder.load_weights(weights)

    # Test forward
    x = mx.random.normal((1, 512, 20))  # (batch, channels, seq)
    out = pre_decoder(x)
    mx.eval(out)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    # Expected: (1, 1024, 20*4=80)
    assert out.shape[0] == 1
    assert out.shape[1] == 1024
    print(f"Upsample factor: {out.shape[2] / x.shape[2]}x")

    print("\n✅ MLXPreDecoder test passed!")


if __name__ == "__main__":
    test_pre_decoder()
