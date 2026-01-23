"""
MLX Audio Decoder v2 for Qwen3-TTS

PyTorch の Audio Decoder を MLX で書き直し、重みロード機能付き。
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple


class SnakeBetaMLX(nn.Module):
    """Snake activation with learnable beta parameter."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = 1.0
        # beta shape: (1, channels, 1) for broadcasting
        self.beta = mx.ones((1, channels, 1))

    def __call__(self, x):
        # Ensure beta has correct shape for broadcasting
        beta = self.beta
        if beta.ndim == 1:
            beta = beta.reshape(1, -1, 1)
        elif beta.ndim == 2:
            beta = beta.reshape(1, beta.shape[0], 1)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(self.alpha * x), 2)


class CausalConv1dMLX(nn.Module):
    """Causal Conv1d for MLX."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        # MLX conv1d weight: (out_channels, kernel_size, in_channels // groups)
        self.weight = mx.zeros((out_channels, kernel_size, in_channels // groups))
        self.bias = mx.zeros((out_channels,))

        self.effective_kernel = (kernel_size - 1) * dilation + 1
        self.padding = self.effective_kernel - stride

    def __call__(self, x):
        # x: (batch, channels, length)
        # Causal padding
        x = mx.pad(x, [(0, 0), (0, 0), (self.padding, 0)])

        # MLX conv1d expects (batch, length, channels)
        x = mx.transpose(x, (0, 2, 1))

        x = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

        x = x + self.bias
        x = mx.transpose(x, (0, 2, 1))
        return x


class CausalTransConv1dMLX(nn.Module):
    """Causal Transposed Conv1d for MLX."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # MLX conv_transpose1d weight: (out_channels, kernel_size, in_channels)
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = pad - self.left_pad

    def __call__(self, x):
        # x: (batch, channels, length)
        x = mx.transpose(x, (0, 2, 1))

        x = mx.conv_transpose1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,
        )

        x = x + self.bias

        # Trim padding
        if self.right_pad > 0:
            x = x[:, self.left_pad:-self.right_pad, :]
        else:
            x = x[:, self.left_pad:, :]

        x = mx.transpose(x, (0, 2, 1))
        return x


class ConvNeXtBlockMLX(nn.Module):
    """ConvNeXt block for MLX."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = CausalConv1dMLX(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.ones((dim,)) * 1e-6

    def __call__(self, x):
        residual = x
        x = self.dwconv(x)
        x = mx.transpose(x, (0, 2, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = mx.transpose(x, (0, 2, 1))
        return residual + x


class DecoderResidualUnitMLX(nn.Module):
    """Residual unit with Snake activation."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBetaMLX(dim)
        self.conv1 = CausalConv1dMLX(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBetaMLX(dim)
        self.conv2 = CausalConv1dMLX(dim, dim, kernel_size=1)

    def __call__(self, x):
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class DecoderBlockMLX(nn.Module):
    """Decoder block with upsampling and residual units."""

    def __init__(self, in_dim: int, out_dim: int, upsample_rate: int):
        super().__init__()
        self.act = SnakeBetaMLX(in_dim)
        self.transconv = CausalTransConv1dMLX(
            in_dim, out_dim, 2 * upsample_rate, upsample_rate
        )
        # 3 residual units with dilations 1, 3, 9
        self.residual_units = [
            DecoderResidualUnitMLX(out_dim, dilation=1),
            DecoderResidualUnitMLX(out_dim, dilation=3),
            DecoderResidualUnitMLX(out_dim, dilation=9),
        ]

    def __call__(self, x):
        x = self.act(x)
        x = self.transconv(x)
        for unit in self.residual_units:
            x = unit(x)
        return x


class Qwen3TTSDecoderMLX(nn.Module):
    """
    MLX implementation of Qwen3-TTS Audio Decoder.

    Structure:
    - pre_conv: latent_dim (1024) -> decoder_dim (1536)
    - upsample: 2 blocks with upsampling_ratios [2, 2]
    - decoder[0]: CausalConv (latent_dim -> decoder_dim)
    - decoder[1-4]: DecoderBlock with upsample_rates [8, 5, 4, 3]
    - decoder[5]: SnakeBeta activation
    - decoder[6]: CausalConv (output_dim -> 1)
    """

    def __init__(self):
        super().__init__()

        # Config from Qwen3-TTS-0.6B
        self.decoder_dim = 1536
        self.latent_dim = 1024
        self.upsample_rates = [8, 5, 4, 3]
        self.upsampling_ratios = [2, 2]
        self.codebook_dim = 512

        # Total upsample factor
        self.total_upsample = int(np.prod(self.upsample_rates + self.upsampling_ratios))

        # Pre-conv: codebook_dim -> latent_dim
        self.pre_conv = CausalConv1dMLX(self.codebook_dim, self.latent_dim, kernel_size=3)

        # Upsample blocks (before main decoder)
        self.upsample_blocks = []
        for factor in self.upsampling_ratios:
            self.upsample_blocks.append({
                'transconv': CausalTransConv1dMLX(self.latent_dim, self.latent_dim, factor, factor),
                'convnext': ConvNeXtBlockMLX(self.latent_dim),
            })

        # Main decoder
        # decoder[0]: CausalConv latent_dim -> decoder_dim
        self.decoder_conv0 = CausalConv1dMLX(self.latent_dim, self.decoder_dim, kernel_size=7)

        # decoder[1-4]: DecoderBlocks
        self.decoder_blocks = []
        in_dim = self.decoder_dim
        for i, rate in enumerate(self.upsample_rates):
            out_dim = self.decoder_dim // (2 ** (i + 1))
            self.decoder_blocks.append(DecoderBlockMLX(in_dim, out_dim, rate))
            in_dim = out_dim

        # decoder[5]: Final activation
        self.final_act = SnakeBetaMLX(in_dim)

        # decoder[6]: Output conv (in_dim -> 1)
        self.final_conv = CausalConv1dMLX(in_dim, 1, kernel_size=7)

    def __call__(self, hidden):
        """
        Forward pass.

        Args:
            hidden: (batch, codebook_dim, length) - output from quantizer

        Returns:
            wav: (batch, 1, length * total_upsample)
        """
        # Pre-conv
        hidden = self.pre_conv(hidden)

        # NOTE: pre_transformer would go here, but we skip it for now
        # hidden = self.pre_transformer(hidden)

        # Upsample blocks
        for block in self.upsample_blocks:
            hidden = block['transconv'](hidden)
            hidden = block['convnext'](hidden)

        # Main decoder
        wav = self.decoder_conv0(hidden)

        for block in self.decoder_blocks:
            wav = block(wav)

        wav = self.final_act(wav)
        wav = self.final_conv(wav)

        return mx.clip(wav, -1, 1)

    def load_weights(self, weights: Dict[str, mx.array]):
        """Load weights from converted PyTorch model."""

        # pre_conv
        self.pre_conv.weight = weights["pre_conv.weight"]
        self.pre_conv.bias = weights["pre_conv.bias"]

        # upsample blocks
        for i, block in enumerate(self.upsample_blocks):
            prefix = f"upsample.{i}"
            block['transconv'].weight = weights[f"{prefix}.transconv.weight"]
            block['transconv'].bias = weights[f"{prefix}.transconv.bias"]
            block['convnext'].dwconv.weight = weights[f"{prefix}.convnext.dwconv.weight"]
            block['convnext'].dwconv.bias = weights[f"{prefix}.convnext.dwconv.bias"]
            block['convnext'].norm.weight = weights[f"{prefix}.convnext.norm.weight"]
            block['convnext'].norm.bias = weights[f"{prefix}.convnext.norm.bias"]
            block['convnext'].pwconv1.weight = weights[f"{prefix}.convnext.pwconv1.weight"]
            block['convnext'].pwconv1.bias = weights[f"{prefix}.convnext.pwconv1.bias"]
            block['convnext'].pwconv2.weight = weights[f"{prefix}.convnext.pwconv2.weight"]
            block['convnext'].pwconv2.bias = weights[f"{prefix}.convnext.pwconv2.bias"]
            block['convnext'].gamma = weights[f"{prefix}.convnext.gamma"]

        # decoder[0] - first conv
        self.decoder_conv0.weight = weights["decoder.0.weight"]
        self.decoder_conv0.bias = weights["decoder.0.bias"]

        # decoder[1-4] - decoder blocks
        for i, block in enumerate(self.decoder_blocks):
            prefix = f"decoder.{i+1}"

            # Activation beta
            if f"{prefix}.act.beta" in weights:
                block.act.beta = weights[f"{prefix}.act.beta"]

            # TransConv
            block.transconv.weight = weights[f"{prefix}.transconv.weight"]
            block.transconv.bias = weights[f"{prefix}.transconv.bias"]

            # Residual units
            for j, unit in enumerate(block.residual_units):
                unit_prefix = f"{prefix}.residual.{j}"
                if f"{unit_prefix}.act1.beta" in weights:
                    unit.act1.beta = weights[f"{unit_prefix}.act1.beta"]
                    unit.conv1.weight = weights[f"{unit_prefix}.conv1.weight"]
                    unit.conv1.bias = weights[f"{unit_prefix}.conv1.bias"]
                    unit.act2.beta = weights[f"{unit_prefix}.act2.beta"]
                    unit.conv2.weight = weights[f"{unit_prefix}.conv2.weight"]
                    unit.conv2.bias = weights[f"{unit_prefix}.conv2.bias"]

        # decoder[5] - final activation
        if "decoder.5.beta" in weights:
            self.final_act.beta = weights["decoder.5.beta"]

        # decoder[6] - final conv
        self.final_conv.weight = weights["decoder.6.weight"]
        self.final_conv.bias = weights["decoder.6.bias"]

        print("Weights loaded successfully!")


if __name__ == "__main__":
    print("=== MLX Decoder v2 Test ===\n")

    # Create decoder
    decoder = Qwen3TTSDecoderMLX()
    print(f"Total upsample factor: {decoder.total_upsample}")

    # Test with random input
    batch_size = 1
    seq_len = 50
    hidden = mx.random.normal((batch_size, decoder.codebook_dim, seq_len))

    print(f"Input shape: {hidden.shape}")

    import time
    start = time.time()
    output = decoder(hidden)
    mx.eval(output)
    elapsed = time.time() - start

    print(f"Output shape: {output.shape}")
    print(f"Expected length: {seq_len * decoder.total_upsample}")
    print(f"Time: {elapsed:.3f}s")

    # Test weight loading
    print("\n--- Testing weight loading ---")
    try:
        weights = dict(mx.load("decoder_weights_mlx.npz"))
        print(f"Loaded {len(weights)} weight tensors")
        decoder.load_weights(weights)

        # Test again with loaded weights
        start = time.time()
        output = decoder(hidden)
        mx.eval(output)
        elapsed = time.time() - start
        print(f"With real weights - Time: {elapsed:.3f}s")

    except FileNotFoundError:
        print("Weight file not found. Run weight_converter.py first.")
