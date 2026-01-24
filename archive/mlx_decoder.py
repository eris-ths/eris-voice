"""
MLX Audio Decoder for Qwen3-TTS

PyTorch の Audio Decoder を MLX で書き直し、Apple Silicon で高速化。
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import List, Optional


class SnakeBetaMLX(nn.Module):
    """Snake activation with learnable beta parameter."""

    def __init__(self, channels: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = mx.ones((1, channels, 1))

    def __call__(self, x):
        # x: (batch, channels, length)
        return x + (1 / (self.beta + 1e-9)) * mx.power(mx.sin(self.alpha * x), 2)


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

        # MLX Conv1d weight: (out_channels, kernel_size, in_channels // groups)
        scale = math.sqrt(2.0 / (kernel_size * in_channels // groups))
        self.weight = mx.random.normal(
            (out_channels, kernel_size, in_channels // groups)
        ) * scale
        self.bias = mx.zeros((out_channels,))

        self.effective_kernel = (kernel_size - 1) * dilation + 1
        self.padding = self.effective_kernel - stride

    def __call__(self, x):
        # x: (batch, channels, length)
        # Pad for causal convolution
        x = mx.pad(x, [(0, 0), (0, 0), (self.padding, 0)])

        # MLX conv1d expects (batch, length, channels)
        x = mx.transpose(x, (0, 2, 1))

        # Apply convolution
        x = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Add bias
        x = x + self.bias

        # Back to (batch, channels, length)
        x = mx.transpose(x, (0, 2, 1))
        return x


class CausalTransConv1dMLX(nn.Module):
    """Causal Transposed Conv1d (upsampling) for MLX."""

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

        # Weight for transposed conv: (C_out, K, C_in)
        scale = math.sqrt(2.0 / (kernel_size * in_channels))
        self.weight = mx.random.normal(
            (out_channels, kernel_size, in_channels)
        ) * scale
        self.bias = mx.zeros((out_channels,))

        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = pad - self.left_pad

    def __call__(self, x):
        # x: (batch, channels, length)
        x = mx.transpose(x, (0, 2, 1))  # (batch, length, in_channels)

        # Transposed convolution (upsampling)
        # input: (N, L, C_in), weight: (C_out, K, C_in)
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

        x = mx.transpose(x, (0, 2, 1))  # (batch, out_channels, length)
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
        x = mx.transpose(x, (0, 2, 1))  # (batch, length, channels)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = mx.transpose(x, (0, 2, 1))  # (batch, channels, length)

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
    """Decoder block with upsampling."""

    def __init__(self, in_dim: int, out_dim: int, upsample_rate: int):
        super().__init__()
        self.act = SnakeBetaMLX(in_dim)
        self.transconv = CausalTransConv1dMLX(
            in_dim, out_dim, 2 * upsample_rate, upsample_rate
        )
        self.residual_units = [
            DecoderResidualUnitMLX(out_dim, dilation=d) for d in [1, 3, 9]
        ]

    def __call__(self, x):
        x = self.act(x)
        x = self.transconv(x)
        for unit in self.residual_units:
            x = unit(x)
        return x


class Qwen3TTSDecoderMLX(nn.Module):
    """Full decoder in MLX."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        decoder_dim = config["decoder_dim"]  # 1536
        latent_dim = config["latent_dim"]  # 1024
        upsample_rates = config["upsample_rates"]  # [8, 5, 4, 3]
        upsampling_ratios = config["upsampling_ratios"]  # [2, 2]

        # Pre-decoder upsampling
        self.upsample_blocks = []
        for factor in upsampling_ratios:
            self.upsample_blocks.append(
                (
                    CausalTransConv1dMLX(latent_dim, latent_dim, factor, factor),
                    ConvNeXtBlockMLX(latent_dim),
                )
            )

        # Main decoder
        self.pre_conv = CausalConv1dMLX(latent_dim, decoder_dim, 7)

        self.decoder_blocks = []
        in_dim = decoder_dim
        for i, rate in enumerate(upsample_rates):
            out_dim = decoder_dim // (2 ** (i + 1))
            self.decoder_blocks.append(DecoderBlockMLX(in_dim, out_dim, rate))
            in_dim = out_dim

        self.final_act = SnakeBetaMLX(in_dim)
        self.final_conv = CausalConv1dMLX(in_dim, 1, 7)

        # Total upsample factor
        self.total_upsample = np.prod(upsample_rates + upsampling_ratios)

    def __call__(self, hidden):
        # hidden: (batch, channels, length) - output from transformer

        # Upsample
        for transconv, convnext in self.upsample_blocks:
            hidden = transconv(hidden)
            hidden = convnext(hidden)

        # Decode
        wav = self.pre_conv(hidden)
        for block in self.decoder_blocks:
            wav = block(wav)
        wav = self.final_act(wav)
        wav = self.final_conv(wav)

        return mx.clip(wav, -1, 1)


def load_weights_from_pytorch(mlx_decoder: Qwen3TTSDecoderMLX, pytorch_decoder) -> None:
    """Load weights from PyTorch decoder to MLX decoder."""
    # This function will map PyTorch state_dict to MLX parameters
    # Implementation depends on exact parameter naming

    print("Weight loading not yet implemented - using random weights for testing")


if __name__ == "__main__":
    print("=== MLX Decoder Test ===")

    # Test config matching Qwen3-TTS-0.6B
    config = {
        "decoder_dim": 1536,
        "latent_dim": 1024,
        "upsample_rates": [8, 5, 4, 3],
        "upsampling_ratios": [2, 2],
        "codebook_dim": 512,
    }

    decoder = Qwen3TTSDecoderMLX(config)
    print(f"Decoder created with total_upsample: {decoder.total_upsample}")

    # Test forward pass
    batch_size = 1
    seq_len = 100
    hidden = mx.random.normal((batch_size, config["latent_dim"], seq_len))

    print(f"Input shape: {hidden.shape}")

    import time
    start = time.time()
    output = decoder(hidden)
    mx.eval(output)
    elapsed = time.time() - start

    print(f"Output shape: {output.shape}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Expected output length: {seq_len * decoder.total_upsample}")
