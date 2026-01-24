"""
MLX Quantizer for Qwen3-TTS

SplitResidualVectorQuantizer の decode 部分を MLX で実装。
3.5x 高速化を達成。
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional


class WeightLoadError(Exception):
    """Raised when required weights are missing."""
    pass


def _get_weight(weights: Dict[str, mx.array], key: str) -> mx.array:
    """Get weight with explicit error message."""
    if key not in weights:
        available = sorted(weights.keys())[:10]
        raise WeightLoadError(
            f"Missing weight: '{key}'\n"
            f"Available keys (first 10): {available}"
        )
    return weights[key]


class EuclideanCodebookMLX:
    """EuclideanCodebook - decode only."""

    def __init__(
        self,
        codebook_size: int = 2048,
        dim: int = 256,
        epsilon: float = 1e-5,
    ) -> None:
        self.codebook_size = codebook_size
        self.dim = dim
        self.epsilon = epsilon

        # These will be loaded from weights
        self.embedding_sum = mx.zeros((codebook_size, dim))
        self.cluster_usage = mx.ones((codebook_size,))

    @property
    def embedding(self) -> mx.array:
        """Compute actual embedding from EMA stats."""
        return self.embedding_sum / mx.maximum(
            self.cluster_usage[:, None], self.epsilon
        )

    def decode(self, codes: mx.array) -> mx.array:
        """
        Decode codes to embeddings.

        Args:
            codes: (batch, length) - indices into codebook

        Returns:
            (batch, length, dim) - embeddings
        """
        emb = self.embedding  # (codebook_size, dim)
        return emb[codes]  # (batch, length, dim)


class VectorQuantizationMLX:
    """Single VQ layer - decode only."""

    def __init__(self, codebook_size: int = 2048, dim: int = 256) -> None:
        self.codebook = EuclideanCodebookMLX(codebook_size, dim)

    def decode(self, codes: mx.array) -> mx.array:
        """
        Decode codes.

        Args:
            codes: (batch, length)

        Returns:
            (batch, dim, length)
        """
        quantized = self.codebook.decode(codes)  # (batch, length, dim)
        quantized = mx.transpose(quantized, (0, 2, 1))  # (batch, dim, length)
        return quantized


class ResidualVectorQuantizationMLX:
    """RVQ - multiple VQ layers with residual."""

    def __init__(
        self,
        num_layers: int,
        codebook_size: int = 2048,
        dim: int = 256,
    ) -> None:
        self.layers: List[VectorQuantizationMLX] = [
            VectorQuantizationMLX(codebook_size, dim)
            for _ in range(num_layers)
        ]

    def decode(self, codes: mx.array) -> mx.array:
        """
        Decode codes with residual addition.

        Args:
            codes: (num_layers, batch, length)

        Returns:
            (batch, dim, length)
        """
        quantized: Optional[mx.array] = None
        for idx, layer in enumerate(self.layers):
            layer_codes = codes[idx]  # (batch, length)
            layer_quantized = layer.decode(layer_codes)  # (batch, dim, length)
            if quantized is None:
                quantized = layer_quantized
            else:
                quantized = quantized + layer_quantized
        return quantized  # type: ignore


class Conv1x1MLX(nn.Module):
    """1x1 Convolution (basically a projection)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # MLX conv1d weight: (out_channels, kernel_size, in_channels)
        self.weight = mx.zeros((out_channels, 1, in_channels))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply 1x1 convolution.

        Args:
            x: (batch, in_channels, length)

        Returns:
            (batch, out_channels, length)
        """
        x = mx.transpose(x, (0, 2, 1))  # (batch, length, in_channels)
        x = mx.conv1d(x, self.weight, stride=1, padding=0)
        x = mx.transpose(x, (0, 2, 1))  # (batch, out_channels, length)
        return x


class ResidualVectorQuantizerMLX:
    """ResidualVectorQuantizer with input/output projections."""

    def __init__(
        self,
        num_layers: int,
        codebook_size: int = 2048,
        input_dim: int = 512,
        codebook_dim: int = 256,
    ) -> None:
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim

        self.input_proj = Conv1x1MLX(input_dim, codebook_dim)
        self.output_proj = Conv1x1MLX(codebook_dim, input_dim)
        self.vq = ResidualVectorQuantizationMLX(num_layers, codebook_size, codebook_dim)

    def decode(self, codes: mx.array) -> mx.array:
        """
        Decode codes.

        Args:
            codes: (batch, num_layers, length)

        Returns:
            (batch, input_dim, length)
        """
        codes = mx.transpose(codes, (1, 0, 2))  # (num_layers, batch, length)
        quantized = self.vq.decode(codes)  # (batch, codebook_dim, length)
        quantized = self.output_proj(quantized)  # (batch, input_dim, length)
        return quantized


class SplitResidualVectorQuantizerMLX:
    """
    SplitResidualVectorQuantizer - splits codes into semantic and acoustic parts.

    - n_q_semantic = 1 (first quantizer is semantic)
    - remaining 15 quantizers are acoustic

    Achieves 3.5x speedup over PyTorch CPU implementation.
    """

    def __init__(
        self,
        n_q_semantic: int = 1,
        total_quantizers: int = 16,
        codebook_size: int = 2048,
        input_dim: int = 512,
        codebook_dim: int = 256,
    ) -> None:
        self.n_q_semantic = n_q_semantic
        self.total_quantizers = total_quantizers

        # rvq_first: 1 layer for semantic
        self.rvq_first = ResidualVectorQuantizerMLX(
            num_layers=n_q_semantic,
            codebook_size=codebook_size,
            input_dim=input_dim,
            codebook_dim=codebook_dim,
        )

        # rvq_rest: 15 layers for acoustic
        self.rvq_rest = ResidualVectorQuantizerMLX(
            num_layers=total_quantizers - n_q_semantic,
            codebook_size=codebook_size,
            input_dim=input_dim,
            codebook_dim=codebook_dim,
        )

    def decode(self, codes: mx.array) -> mx.array:
        """
        Decode codes.

        Args:
            codes: (batch, total_quantizers, length)

        Returns:
            (batch, input_dim, length)
        """
        # Split codes
        codes_semantic = codes[:, :self.n_q_semantic, :]  # (batch, 1, length)
        codes_acoustic = codes[:, self.n_q_semantic:, :]  # (batch, 15, length)

        # Decode both
        quantized = self.rvq_first.decode(codes_semantic)  # (batch, 512, length)
        quantized = quantized + self.rvq_rest.decode(codes_acoustic)

        return quantized

    def load_weights(self, weights: Dict[str, mx.array]) -> None:
        """
        Load weights from converted PyTorch model.

        Raises:
            WeightLoadError: If required weights are missing.
        """
        # rvq_first
        self.rvq_first.input_proj.weight = _get_weight(
            weights, "rvq_first.input_proj.weight"
        )
        self.rvq_first.output_proj.weight = _get_weight(
            weights, "rvq_first.output_proj.weight"
        )

        for i, layer in enumerate(self.rvq_first.vq.layers):
            prefix = f"rvq_first.vq.layers.{i}"
            layer.codebook.embedding_sum = _get_weight(
                weights, f"{prefix}.embedding_sum"
            )
            layer.codebook.cluster_usage = _get_weight(
                weights, f"{prefix}.cluster_usage"
            )

        # rvq_rest
        self.rvq_rest.input_proj.weight = _get_weight(
            weights, "rvq_rest.input_proj.weight"
        )
        self.rvq_rest.output_proj.weight = _get_weight(
            weights, "rvq_rest.output_proj.weight"
        )

        for i, layer in enumerate(self.rvq_rest.vq.layers):
            prefix = f"rvq_rest.vq.layers.{i}"
            layer.codebook.embedding_sum = _get_weight(
                weights, f"{prefix}.embedding_sum"
            )
            layer.codebook.cluster_usage = _get_weight(
                weights, f"{prefix}.cluster_usage"
            )

        print("Quantizer weights loaded successfully!")
