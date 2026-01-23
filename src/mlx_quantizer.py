"""
MLX Quantizer for Qwen3-TTS

SplitResidualVectorQuantizer の decode 部分を MLX で実装。
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Optional


class EuclideanCodebookMLX:
    """EuclideanCodebook - decode only."""

    def __init__(self, codebook_size: int = 2048, dim: int = 256, epsilon: float = 1e-5):
        self.codebook_size = codebook_size
        self.dim = dim
        self.epsilon = epsilon

        # These will be loaded from weights
        self.embedding_sum = mx.zeros((codebook_size, dim))
        self.cluster_usage = mx.ones((codebook_size,))

    @property
    def embedding(self) -> mx.array:
        """Compute actual embedding from EMA stats."""
        return self.embedding_sum / mx.maximum(self.cluster_usage[:, None], self.epsilon)

    def decode(self, codes: mx.array) -> mx.array:
        """
        Decode codes to embeddings.

        Args:
            codes: (batch, length) - indices into codebook

        Returns:
            (batch, length, dim) - embeddings
        """
        emb = self.embedding  # (codebook_size, dim)
        # Equivalent to F.embedding
        return emb[codes]  # (batch, length, dim)


class VectorQuantizationMLX:
    """Single VQ layer - decode only."""

    def __init__(self, codebook_size: int = 2048, dim: int = 256):
        self.codebook = EuclideanCodebookMLX(codebook_size, dim)
        # project_out is Identity in this model

    def decode(self, codes: mx.array) -> mx.array:
        """
        Decode codes.

        Args:
            codes: (batch, length)

        Returns:
            (batch, dim, length)
        """
        quantized = self.codebook.decode(codes)  # (batch, length, dim)
        # project_out is Identity, skip
        quantized = mx.transpose(quantized, (0, 2, 1))  # (batch, dim, length)
        return quantized


class ResidualVectorQuantizationMLX:
    """RVQ - multiple VQ layers with residual."""

    def __init__(self, num_layers: int, codebook_size: int = 2048, dim: int = 256):
        self.layers = [VectorQuantizationMLX(codebook_size, dim) for _ in range(num_layers)]

    def decode(self, codes: mx.array) -> mx.array:
        """
        Decode codes with residual addition.

        Args:
            codes: (num_layers, batch, length)

        Returns:
            (batch, dim, length)
        """
        quantized = None
        for idx, layer in enumerate(self.layers):
            layer_codes = codes[idx]  # (batch, length)
            layer_quantized = layer.decode(layer_codes)  # (batch, dim, length)
            if quantized is None:
                quantized = layer_quantized
            else:
                quantized = quantized + layer_quantized
        return quantized


class Conv1x1MLX(nn.Module):
    """1x1 Convolution (basically a projection)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # MLX conv1d weight: (out_channels, kernel_size, in_channels)
        self.weight = mx.zeros((out_channels, 1, in_channels))
        # No bias in the original model

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, in_channels, length)

        Returns:
            (batch, out_channels, length)
        """
        # MLX conv1d expects (batch, length, channels)
        x = mx.transpose(x, (0, 2, 1))  # (batch, length, in_channels)
        x = mx.conv1d(x, self.weight, stride=1, padding=0)
        x = mx.transpose(x, (0, 2, 1))  # (batch, out_channels, length)
        return x


class ResidualVectorQuantizerMLX:
    """ResidualVectorQuantizer with input/output projections."""

    def __init__(self, num_layers: int, codebook_size: int = 2048,
                 input_dim: int = 512, codebook_dim: int = 256):
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
        # Transpose to (num_layers, batch, length)
        codes = mx.transpose(codes, (1, 0, 2))
        quantized = self.vq.decode(codes)  # (batch, codebook_dim, length)
        quantized = self.output_proj(quantized)  # (batch, input_dim, length)
        return quantized


class SplitResidualVectorQuantizerMLX:
    """
    SplitResidualVectorQuantizer - splits codes into semantic and acoustic parts.

    - n_q_semantic = 1 (first quantizer is semantic)
    - remaining 15 quantizers are acoustic
    """

    def __init__(self, n_q_semantic: int = 1, total_quantizers: int = 16,
                 codebook_size: int = 2048, input_dim: int = 512, codebook_dim: int = 256):
        self.n_q_semantic = n_q_semantic
        self.total_quantizers = total_quantizers

        # rvq_first: 1 layer for semantic
        self.rvq_first = ResidualVectorQuantizerMLX(
            num_layers=n_q_semantic,
            codebook_size=codebook_size,
            input_dim=input_dim,
            codebook_dim=codebook_dim
        )

        # rvq_rest: 15 layers for acoustic
        self.rvq_rest = ResidualVectorQuantizerMLX(
            num_layers=total_quantizers - n_q_semantic,
            codebook_size=codebook_size,
            input_dim=input_dim,
            codebook_dim=codebook_dim
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
        quantized = quantized + self.rvq_rest.decode(codes_acoustic)  # (batch, 512, length)

        return quantized

    def load_weights(self, weights: Dict[str, mx.array]):
        """Load weights from converted PyTorch model."""

        # rvq_first
        self.rvq_first.input_proj.weight = weights["rvq_first.input_proj.weight"]
        self.rvq_first.output_proj.weight = weights["rvq_first.output_proj.weight"]

        for i, layer in enumerate(self.rvq_first.vq.layers):
            prefix = f"rvq_first.vq.layers.{i}"
            layer.codebook.embedding_sum = weights[f"{prefix}.embedding_sum"]
            layer.codebook.cluster_usage = weights[f"{prefix}.cluster_usage"]

        # rvq_rest
        self.rvq_rest.input_proj.weight = weights["rvq_rest.input_proj.weight"]
        self.rvq_rest.output_proj.weight = weights["rvq_rest.output_proj.weight"]

        for i, layer in enumerate(self.rvq_rest.vq.layers):
            prefix = f"rvq_rest.vq.layers.{i}"
            layer.codebook.embedding_sum = weights[f"{prefix}.embedding_sum"]
            layer.codebook.cluster_usage = weights[f"{prefix}.cluster_usage"]

        print("Quantizer weights loaded successfully!")


if __name__ == "__main__":
    print("=== MLX Quantizer Test ===\n")

    # Create quantizer
    quantizer = SplitResidualVectorQuantizerMLX()

    # Test with random codes
    batch_size = 1
    seq_len = 50
    codes = mx.random.randint(0, 2048, (batch_size, 16, seq_len))

    print(f"Input codes shape: {codes.shape}")

    import time
    start = time.time()
    hidden = quantizer.decode(codes)
    mx.eval(hidden)
    elapsed = time.time() - start

    print(f"Output hidden shape: {hidden.shape}")
    print(f"Expected: (1, 512, {seq_len})")
    print(f"Time: {elapsed:.3f}s")

    # Test weight loading
    print("\n--- Testing weight loading ---")
    try:
        weights = dict(mx.load("quantizer_weights_mlx.npz"))
        print(f"Loaded {len(weights)} weight tensors")
        quantizer.load_weights(weights)

        # Test again with loaded weights
        start = time.time()
        hidden = quantizer.decode(codes)
        mx.eval(hidden)
        elapsed = time.time() - start
        print(f"With real weights - Time: {elapsed:.3f}s")

    except FileNotFoundError:
        print("Weight file not found. Run weight converter first.")
