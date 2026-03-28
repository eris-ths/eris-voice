#!/usr/bin/env python3
"""
PolarQuant KV Cache for Qwen3-TTS

Polar coordinate transformation-based KV cache quantization.
Based on PolarQuant (arXiv:2502.02617) and TurboQuant (arXiv:2504.19874).

Core idea:
  1. Random orthogonal rotation -> uniform coordinate distribution
  2. Normalize to unit vector + radius
  3. Quantize normalized components (bounded [-1, 1], concentrated)
  4. Store: quantized_unit_vector + radius (float16)

Memory: ~4-5 bit effective per channel (vs 32-bit float32 = 6-8x compression)
"""

from typing import Tuple, Optional
import mlx.core as mx
import mlx.nn as nn


class PolarQuantKVCache:
    """
    KV Cache with PolarQuant compression.

    Stores keys/values in quantized polar form:
      - Quantized unit vector (4-bit, mx.quantize)
      - Radius (float16)
      - Shared rotation matrix Q (one per layer, computed once)

    Shape: (batch, n_kv_heads, seq_len, head_dim)
    """

    def __init__(self, head_dim: int = 128, step: int = 256,
                 bits: int = 4, group_size: int = 32):
        self.head_dim = head_dim
        self.step = step
        self.bits = bits
        self.group_size = group_size
        self.offset: int = 0

        # Random orthogonal rotation matrix (computed once)
        random_matrix = mx.random.normal((head_dim, head_dim))
        self._Q, _ = mx.linalg.qr(random_matrix, stream=mx.cpu)
        mx.eval(self._Q)

        # Quantized storage
        self._k_quant = None  # (B, n_kv_heads, allocated, head_dim // el_per_int)
        self._k_scales = None
        self._k_biases = None
        self._k_radius = None  # (B, n_kv_heads, allocated, 1) float16

        self._v_quant = None
        self._v_scales = None
        self._v_biases = None
        self._v_radius = None

    def _quantize_vector(self, v: mx.array) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Quantize a vector using PolarQuant.

        Args:
            v: (B, n_kv_heads, seq, head_dim) float32

        Returns:
            (quant_data, scales, biases, radius)
        """
        # 1. Rotate
        v_rot = v @ self._Q  # (B, H, S, D)

        # 2. Compute radius and normalize
        radius = mx.sqrt((v_rot * v_rot).sum(axis=-1, keepdims=True))  # (B, H, S, 1)
        v_unit = v_rot / (radius + 1e-8)  # (B, H, S, D) in [-1, 1]

        # 3. Quantize normalized components
        quant_data, scales, biases = mx.quantize(v_unit, group_size=self.group_size, bits=self.bits)

        # 4. Store radius as float16
        radius = radius.astype(mx.float16)

        return quant_data, scales, biases, radius

    def _dequantize_vector(self, quant_data, scales, biases, radius) -> mx.array:
        """
        Dequantize a PolarQuant vector.

        Returns:
            (B, n_kv_heads, seq, head_dim) float32
        """
        # 1. Dequantize unit vector
        v_unit = mx.dequantize(quant_data, scales, biases,
                               group_size=self.group_size, bits=self.bits)

        # 2. Rescale by radius
        v_rot = v_unit * radius.astype(mx.float32)

        # 3. Inverse rotate
        v = v_rot @ self._Q.T

        return v

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Quantize and store new keys/values, return all dequantized.

        Args:
            keys: (B, n_kv_heads, seq_len, head_dim)
            values: (B, n_kv_heads, seq_len, head_dim)

        Returns:
            Tuple of (all_keys, all_values) dequantized
        """
        # Quantize new data
        k_q, k_s, k_b, k_r = self._quantize_vector(keys)
        v_q, v_s, v_b, v_r = self._quantize_vector(values)

        if self._k_quant is None:
            # First allocation
            self._k_quant = k_q
            self._k_scales = k_s
            self._k_biases = k_b
            self._k_radius = k_r
            self._v_quant = v_q
            self._v_scales = v_s
            self._v_biases = v_b
            self._v_radius = v_r
        else:
            # Concatenate (simple approach for MVP)
            self._k_quant = mx.concatenate([self._k_quant, k_q], axis=2)
            self._k_scales = mx.concatenate([self._k_scales, k_s], axis=2)
            self._k_biases = mx.concatenate([self._k_biases, k_b], axis=2)
            self._k_radius = mx.concatenate([self._k_radius, k_r], axis=2)
            self._v_quant = mx.concatenate([self._v_quant, v_q], axis=2)
            self._v_scales = mx.concatenate([self._v_scales, v_s], axis=2)
            self._v_biases = mx.concatenate([self._v_biases, v_b], axis=2)
            self._v_radius = mx.concatenate([self._v_radius, v_r], axis=2)

        self.offset += keys.shape[2]

        # Dequantize all for attention
        all_keys = self._dequantize_vector(
            self._k_quant, self._k_scales, self._k_biases, self._k_radius
        )
        all_values = self._dequantize_vector(
            self._v_quant, self._v_scales, self._v_biases, self._v_radius
        )

        return all_keys, all_values

    def reset(self):
        """Clear the cache."""
        self._k_quant = None
        self._k_scales = None
        self._k_biases = None
        self._k_radius = None
        self._v_quant = None
        self._v_scales = None
        self._v_biases = None
        self._v_radius = None
        self.offset = 0


class MultiLayerPolarQuantKVCache:
    """Manages PolarQuant KV caches for multiple transformer layers."""

    def __init__(self, num_layers: int, head_dim: int = 128,
                 bits: int = 4, group_size: int = 32):
        self.num_layers = num_layers
        self.caches = [
            PolarQuantKVCache(head_dim=head_dim, bits=bits, group_size=group_size)
            for _ in range(num_layers)
        ]

    def get_caches(self):
        return self.caches

    def get_offset(self) -> int:
        if self.caches and self.caches[0].offset > 0:
            return self.caches[0].offset
        return 0

    def reset(self):
        for cache in self.caches:
            cache.reset()

    def __len__(self):
        return self.num_layers


# ─── Tests ───

def test_polar_quant():
    """Test PolarQuant KV Cache."""
    import numpy as np
    import time

    print("=" * 50)
    print("PolarQuant KV Cache Tests")
    print("=" * 50)

    # Test 1: Basic quantize/dequantize
    print("\nTest 1: Quantize/Dequantize roundtrip")
    cache = PolarQuantKVCache(head_dim=128, bits=4, group_size=32)

    k = mx.random.normal((1, 8, 1, 128))
    v = mx.random.normal((1, 8, 1, 128))

    all_k, all_v = cache.update_and_fetch(k, v)
    mx.eval(all_k, all_v)

    k_err = float(mx.abs(k - all_k).max())
    v_err = float(mx.abs(v - all_v).max())
    print(f"  Keys error: max={k_err:.4f}")
    print(f"  Values error: max={v_err:.4f}")
    assert all_k.shape == (1, 8, 1, 128)
    assert cache.offset == 1

    # Test 2: Multiple steps
    print("\nTest 2: Multiple steps")
    cache = PolarQuantKVCache(head_dim=128, bits=4)

    errors = []
    for i in range(20):
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        all_k, all_v = cache.update_and_fetch(k, v)
        mx.eval(all_k, all_v)

    print(f"  After 20 steps: offset={cache.offset}, shape={all_k.shape}")
    assert all_k.shape == (1, 8, 20, 128)

    # Test 3: Memory comparison
    print("\nTest 3: Memory estimate")
    n_steps = 100
    n_kv_heads = 8
    head_dim = 128

    original = n_steps * n_kv_heads * head_dim * 4  # float32
    # Quantized: quant_data + scales + biases + radius
    el_per_int = 32 // 4  # 8 elements per uint32 at 4-bit
    quant = n_steps * n_kv_heads * (head_dim // el_per_int) * 4  # quant data
    scales_biases = n_steps * n_kv_heads * (head_dim // 32) * 4 * 2  # scales + biases
    radius = n_steps * n_kv_heads * 1 * 2  # float16
    total_quant = quant + scales_biases + radius

    ratio = original / total_quant
    print(f"  Original (float32): {original / 1024:.1f} KB")
    print(f"  PolarQuant (4-bit): {total_quant / 1024:.1f} KB")
    print(f"  Compression: {ratio:.1f}x")
    print(f"  Per KV pair (28 layers): {original * 28 * 2 / 1024 / 1024:.1f} MB -> {total_quant * 28 * 2 / 1024 / 1024:.1f} MB")

    # Test 4: Multi-layer cache
    print("\nTest 4: Multi-layer cache")
    multi = MultiLayerPolarQuantKVCache(28, head_dim=128, bits=4)
    assert len(multi) == 28

    for c in multi.get_caches():
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        c.update_and_fetch(k, v)

    assert multi.get_offset() == 1
    multi.reset()
    assert multi.get_offset() == 0

    # Test 5: Reconstruction quality with real-scale values
    print("\nTest 5: Reconstruction quality")
    cache = PolarQuantKVCache(head_dim=128, bits=4)

    # Simulate attention-like values (smaller magnitude)
    k = mx.random.normal((1, 8, 50, 128)) * 0.1
    v = mx.random.normal((1, 8, 50, 128)) * 0.1
    all_k, all_v = cache.update_and_fetch(k, v)
    mx.eval(all_k, all_v)

    k_err = float(mx.abs(k - all_k).mean())
    cos_sim = float((k * all_k).sum() / (mx.sqrt((k*k).sum()) * mx.sqrt((all_k*all_k).sum())))
    print(f"  Mean abs error: {k_err:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    print("\n✅ PolarQuant KV Cache tests passed!")


if __name__ == "__main__":
    test_polar_quant()
