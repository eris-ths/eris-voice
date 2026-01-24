"""
Tests for MLX Audio Decoder v2
"""

import mlx.core as mx
import time
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_decoder_v2 import Qwen3TTSDecoderMLX


def test_decoder_forward():
    """Test decoder forward pass with random input."""
    print("=== MLX Decoder v2 Test ===\n")

    decoder = Qwen3TTSDecoderMLX()
    print(f"Total upsample factor: {decoder.total_upsample}")

    batch_size = 1
    seq_len = 50
    hidden = mx.random.normal((batch_size, decoder.codebook_dim, seq_len))

    print(f"Input shape: {hidden.shape}")

    start = time.time()
    output = decoder(hidden)
    mx.eval(output)
    elapsed = time.time() - start

    print(f"Output shape: {output.shape}")
    print(f"Expected length: {seq_len * decoder.total_upsample}")
    print(f"Time: {elapsed:.3f}s")

    # Verify output shape
    expected_length = seq_len * decoder.total_upsample
    assert output.shape == (batch_size, 1, expected_length), \
        f"Shape mismatch: {output.shape} != (1, 1, {expected_length})"

    print("\n✅ Forward pass test passed!")


def test_decoder_weight_loading():
    """Test decoder weight loading from .npz file."""
    print("\n--- Testing weight loading ---")

    decoder = Qwen3TTSDecoderMLX()
    weights_path = Path(__file__).parent.parent / "decoder_weights_mlx.npz"

    if not weights_path.exists():
        print(f"⚠️ Weight file not found: {weights_path}")
        print("   Run weight_converter.py first.")
        return

    weights = dict(mx.load(str(weights_path)))
    print(f"Loaded {len(weights)} weight tensors")

    decoder.load_weights(weights)

    # Test forward with loaded weights
    hidden = mx.random.normal((1, decoder.codebook_dim, 50))

    start = time.time()
    output = decoder(hidden)
    mx.eval(output)
    elapsed = time.time() - start

    print(f"With real weights - Time: {elapsed:.3f}s")
    print("\n✅ Weight loading test passed!")


if __name__ == "__main__":
    test_decoder_forward()
    test_decoder_weight_loading()
