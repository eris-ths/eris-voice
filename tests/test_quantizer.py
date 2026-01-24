"""
Tests for MLX Quantizer
"""

import mlx.core as mx
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_quantizer import SplitResidualVectorQuantizerMLX


def test_quantizer_decode():
    """Test quantizer decode with random codes."""
    print("=== MLX Quantizer Test ===\n")

    quantizer = SplitResidualVectorQuantizerMLX()

    batch_size = 1
    seq_len = 50
    codes = mx.random.randint(0, 2048, (batch_size, 16, seq_len))

    print(f"Input codes shape: {codes.shape}")

    start = time.time()
    hidden = quantizer.decode(codes)
    mx.eval(hidden)
    elapsed = time.time() - start

    print(f"Output hidden shape: {hidden.shape}")
    print(f"Expected: (1, 512, {seq_len})")
    print(f"Time: {elapsed:.3f}s")

    assert hidden.shape == (batch_size, 512, seq_len), \
        f"Shape mismatch: {hidden.shape} != (1, 512, {seq_len})"

    print("\n✅ Quantizer decode test passed!")


def test_quantizer_weight_loading():
    """Test quantizer weight loading from .npz file."""
    print("\n--- Testing weight loading ---")

    quantizer = SplitResidualVectorQuantizerMLX()
    weights_path = Path(__file__).parent.parent / "quantizer_weights_mlx.npz"

    if not weights_path.exists():
        print(f"⚠️ Weight file not found: {weights_path}")
        print("   Run weight_converter.py first.")
        return

    weights = dict(mx.load(str(weights_path)))
    print(f"Loaded {len(weights)} weight tensors")

    quantizer.load_weights(weights)

    # Test decode with loaded weights
    codes = mx.random.randint(0, 2048, (1, 16, 50))

    start = time.time()
    hidden = quantizer.decode(codes)
    mx.eval(hidden)
    elapsed = time.time() - start

    print(f"With real weights - Time: {elapsed:.3f}s")
    print("\n✅ Weight loading test passed!")


if __name__ == "__main__":
    test_quantizer_decode()
    test_quantizer_weight_loading()
