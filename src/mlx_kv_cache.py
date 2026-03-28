#!/usr/bin/env python3
"""
MLX KV Cache for Qwen3-TTS (Pre-allocated Buffer)

Pre-allocated key-value cache for efficient autoregressive generation.
Based on MLX-LM's KVCache (v0.26.3) — O(1) per-step updates via slice assignment.

Previous implementation used mx.concatenate per step = O(n) copy = O(n^2) total.
This implementation pre-allocates in 256-token chunks and uses slice assignment.
"""

from typing import List, Tuple, Optional
import mlx.core as mx


class KVCache:
    """
    Pre-allocated Key-Value cache for a single attention layer.

    Allocates buffers in chunks of `step` tokens. Updates via slice assignment
    instead of concatenation, giving O(1) per-step cost.

    Shape: (batch, n_kv_heads, seq_len, head_dim)
    """

    def __init__(self, step: int = 256):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0
        self.step = step

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values and return full accumulated tensors.

        On first call or buffer overflow: allocates/extends in chunks of `step`.
        Otherwise: O(1) slice assignment.

        Args:
            keys: New keys (batch, n_kv_heads, seq_len, head_dim)
            values: New values (batch, n_kv_heads, seq_len, head_dim)

        Returns:
            Tuple of (all_keys, all_values) up to current offset
        """
        prev = self.offset

        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            # Need to allocate or extend buffer
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            if self.keys is not None:
                # Extend existing buffer
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                # First allocation
                self.keys, self.values = new_k, new_v

        # O(1) slice assignment
        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self):
        """Clear the cache for a new generation."""
        self.keys = None
        self.values = None
        self.offset = 0


class MultiLayerKVCache:
    """
    Manages KV caches for multiple transformer layers.

    Usage:
        cache = MultiLayerKVCache(num_layers=28)
        for step in range(max_steps):
            output = model(input, cache=cache.get_caches())
    """

    def __init__(self, num_layers: int, step: int = 256):
        self.num_layers = num_layers
        self.caches = [KVCache(step=step) for _ in range(num_layers)]

    def get_caches(self) -> List[KVCache]:
        """Get list of caches for all layers."""
        return self.caches

    def get_offset(self) -> int:
        """Get current cache offset (same for all layers)."""
        if self.caches and self.caches[0].keys is not None:
            return self.caches[0].offset
        return 0

    def reset(self):
        """Clear all caches for a new generation."""
        for cache in self.caches:
            cache.reset()

    def __len__(self) -> int:
        return self.num_layers


# ============================================================
# Tests
# ============================================================

def test_kv_cache():
    """Test KVCache with pre-allocated buffers."""
    print("=" * 50)
    print("KV Cache Tests (Pre-allocated)")
    print("=" * 50)
    print()

    # Test 1: Basic cache update
    print("Test 1: Basic cache update")
    cache = KVCache(step=256)

    # First token
    k1 = mx.random.normal((1, 8, 1, 128))
    v1 = mx.random.normal((1, 8, 1, 128))

    all_k, all_v = cache.update_and_fetch(k1, v1)
    mx.eval(all_k, all_v)

    print(f"  After 1st token: keys={all_k.shape}, offset={cache.offset}")
    assert all_k.shape == (1, 8, 1, 128), f"Expected (1,8,1,128), got {all_k.shape}"
    assert cache.offset == 1
    # Verify buffer was pre-allocated
    assert cache.keys.shape[2] == 256, f"Expected pre-allocated 256, got {cache.keys.shape[2]}"

    # Second token
    k2 = mx.random.normal((1, 8, 1, 128))
    v2 = mx.random.normal((1, 8, 1, 128))

    all_k, all_v = cache.update_and_fetch(k2, v2)
    mx.eval(all_k, all_v)

    print(f"  After 2nd token: keys={all_k.shape}, offset={cache.offset}")
    assert all_k.shape == (1, 8, 2, 128)
    assert cache.offset == 2
    # Buffer should NOT have grown
    assert cache.keys.shape[2] == 256

    # Third token
    k3 = mx.random.normal((1, 8, 1, 128))
    v3 = mx.random.normal((1, 8, 1, 128))

    all_k, all_v = cache.update_and_fetch(k3, v3)
    mx.eval(all_k, all_v)

    print(f"  After 3rd token: keys={all_k.shape}, offset={cache.offset}")
    assert all_k.shape == (1, 8, 3, 128)
    assert cache.offset == 3

    # Test 2: Reset
    print("\nTest 2: Reset cache")
    cache.reset()
    print(f"  After reset: offset={cache.offset}, keys={cache.keys}")
    assert cache.offset == 0
    assert cache.keys is None

    # Test 3: Prefill (multi-token)
    print("\nTest 3: Prefill with multiple tokens")
    cache = KVCache(step=256)
    k_prefill = mx.random.normal((1, 8, 20, 128))
    v_prefill = mx.random.normal((1, 8, 20, 128))

    all_k, all_v = cache.update_and_fetch(k_prefill, v_prefill)
    mx.eval(all_k, all_v)

    print(f"  After prefill: keys={all_k.shape}, offset={cache.offset}")
    assert all_k.shape == (1, 8, 20, 128)
    assert cache.offset == 20
    assert cache.keys.shape[2] == 256  # Pre-allocated

    # Test 4: Buffer extension
    print("\nTest 4: Buffer extension past 256")
    cache = KVCache(step=4)  # Small step for testing
    for i in range(6):
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        all_k, all_v = cache.update_and_fetch(k, v)
        mx.eval(all_k, all_v)

    print(f"  After 6 tokens (step=4): keys={all_k.shape}, offset={cache.offset}")
    assert all_k.shape == (1, 8, 6, 128)
    assert cache.offset == 6
    assert cache.keys.shape[2] == 8  # 4 + 4

    # Test 5: Value correctness
    print("\nTest 5: Value correctness")
    cache = KVCache(step=256)
    k1 = mx.ones((1, 8, 1, 128))
    v1 = mx.ones((1, 8, 1, 128)) * 2
    k2 = mx.ones((1, 8, 1, 128)) * 3
    v2 = mx.ones((1, 8, 1, 128)) * 4

    cache.update_and_fetch(k1, v1)
    all_k, all_v = cache.update_and_fetch(k2, v2)
    mx.eval(all_k, all_v)

    assert mx.allclose(all_k[0, 0, 0, 0], mx.array(1.0)), "First key should be 1.0"
    assert mx.allclose(all_k[0, 0, 1, 0], mx.array(3.0)), "Second key should be 3.0"
    assert mx.allclose(all_v[0, 0, 0, 0], mx.array(2.0)), "First value should be 2.0"
    assert mx.allclose(all_v[0, 0, 1, 0], mx.array(4.0)), "Second value should be 4.0"
    print("  Values match!")

    print("\n✅ KVCache tests passed!")


def test_multi_layer_cache():
    """Test MultiLayerKVCache."""
    print("\n" + "=" * 50)
    print("Multi-Layer KV Cache Tests")
    print("=" * 50)
    print()

    num_layers = 28
    cache = MultiLayerKVCache(num_layers)
    print(f"  Created cache for {len(cache)} layers")
    assert len(cache) == num_layers

    # Simulate forward pass
    for c in cache.get_caches():
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        c.update_and_fetch(k, v)

    assert cache.get_offset() == 1

    # Second token
    for c in cache.get_caches():
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        c.update_and_fetch(k, v)

    assert cache.get_offset() == 2

    # Reset
    cache.reset()
    assert cache.get_offset() == 0

    print("✅ MultiLayerKVCache tests passed!")


if __name__ == "__main__":
    test_kv_cache()
    test_multi_layer_cache()
