#!/usr/bin/env python3
"""
MLX KV Cache for Qwen3-TTS

Key-Value cache for efficient incremental (autoregressive) generation.
Manages caches for multiple transformer layers.
"""

from typing import List, Tuple, Optional
import mlx.core as mx


class KVCache:
    """
    Key-Value cache for a single attention layer.

    Stores accumulated keys and values for incremental generation.
    """

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._offset: int = 0

    @property
    def offset(self) -> int:
        """Current cache offset (number of cached tokens)."""
        return self._offset

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values and return full accumulated tensors.

        Args:
            keys: New keys (batch, n_kv_heads, seq_len, head_dim)
            values: New values (batch, n_kv_heads, seq_len, head_dim)

        Returns:
            Tuple of (all_keys, all_values) including cached
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self._offset = self.keys.shape[2]
        return self.keys, self.values

    def reset(self):
        """Clear the cache for a new generation."""
        self.keys = None
        self.values = None
        self._offset = 0


class MultiLayerKVCache:
    """
    Manages KV caches for multiple transformer layers.

    Usage:
        cache = MultiLayerKVCache(num_layers=28)
        for step in range(max_steps):
            output = model(input, cache=cache.get_caches())
            # caches are automatically updated during forward pass
    """

    def __init__(self, num_layers: int):
        """
        Initialize caches for all layers.

        Args:
            num_layers: Number of transformer layers
        """
        self.num_layers = num_layers
        self.caches = [KVCache() for _ in range(num_layers)]

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
    """Test KVCache functionality."""
    print("=" * 50)
    print("KV Cache Tests")
    print("=" * 50)
    print()

    # Test 1: Basic cache update
    print("Test 1: Basic cache update")
    cache = KVCache()

    # First token
    k1 = mx.random.normal((1, 8, 1, 128))  # (batch, n_kv_heads, seq=1, head_dim)
    v1 = mx.random.normal((1, 8, 1, 128))

    all_k, all_v = cache.update_and_fetch(k1, v1)
    mx.eval(all_k, all_v)

    print(f"  After 1st token: keys={all_k.shape}, offset={cache.offset}")
    assert all_k.shape == (1, 8, 1, 128)
    assert cache.offset == 1

    # Second token
    k2 = mx.random.normal((1, 8, 1, 128))
    v2 = mx.random.normal((1, 8, 1, 128))

    all_k, all_v = cache.update_and_fetch(k2, v2)
    mx.eval(all_k, all_v)

    print(f"  After 2nd token: keys={all_k.shape}, offset={cache.offset}")
    assert all_k.shape == (1, 8, 2, 128)
    assert cache.offset == 2

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

    print("\n✅ KVCache tests passed!")


def test_multi_layer_cache():
    """Test MultiLayerKVCache functionality."""
    print("\n" + "=" * 50)
    print("Multi-Layer KV Cache Tests")
    print("=" * 50)
    print()

    # Test 1: Create multi-layer cache
    print("Test 1: Create multi-layer cache")
    num_layers = 28
    cache = MultiLayerKVCache(num_layers)
    print(f"  Created cache for {len(cache)} layers")
    assert len(cache) == num_layers

    # Test 2: Get caches
    print("\nTest 2: Get caches list")
    caches = cache.get_caches()
    print(f"  Got {len(caches)} caches")
    assert len(caches) == num_layers
    assert all(isinstance(c, KVCache) for c in caches)

    # Test 3: Update caches (simulating forward pass)
    print("\nTest 3: Simulate forward pass updates")

    # Simulate adding one token to all layers
    for i, c in enumerate(caches):
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        c.update_and_fetch(k, v)

    offset = cache.get_offset()
    print(f"  After 1 token: offset={offset}")
    assert offset == 1

    # Add another token
    for c in caches:
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        c.update_and_fetch(k, v)

    offset = cache.get_offset()
    print(f"  After 2 tokens: offset={offset}")
    assert offset == 2

    # Test 4: Reset all
    print("\nTest 4: Reset all caches")
    cache.reset()
    offset = cache.get_offset()
    print(f"  After reset: offset={offset}")
    assert offset == 0
    assert all(c.keys is None for c in caches)

    print("\n✅ MultiLayerKVCache tests passed!")


if __name__ == "__main__":
    test_kv_cache()
    test_multi_layer_cache()
