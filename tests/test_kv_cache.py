"""
Tests for Pre-allocated KV Cache (MLX-LM based)
"""

import mlx.core as mx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_kv_cache import KVCache, MultiLayerKVCache


def test_basic_update():
    """Test basic cache update with O(1) slice assignment."""
    cache = KVCache(step=256)

    k1 = mx.random.normal((1, 8, 1, 128))
    v1 = mx.random.normal((1, 8, 1, 128))

    all_k, all_v = cache.update_and_fetch(k1, v1)
    mx.eval(all_k, all_v)

    assert all_k.shape == (1, 8, 1, 128)
    assert cache.offset == 1
    assert cache.keys.shape[2] == 256  # Pre-allocated


def test_pre_allocation():
    """Test buffer is pre-allocated and reused."""
    cache = KVCache(step=256)

    for i in range(10):
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        cache.update_and_fetch(k, v)

    assert cache.offset == 10
    assert cache.keys.shape[2] == 256  # No growth


def test_buffer_extension():
    """Test buffer extends when full."""
    cache = KVCache(step=4)

    for i in range(6):
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        all_k, all_v = cache.update_and_fetch(k, v)
        mx.eval(all_k, all_v)

    assert all_k.shape == (1, 8, 6, 128)
    assert cache.keys.shape[2] == 8  # 4 + 4


def test_prefill():
    """Test multi-token prefill."""
    cache = KVCache(step=256)
    k = mx.random.normal((1, 8, 20, 128))
    v = mx.random.normal((1, 8, 20, 128))

    all_k, all_v = cache.update_and_fetch(k, v)
    mx.eval(all_k, all_v)

    assert all_k.shape == (1, 8, 20, 128)
    assert cache.offset == 20


def test_value_correctness():
    """Test values are stored and retrieved correctly."""
    cache = KVCache(step=256)

    k1 = mx.ones((1, 8, 1, 128))
    v1 = mx.ones((1, 8, 1, 128)) * 2
    k2 = mx.ones((1, 8, 1, 128)) * 3
    v2 = mx.ones((1, 8, 1, 128)) * 4

    cache.update_and_fetch(k1, v1)
    all_k, all_v = cache.update_and_fetch(k2, v2)
    mx.eval(all_k, all_v)

    assert mx.allclose(all_k[0, 0, 0, 0], mx.array(1.0))
    assert mx.allclose(all_k[0, 0, 1, 0], mx.array(3.0))
    assert mx.allclose(all_v[0, 0, 0, 0], mx.array(2.0))
    assert mx.allclose(all_v[0, 0, 1, 0], mx.array(4.0))


def test_reset():
    """Test cache reset."""
    cache = KVCache(step=256)
    k = mx.random.normal((1, 8, 5, 128))
    v = mx.random.normal((1, 8, 5, 128))
    cache.update_and_fetch(k, v)

    cache.reset()
    assert cache.offset == 0
    assert cache.keys is None


def test_multi_layer():
    """Test MultiLayerKVCache with 28 layers."""
    cache = MultiLayerKVCache(28)
    assert len(cache) == 28

    for c in cache.get_caches():
        k = mx.random.normal((1, 8, 1, 128))
        v = mx.random.normal((1, 8, 1, 128))
        c.update_and_fetch(k, v)

    assert cache.get_offset() == 1

    cache.reset()
    assert cache.get_offset() == 0


if __name__ == "__main__":
    test_basic_update()
    test_pre_allocation()
    test_buffer_extension()
    test_prefill()
    test_value_correctness()
    test_reset()
    test_multi_layer()
    print("✅ All KV Cache tests passed!")
