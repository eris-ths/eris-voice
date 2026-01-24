#!/usr/bin/env python3
"""
MLX Sampling Functions for Qwen3-TTS

Implements temperature scaling, top-p (nucleus) sampling for autoregressive generation.
"""

import mlx.core as mx
from typing import Optional


def sample_next_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> mx.array:
    """
    Sample next token from logits using temperature and top-p sampling.

    Args:
        logits: Logits from model output, shape (batch, vocab_size) or (vocab_size,)
        temperature: Temperature for scaling logits (higher = more random)
        top_p: Nucleus sampling threshold (1.0 = no filtering)
        top_k: Optional top-k filtering (None = no filtering)

    Returns:
        Sampled token indices, shape (batch,) or scalar
    """
    # Handle 1D input
    squeeze_output = False
    if logits.ndim == 1:
        logits = logits[None, :]  # (1, vocab_size)
        squeeze_output = True

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering (optional)
    if top_k is not None and top_k > 0:
        logits = top_k_filter(logits, top_k)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)

    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)

    # Categorical sampling
    tokens = categorical_sample(probs)

    if squeeze_output:
        tokens = tokens.squeeze(0)

    return tokens


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    """
    Filter logits to keep only top-k values.

    Args:
        logits: Shape (batch, vocab_size)
        k: Number of top values to keep

    Returns:
        Filtered logits with -inf for values outside top-k
    """
    # Sort in descending order
    sorted_logits = mx.sort(logits, axis=-1)
    # Get the k-th largest value (threshold)
    threshold = sorted_logits[:, -k:None][:, 0:1]  # Shape (batch, 1)

    # Mask values below threshold
    mask = logits >= threshold
    filtered = mx.where(mask, logits, mx.array(-float('inf')))
    return filtered


def top_p_filter(logits: mx.array, p: float) -> mx.array:
    """
    Filter logits using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability >= p.

    Args:
        logits: Shape (batch, vocab_size)
        p: Cumulative probability threshold

    Returns:
        Filtered logits with -inf for values outside nucleus
    """
    # Sort logits in descending order
    sorted_indices = mx.argsort(-logits, axis=-1)  # Descending
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Shift cumulative probs to the right (so first token always included)
    # Create mask: keep tokens where cumulative prob (before this token) < p
    shifted_cumulative = mx.concatenate([
        mx.zeros((logits.shape[0], 1)),
        cumulative_probs[:, :-1]
    ], axis=-1)
    sorted_mask = shifted_cumulative < p

    # Set filtered values to -inf
    sorted_logits = mx.where(sorted_mask, sorted_logits, mx.array(-float('inf')))

    # Unsort back to original order using inverse permutation
    # argsort of sorted_indices gives the inverse permutation
    inverse_indices = mx.argsort(sorted_indices, axis=-1)
    unsorted_logits = mx.take_along_axis(sorted_logits, inverse_indices, axis=-1)

    return unsorted_logits


def categorical_sample(probs: mx.array) -> mx.array:
    """
    Sample from categorical distribution.

    Args:
        probs: Probability distribution, shape (batch, vocab_size)

    Returns:
        Sampled indices, shape (batch,)
    """
    # Use Gumbel-max trick for categorical sampling
    # Sample from Gumbel(0, 1)
    u = mx.random.uniform(shape=probs.shape, low=1e-10, high=1.0)
    gumbel = -mx.log(-mx.log(u))

    # Add Gumbel noise to log probabilities and take argmax
    log_probs = mx.log(probs + 1e-10)
    noisy_logits = log_probs + gumbel

    return mx.argmax(noisy_logits, axis=-1)


def greedy_sample(logits: mx.array) -> mx.array:
    """
    Greedy sampling (argmax).

    Args:
        logits: Shape (batch, vocab_size) or (vocab_size,)

    Returns:
        Token indices
    """
    return mx.argmax(logits, axis=-1)


# ============================================================
# Tests
# ============================================================

def test_sampling():
    """Test sampling functions."""
    print("=" * 50)
    print("MLX Sampling Tests")
    print("=" * 50)
    print()

    # Test 1: Basic sampling
    print("Test 1: Basic temperature sampling")
    mx.random.seed(42)
    logits = mx.array([[1.0, 2.0, 3.0, 0.5, 0.1]])

    # Greedy should always pick index 2 (highest logit)
    greedy = greedy_sample(logits)
    print(f"  Greedy: {greedy.item()} (expected: 2)")
    assert greedy.item() == 2, "Greedy sampling failed"

    # With temperature=1.0, sample multiple times
    samples = []
    for _ in range(100):
        token = sample_next_token(logits, temperature=1.0, top_p=1.0)
        samples.append(int(token.item()))

    from collections import Counter
    counts = Counter(samples)
    print(f"  Temperature=1.0 samples: {dict(counts)}")
    # Token 2 should be most common
    assert counts[2] > counts.get(0, 0), "Token 2 should be most common"

    # Test 2: Top-p filtering
    print("\nTest 2: Top-p filtering")
    # With very low top_p, only highest prob token should be sampled
    samples_topp = []
    for _ in range(20):
        token = sample_next_token(logits, temperature=1.0, top_p=0.3)
        samples_topp.append(int(token.item()))

    counts_topp = Counter(samples_topp)
    print(f"  Top-p=0.3 samples: {dict(counts_topp)}")

    # Test 3: Low temperature (more deterministic)
    print("\nTest 3: Low temperature")
    samples_cold = []
    for _ in range(20):
        token = sample_next_token(logits, temperature=0.1, top_p=1.0)
        samples_cold.append(int(token.item()))

    counts_cold = Counter(samples_cold)
    print(f"  Temperature=0.1 samples: {dict(counts_cold)}")
    # Should mostly be token 2
    assert counts_cold[2] >= 18, "Low temperature should be nearly deterministic"

    # Test 4: High temperature (more random)
    print("\nTest 4: High temperature")
    samples_hot = []
    for _ in range(100):
        token = sample_next_token(logits, temperature=2.0, top_p=1.0)
        samples_hot.append(int(token.item()))

    counts_hot = Counter(samples_hot)
    print(f"  Temperature=2.0 samples: {dict(counts_hot)}")
    # Should be more distributed
    unique_tokens = len(counts_hot)
    print(f"  Unique tokens sampled: {unique_tokens}")
    assert unique_tokens >= 3, "High temperature should sample more diverse tokens"

    # Test 5: Batch sampling
    print("\nTest 5: Batch sampling")
    batch_logits = mx.array([
        [1.0, 2.0, 3.0],  # Should favor token 2
        [3.0, 1.0, 2.0],  # Should favor token 0
    ])
    batch_greedy = greedy_sample(batch_logits)
    print(f"  Batch greedy: {batch_greedy.tolist()} (expected: [2, 0])")
    assert batch_greedy.tolist() == [2, 0], "Batch greedy failed"

    print("\n✅ All sampling tests passed!")
    return True


if __name__ == "__main__":
    test_sampling()
