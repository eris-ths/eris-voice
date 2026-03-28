#!/usr/bin/env python3
"""
Precompute Lloyd-Max quantization codebooks for PolarQuant.

After random orthogonal rotation, each coordinate of a d-dimensional unit vector
follows a Beta((d-1)/2, (d-1)/2) distribution scaled to [-1, 1].
Lloyd-Max finds optimal quantization levels that minimize MSE for this distribution.

Usage:
    python compute_lloyd_max.py
    # Outputs: lloyd_max_codebooks.npz
"""

import numpy as np
from scipy import stats, integrate
from pathlib import Path


def compute_lloyd_max_codebook(dim: int, bits: int, n_iter: int = 100):
    """
    Compute Lloyd-Max quantization levels for the marginal distribution
    of unit vector coordinates after random rotation.

    Args:
        dim: Vector dimension (e.g., 64, 128, 256)
        bits: Quantization bit width (1-4)
        n_iter: Number of Lloyd-Max iterations

    Returns:
        levels: (2^bits,) optimal quantization levels
    """
    n_levels = 2 ** bits

    # Marginal distribution of a coordinate of a uniform point on S^{d-1}
    # is Beta((d-1)/2, (d-1)/2) scaled from [0,1] to [-1,1]
    a = (dim - 1) / 2
    dist = stats.beta(a, a, loc=-1, scale=2)

    # Initialize levels uniformly in the high-density region
    # Use quantiles for better initial placement
    quantiles = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
    levels = dist.ppf(quantiles)

    for _ in range(n_iter):
        # Boundaries: midpoints between levels
        boundaries = np.concatenate([[-1.0], (levels[:-1] + levels[1:]) / 2, [1.0]])

        # Update levels: centroid of each region under the distribution
        new_levels = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            num, _ = integrate.quad(lambda x: x * dist.pdf(x), lo, hi)
            den, _ = integrate.quad(lambda x: dist.pdf(x), lo, hi)
            new_levels[i] = num / den if den > 1e-10 else (lo + hi) / 2

        levels = new_levels

    return levels.astype(np.float32)


def main():
    dims = [64, 128, 256]
    bits_range = [2, 3, 4]

    codebooks = {}

    for dim in dims:
        for bits in bits_range:
            key = f"dim{dim}_bits{bits}"
            levels = compute_lloyd_max_codebook(dim, bits)

            # Verify: compare MSE with uniform quantization
            dist = stats.beta((dim - 1) / 2, (dim - 1) / 2, loc=-1, scale=2)
            x = dist.rvs(100000)

            # Uniform
            uniform_levels = np.linspace(-1, 1, 2 ** bits)
            uniform_q = uniform_levels[np.argmin(np.abs(x[:, None] - uniform_levels[None, :]), axis=1)]
            uniform_mse = np.mean((x - uniform_q) ** 2)

            # Lloyd-Max
            lm_q = levels[np.argmin(np.abs(x[:, None] - levels[None, :]), axis=1)]
            lm_mse = np.mean((x - lm_q) ** 2)

            improvement = uniform_mse / lm_mse

            codebooks[key] = levels
            print(f"{key}: {2**bits} levels, MSE improvement {improvement:.1f}x")
            print(f"  levels: {np.round(levels, 4).tolist()}")

    # Save
    out_path = Path(__file__).parent.parent / "lloyd_max_codebooks.npz"
    np.savez(str(out_path), **codebooks)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
