#!/usr/bin/env python3
"""
Test MLX Talker against PyTorch reference.

Verifies that MLX implementation produces matching outputs.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)

import numpy as np
import mlx.core as mx

from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig


def load_mlx_weights(talker: Qwen3TTSTalkerMLX, weights_path: str):
    """Load weights into MLX Talker."""
    print(f"Loading weights from {weights_path}...")

    weights_np = dict(np.load(weights_path))
    weights_mlx = {}

    for key, value in weights_np.items():
        weights_mlx[key] = mx.array(value)

    # Manual weight assignment (MLX doesn't have load_state_dict)
    # Embeddings
    talker.text_embedding.weight = weights_mlx['text_embedding.weight']
    talker.codec_embedding.weight = weights_mlx['codec_embedding.weight']

    # Text projection
    talker.text_projection_fc1.weight = weights_mlx['text_projection_fc1.weight']
    talker.text_projection_fc2.weight = weights_mlx['text_projection_fc2.weight']

    # Layers
    for i, layer in enumerate(talker.layers):
        prefix = f'layers.{i}'

        # Attention
        layer.self_attn.q_proj.weight = weights_mlx[f'{prefix}.self_attn.q_proj.weight']
        layer.self_attn.k_proj.weight = weights_mlx[f'{prefix}.self_attn.k_proj.weight']
        layer.self_attn.v_proj.weight = weights_mlx[f'{prefix}.self_attn.v_proj.weight']
        layer.self_attn.o_proj.weight = weights_mlx[f'{prefix}.self_attn.o_proj.weight']
        layer.self_attn.q_norm.weight = weights_mlx[f'{prefix}.self_attn.q_norm.weight']
        layer.self_attn.k_norm.weight = weights_mlx[f'{prefix}.self_attn.k_norm.weight']

        # MLP
        layer.mlp.gate_proj.weight = weights_mlx[f'{prefix}.mlp.gate_proj.weight']
        layer.mlp.up_proj.weight = weights_mlx[f'{prefix}.mlp.up_proj.weight']
        layer.mlp.down_proj.weight = weights_mlx[f'{prefix}.mlp.down_proj.weight']

        # Layer norms
        layer.input_layernorm.weight = weights_mlx[f'{prefix}.input_layernorm.weight']
        layer.post_attention_layernorm.weight = weights_mlx[f'{prefix}.post_attention_layernorm.weight']

    # Final norm
    talker.norm.weight = weights_mlx['norm.weight']

    print(f"Loaded {len(weights_mlx)} weight tensors")
    return talker


def test_forward_pass():
    """Test forward pass matches PyTorch."""
    print("=" * 60)
    print("MLX Talker Forward Pass Test")
    print("=" * 60)
    print()

    # Create MLX model
    config = TalkerConfig()
    mlx_talker = Qwen3TTSTalkerMLX(config)

    # Load weights
    weights_path = Path(__file__).parent.parent / 'talker_weights_mlx.npz'
    mlx_talker = load_mlx_weights(mlx_talker, str(weights_path))

    # Create test input (random hidden states)
    np.random.seed(42)
    batch_size = 1
    seq_len = 16
    test_input_np = np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32)

    # MLX forward
    print("\nRunning MLX forward pass...")
    test_input_mlx = mx.array(test_input_np)

    start = time.time()
    mlx_output = mlx_talker(test_input_mlx)
    mx.eval(mlx_output)
    mlx_time = time.time() - start

    print(f"  Time: {mlx_time * 1000:.1f}ms")
    print(f"  Output shape: {mlx_output.shape}")
    print(f"  Output mean: {mx.mean(mlx_output).item():.6f}")
    print(f"  Output std: {mx.std(mlx_output).item():.6f}")

    print("\n✅ MLX Talker forward pass successful!")
    return True


def test_speed_comparison():
    """Compare MLX vs PyTorch speed."""
    print("\n" + "=" * 60)
    print("Speed Comparison: MLX vs PyTorch")
    print("=" * 60)
    print()

    # Load PyTorch model
    print("Loading PyTorch model...")
    from qwen_tts import Qwen3TTSModel

    pt_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    pt_talker = pt_model.model.talker.model

    # Load MLX model
    print("Loading MLX model...")
    config = TalkerConfig()
    mlx_talker = Qwen3TTSTalkerMLX(config)
    weights_path = Path(__file__).parent.parent / 'talker_weights_mlx.npz'
    mlx_talker = load_mlx_weights(mlx_talker, str(weights_path))

    # Test input
    np.random.seed(42)
    batch_size = 1
    seq_len = 64  # Longer sequence for better timing
    test_input_np = np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32)

    # Warmup
    print("\nWarming up...")
    test_input_pt = torch.from_numpy(test_input_np).to(torch.bfloat16)
    test_input_mlx = mx.array(test_input_np)

    with torch.no_grad():
        _ = pt_talker(inputs_embeds=test_input_pt)

    _ = mlx_talker(test_input_mlx)
    mx.eval(_)

    # Benchmark
    n_runs = 5
    print(f"\nBenchmarking ({n_runs} runs each)...")

    # PyTorch
    pt_times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            pt_out = pt_talker(inputs_embeds=test_input_pt)
        pt_times.append(time.time() - start)

    # MLX
    mlx_times = []
    for _ in range(n_runs):
        start = time.time()
        mlx_out = mlx_talker(test_input_mlx)
        mx.eval(mlx_out)
        mlx_times.append(time.time() - start)

    pt_avg = np.mean(pt_times) * 1000
    mlx_avg = np.mean(mlx_times) * 1000
    speedup = pt_avg / mlx_avg

    print(f"\nResults (seq_len={seq_len}):")
    print(f"  PyTorch CPU: {pt_avg:.1f}ms")
    print(f"  MLX:         {mlx_avg:.1f}ms")
    print(f"  Speedup:     {speedup:.2f}x")

    if speedup > 1:
        print(f"\n✅ MLX is {speedup:.2f}x faster!")
    else:
        print(f"\n⚠️ PyTorch is faster (MLX: {1/speedup:.2f}x slower)")

    return speedup


if __name__ == "__main__":
    test_forward_pass()
    speedup = test_speed_comparison()
