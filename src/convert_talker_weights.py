#!/usr/bin/env python3
"""
Convert PyTorch Qwen3-TTS Talker weights to MLX format.

Usage:
    python convert_talker_weights.py [--output talker_weights_mlx.npz]
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))


def get_talker_config(pt_model):
    """Extract config from PyTorch model."""
    talker = pt_model.model.talker
    config = talker.config

    return {
        'hidden_size': config.hidden_size,
        'num_hidden_layers': config.num_hidden_layers,
        'intermediate_size': config.intermediate_size,
        'num_attention_heads': config.num_attention_heads,
        'num_key_value_heads': getattr(config, 'num_key_value_heads', config.num_attention_heads // 2),
        'rms_norm_eps': getattr(config, 'rms_norm_eps', 1e-6),
        'vocab_size': config.vocab_size,
        'head_dim': config.hidden_size // config.num_attention_heads,
    }


def convert_attention_weights(pt_attn, layer_prefix: str) -> dict:
    """Convert attention layer weights."""
    weights = {}

    # Q, K, V, O projections
    weights[f'{layer_prefix}.self_attn.q_proj.weight'] = pt_attn.q_proj.weight.data
    weights[f'{layer_prefix}.self_attn.k_proj.weight'] = pt_attn.k_proj.weight.data
    weights[f'{layer_prefix}.self_attn.v_proj.weight'] = pt_attn.v_proj.weight.data
    weights[f'{layer_prefix}.self_attn.o_proj.weight'] = pt_attn.o_proj.weight.data

    # Q and K norms
    weights[f'{layer_prefix}.self_attn.q_norm.weight'] = pt_attn.q_norm.weight.data
    weights[f'{layer_prefix}.self_attn.k_norm.weight'] = pt_attn.k_norm.weight.data

    return weights


def convert_mlp_weights(pt_mlp, layer_prefix: str) -> dict:
    """Convert MLP layer weights."""
    weights = {}

    weights[f'{layer_prefix}.mlp.gate_proj.weight'] = pt_mlp.gate_proj.weight.data
    weights[f'{layer_prefix}.mlp.up_proj.weight'] = pt_mlp.up_proj.weight.data
    weights[f'{layer_prefix}.mlp.down_proj.weight'] = pt_mlp.down_proj.weight.data

    return weights


def convert_talker_weights(pt_model) -> dict:
    """Convert all Talker weights to MLX format."""
    talker = pt_model.model.talker
    weights = {}

    print("Converting Talker weights...")

    # Embeddings
    weights['text_embedding.weight'] = talker.model.text_embedding.weight.data
    weights['codec_embedding.weight'] = talker.model.codec_embedding.weight.data

    # Text projection (ResizeMLP: fc1 -> SiLU -> fc2)
    weights['text_projection_fc1.weight'] = talker.text_projection.linear_fc1.weight.data
    weights['text_projection_fc2.weight'] = talker.text_projection.linear_fc2.weight.data

    # Transformer layers
    num_layers = len(talker.model.layers)
    print(f"  Converting {num_layers} transformer layers...")

    for i, layer in enumerate(talker.model.layers):
        layer_prefix = f'layers.{i}'

        # Attention
        weights.update(convert_attention_weights(layer.self_attn, layer_prefix))

        # MLP
        weights.update(convert_mlp_weights(layer.mlp, layer_prefix))

        # Layer norms
        weights[f'{layer_prefix}.input_layernorm.weight'] = layer.input_layernorm.weight.data
        weights[f'{layer_prefix}.post_attention_layernorm.weight'] = layer.post_attention_layernorm.weight.data

        if (i + 1) % 7 == 0:
            print(f"    Layer {i + 1}/{num_layers} done")

    # Final norm
    weights['norm.weight'] = talker.model.norm.weight.data

    print(f"  Total: {len(weights)} weight tensors")
    return weights


def convert_code_predictor_weights(pt_model) -> dict:
    """Convert CodePredictor weights to MLX format."""
    cp = pt_model.model.talker.code_predictor
    weights = {}

    print("Converting CodePredictor weights...")

    # Codec embeddings (multiple)
    for i, emb in enumerate(cp.model.codec_embedding):
        weights[f'codec_embedding.{i}.weight'] = emb.weight.data

    # Transformer layers
    num_layers = len(cp.model.layers)
    print(f"  Converting {num_layers} transformer layers...")

    for i, layer in enumerate(cp.model.layers):
        layer_prefix = f'layers.{i}'

        # Attention
        weights.update(convert_attention_weights(layer.self_attn, layer_prefix))

        # MLP
        weights.update(convert_mlp_weights(layer.mlp, layer_prefix))

        # Layer norms
        weights[f'{layer_prefix}.input_layernorm.weight'] = layer.input_layernorm.weight.data
        weights[f'{layer_prefix}.post_attention_layernorm.weight'] = layer.post_attention_layernorm.weight.data

    # Final norm
    weights['norm.weight'] = cp.model.norm.weight.data

    # LM heads (multiple)
    for i, head in enumerate(cp.lm_head):
        weights[f'lm_head.{i}.weight'] = head.weight.data

    print(f"  Total: {len(weights)} weight tensors")
    return weights


def to_mlx(weights: dict) -> dict:
    """Convert PyTorch tensors to MLX arrays."""
    mlx_weights = {}
    for key, value in weights.items():
        np_value = value.detach().cpu().float().numpy()
        mlx_weights[key] = np_value  # Save as numpy for .npz
    return mlx_weights


def main():
    parser = argparse.ArgumentParser(description='Convert Talker weights to MLX')
    parser.add_argument('--output', type=str, default='talker_weights_mlx.npz',
                        help='Output file path')
    parser.add_argument('--code-predictor-output', type=str, default='code_predictor_weights_mlx.npz',
                        help='CodePredictor output file path')
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-TTS Talker Weight Converter")
    print("=" * 60)
    print()

    # Load PyTorch model
    print("Loading PyTorch model...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    print("Model loaded!")
    print()

    # Get config
    config = get_talker_config(model)
    print("Talker config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Convert Talker weights
    talker_weights = convert_talker_weights(model)
    talker_weights_np = to_mlx(talker_weights)

    # Save Talker weights
    output_path = Path(__file__).parent.parent / args.output
    np.savez(output_path, **talker_weights_np)
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Talker weights saved to: {output_path} ({file_size:.1f} MB)")

    # Convert CodePredictor weights
    cp_weights = convert_code_predictor_weights(model)
    cp_weights_np = to_mlx(cp_weights)

    # Save CodePredictor weights
    cp_output_path = Path(__file__).parent.parent / args.code_predictor_output
    np.savez(cp_output_path, **cp_weights_np)
    cp_file_size = cp_output_path.stat().st_size / (1024 * 1024)
    print(f"✅ CodePredictor weights saved to: {cp_output_path} ({cp_file_size:.1f} MB)")

    print()
    print("=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
