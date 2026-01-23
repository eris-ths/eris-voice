"""
PyTorch → MLX Weight Converter for Qwen3-TTS

PyTorch の decoder/quantizer weights を MLX 形式に変換する。
"""

import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Any


def pytorch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array."""
    # Convert bfloat16 to float32 first (numpy doesn't support bfloat16)
    return mx.array(tensor.detach().cpu().float().numpy())


def convert_conv1d_weight(weight: torch.Tensor) -> mx.array:
    """
    Convert PyTorch Conv1d weight to MLX format.

    PyTorch Conv1d weight: (out_channels, in_channels, kernel_size)
    MLX conv1d weight: (out_channels, kernel_size, in_channels)
    """
    # (out, in, k) -> (out, k, in)
    weight_np = weight.detach().cpu().float().numpy()
    weight_np = np.transpose(weight_np, (0, 2, 1))
    return mx.array(weight_np)


def convert_conv_transpose1d_weight(weight: torch.Tensor) -> mx.array:
    """
    Convert PyTorch ConvTranspose1d weight to MLX format.

    PyTorch ConvTranspose1d weight: (in_channels, out_channels, kernel_size)
    MLX conv_transpose1d weight: (out_channels, kernel_size, in_channels)
    """
    # (in, out, k) -> (out, k, in)
    weight_np = weight.detach().cpu().float().numpy()
    weight_np = np.transpose(weight_np, (1, 2, 0))
    return mx.array(weight_np)


def extract_decoder_weights(pytorch_model) -> Dict[str, Any]:
    """
    Extract decoder weights from PyTorch Qwen3-TTS model.

    Returns a dictionary with MLX-compatible weights.
    """
    decoder = pytorch_model.model.speech_tokenizer.model.decoder

    weights = {}

    # Pre-conv (latent_dim -> decoder_dim)
    weights["pre_conv.weight"] = convert_conv1d_weight(decoder.pre_conv.conv.weight)
    weights["pre_conv.bias"] = pytorch_to_mlx(decoder.pre_conv.conv.bias)

    # Upsampling blocks
    for i, blocks in enumerate(decoder.upsample):
        # TransConv
        transconv = blocks[0]
        weights[f"upsample.{i}.transconv.weight"] = convert_conv_transpose1d_weight(
            transconv.conv.weight
        )
        weights[f"upsample.{i}.transconv.bias"] = pytorch_to_mlx(transconv.conv.bias)

        # ConvNeXt block
        convnext = blocks[1]
        weights[f"upsample.{i}.convnext.dwconv.weight"] = convert_conv1d_weight(
            convnext.dwconv.conv.weight
        )
        weights[f"upsample.{i}.convnext.dwconv.bias"] = pytorch_to_mlx(
            convnext.dwconv.conv.bias
        )
        weights[f"upsample.{i}.convnext.norm.weight"] = pytorch_to_mlx(
            convnext.norm.weight
        )
        weights[f"upsample.{i}.convnext.norm.bias"] = pytorch_to_mlx(convnext.norm.bias)
        weights[f"upsample.{i}.convnext.pwconv1.weight"] = pytorch_to_mlx(
            convnext.pwconv1.weight
        )
        weights[f"upsample.{i}.convnext.pwconv1.bias"] = pytorch_to_mlx(
            convnext.pwconv1.bias
        )
        weights[f"upsample.{i}.convnext.pwconv2.weight"] = pytorch_to_mlx(
            convnext.pwconv2.weight
        )
        weights[f"upsample.{i}.convnext.pwconv2.bias"] = pytorch_to_mlx(
            convnext.pwconv2.bias
        )
        weights[f"upsample.{i}.convnext.gamma"] = pytorch_to_mlx(convnext.gamma)

    # Decoder blocks
    for i, block in enumerate(decoder.decoder):
        block_name = f"decoder.{i}"

        if hasattr(block, "conv"):
            # CausalConvNet
            weights[f"{block_name}.weight"] = convert_conv1d_weight(block.conv.weight)
            weights[f"{block_name}.bias"] = pytorch_to_mlx(block.conv.bias)

        elif hasattr(block, "block"):
            # DecoderBlock with transconv and residual units
            # First is SnakeBeta activation (has beta parameter)
            snake = block.block[0]
            if hasattr(snake, "beta"):
                weights[f"{block_name}.act.beta"] = pytorch_to_mlx(snake.beta)

            # Second is TransConv
            transconv = block.block[1]
            weights[f"{block_name}.transconv.weight"] = convert_conv_transpose1d_weight(
                transconv.conv.weight
            )
            weights[f"{block_name}.transconv.bias"] = pytorch_to_mlx(
                transconv.conv.bias
            )

            # Residual units (3 units with different dilations)
            for j, unit in enumerate(block.block[2:]):
                unit_name = f"{block_name}.residual.{j}"
                if hasattr(unit, "act1"):
                    weights[f"{unit_name}.act1.beta"] = pytorch_to_mlx(unit.act1.beta)
                    weights[f"{unit_name}.conv1.weight"] = convert_conv1d_weight(
                        unit.conv1.conv.weight
                    )
                    weights[f"{unit_name}.conv1.bias"] = pytorch_to_mlx(
                        unit.conv1.conv.bias
                    )
                    weights[f"{unit_name}.act2.beta"] = pytorch_to_mlx(unit.act2.beta)
                    weights[f"{unit_name}.conv2.weight"] = convert_conv1d_weight(
                        unit.conv2.conv.weight
                    )
                    weights[f"{unit_name}.conv2.bias"] = pytorch_to_mlx(
                        unit.conv2.conv.bias
                    )

        elif hasattr(block, "beta"):
            # SnakeBeta activation
            weights[f"{block_name}.beta"] = pytorch_to_mlx(block.beta)

    return weights


def extract_quantizer_weights(pytorch_model) -> Dict[str, Any]:
    """
    Extract quantizer weights from PyTorch Qwen3-TTS model.

    Returns a dictionary with MLX-compatible weights.
    """
    quantizer = pytorch_model.model.speech_tokenizer.model.decoder.quantizer

    weights = {}

    # rvq_first
    weights["rvq_first.input_proj.weight"] = convert_conv1d_weight(
        quantizer.rvq_first.input_proj.weight
    )
    weights["rvq_first.output_proj.weight"] = convert_conv1d_weight(
        quantizer.rvq_first.output_proj.weight
    )

    for i, layer in enumerate(quantizer.rvq_first.vq.layers):
        cb = layer._codebook
        prefix = f"rvq_first.vq.layers.{i}"
        weights[f"{prefix}.embedding_sum"] = pytorch_to_mlx(cb.embedding_sum)
        weights[f"{prefix}.cluster_usage"] = pytorch_to_mlx(cb.cluster_usage)

    # rvq_rest
    weights["rvq_rest.input_proj.weight"] = convert_conv1d_weight(
        quantizer.rvq_rest.input_proj.weight
    )
    weights["rvq_rest.output_proj.weight"] = convert_conv1d_weight(
        quantizer.rvq_rest.output_proj.weight
    )

    for i, layer in enumerate(quantizer.rvq_rest.vq.layers):
        cb = layer._codebook
        prefix = f"rvq_rest.vq.layers.{i}"
        weights[f"{prefix}.embedding_sum"] = pytorch_to_mlx(cb.embedding_sum)
        weights[f"{prefix}.cluster_usage"] = pytorch_to_mlx(cb.cluster_usage)

    return weights


def save_mlx_weights(weights: Dict[str, mx.array], path: str) -> None:
    """Save MLX weights to file."""
    mx.savez(path, **weights)
    print(f"Saved MLX weights to {path}")


def load_mlx_weights(path: str) -> Dict[str, mx.array]:
    """Load MLX weights from file."""
    data = mx.load(path)
    return dict(data)


if __name__ == "__main__":
    print("=== Weight Converter ===")

    # Load PyTorch model
    from qwen_tts import Qwen3TTSModel

    print("Loading PyTorch model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    # Extract decoder weights
    print("\n1. Extracting decoder weights...")
    decoder_weights = extract_decoder_weights(model)
    print(f"   Extracted {len(decoder_weights)} decoder weight tensors")

    # Save decoder weights
    save_mlx_weights(decoder_weights, "decoder_weights_mlx.npz")

    # Extract quantizer weights
    print("\n2. Extracting quantizer weights...")
    quantizer_weights = extract_quantizer_weights(model)
    print(f"   Extracted {len(quantizer_weights)} quantizer weight tensors")

    for name, w in list(quantizer_weights.items())[:5]:
        print(f"     {name}: {w.shape}")
    print("     ...")

    # Save quantizer weights
    save_mlx_weights(quantizer_weights, "quantizer_weights_mlx.npz")

    print("\nDone!")
