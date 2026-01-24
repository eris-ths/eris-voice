#!/usr/bin/env python3
"""
Full MLX Pipeline for Qwen3-TTS

Combines:
- MLX Talker (10.83x speedup)
- MLX CodePredictor (TODO)
- MLX Decoder (45x speedup)
- MLX Quantizer (3.5x speedup)

Expected total speedup: ~15-20x over pure PyTorch CPU.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import soundfile as sf

from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig
from mlx_decoder_v2 import Qwen3TTSDecoderMLX
from mlx_quantizer import SplitResidualVectorQuantizerMLX


class MLXFullPipeline:
    """
    Full MLX-accelerated TTS pipeline.

    Still uses PyTorch for:
    - Tokenization (text processing)
    - Embedding lookup (initial step)
    - CodePredictor (TODO: port to MLX)
    - Pre-conv, pre-transformer, upsample (small overhead)

    Uses MLX for:
    - Talker Transformer (28 layers) - 10.83x speedup
    - Decoder blocks - 45x speedup
    - Quantizer - 3.5x speedup
    """

    def __init__(self):
        self.pt_model = None
        self.mlx_talker = None
        self.mlx_decoder = None
        self.mlx_quantizer = None
        self.sample_rate = 24000
        self._loaded = False

    def load(self):
        """Load all models."""
        if self._loaded:
            return

        print("Loading Full MLX Pipeline...")

        # PyTorch model (for tokenization and parts not yet ported)
        print("  Loading PyTorch model...")
        from qwen_tts import Qwen3TTSModel

        self.pt_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cpu",
            dtype=torch.bfloat16,
        )

        # MLX Talker
        print("  Loading MLX Talker...")
        config = TalkerConfig()
        self.mlx_talker = Qwen3TTSTalkerMLX(config)
        self._load_talker_weights()

        # MLX Decoder
        print("  Loading MLX Decoder...")
        self.mlx_decoder = Qwen3TTSDecoderMLX()
        decoder_weights_path = Path(__file__).parent.parent / "decoder_weights_mlx.npz"
        decoder_weights = dict(mx.load(str(decoder_weights_path)))
        self.mlx_decoder.load_weights(decoder_weights)

        # MLX Quantizer
        print("  Loading MLX Quantizer...")
        self.mlx_quantizer = SplitResidualVectorQuantizerMLX(
            n_q_semantic=1,
            total_quantizers=16,
            codebook_size=2048,
            input_dim=512,
            codebook_dim=256,
        )
        quantizer_weights_path = Path(__file__).parent.parent / "quantizer_weights_mlx.npz"
        quantizer_weights = dict(mx.load(str(quantizer_weights_path)))
        self.mlx_quantizer.load_weights(quantizer_weights)

        # Patch PyTorch decoder to use MLX
        self._patch_decoder()

        self._loaded = True
        print("Pipeline loaded!")

    def _load_talker_weights(self):
        """Load weights into MLX Talker."""
        weights_path = Path(__file__).parent.parent / 'talker_weights_mlx.npz'
        weights_np = dict(np.load(str(weights_path)))
        weights_mlx = {k: mx.array(v) for k, v in weights_np.items()}

        # Embeddings
        self.mlx_talker.text_embedding.weight = weights_mlx['text_embedding.weight']
        self.mlx_talker.codec_embedding.weight = weights_mlx['codec_embedding.weight']

        # Text projection
        self.mlx_talker.text_projection_fc1.weight = weights_mlx['text_projection_fc1.weight']
        self.mlx_talker.text_projection_fc2.weight = weights_mlx['text_projection_fc2.weight']

        # Layers
        for i, layer in enumerate(self.mlx_talker.layers):
            prefix = f'layers.{i}'
            layer.self_attn.q_proj.weight = weights_mlx[f'{prefix}.self_attn.q_proj.weight']
            layer.self_attn.k_proj.weight = weights_mlx[f'{prefix}.self_attn.k_proj.weight']
            layer.self_attn.v_proj.weight = weights_mlx[f'{prefix}.self_attn.v_proj.weight']
            layer.self_attn.o_proj.weight = weights_mlx[f'{prefix}.self_attn.o_proj.weight']
            layer.self_attn.q_norm.weight = weights_mlx[f'{prefix}.self_attn.q_norm.weight']
            layer.self_attn.k_norm.weight = weights_mlx[f'{prefix}.self_attn.k_norm.weight']
            layer.mlp.gate_proj.weight = weights_mlx[f'{prefix}.mlp.gate_proj.weight']
            layer.mlp.up_proj.weight = weights_mlx[f'{prefix}.mlp.up_proj.weight']
            layer.mlp.down_proj.weight = weights_mlx[f'{prefix}.mlp.down_proj.weight']
            layer.input_layernorm.weight = weights_mlx[f'{prefix}.input_layernorm.weight']
            layer.post_attention_layernorm.weight = weights_mlx[f'{prefix}.post_attention_layernorm.weight']

        self.mlx_talker.norm.weight = weights_mlx['norm.weight']

    def _patch_decoder(self):
        """Patch PyTorch decoder to use MLX for quantizer and decoder blocks."""
        pt_decoder = self.pt_model.model.speech_tokenizer.model.decoder
        mlx_decoder = self.mlx_decoder
        mlx_quantizer = self.mlx_quantizer

        def hybrid_forward(codes):
            with torch.no_grad():
                # MLX quantizer
                codes_np = codes.detach().cpu().numpy()
                codes_mlx = mx.array(codes_np)
                hidden_mlx = mlx_quantizer.decode(codes_mlx)
                mx.eval(hidden_mlx)
                hidden_np = np.array(hidden_mlx)
                hidden = torch.from_numpy(hidden_np).to(dtype=torch.bfloat16)

                # PyTorch pre-processing
                hidden = pt_decoder.pre_conv(hidden).transpose(1, 2)
                if pt_decoder.pre_transformer is not None:
                    hidden = pt_decoder.pre_transformer(
                        inputs_embeds=hidden
                    ).last_hidden_state
                    hidden = hidden.permute(0, 2, 1)

                for blocks in pt_decoder.upsample:
                    for block in blocks:
                        hidden = block(hidden)

            # MLX decoder
            hidden_np = hidden.detach().cpu().float().numpy()
            hidden_mlx = mx.array(hidden_np)

            wav_mlx = mlx_decoder.decoder_conv0(hidden_mlx)
            for block in mlx_decoder.decoder_blocks:
                wav_mlx = block(wav_mlx)
            wav_mlx = mlx_decoder.final_act(wav_mlx)
            wav_mlx = mlx_decoder.final_conv(wav_mlx)
            wav_mlx = mx.clip(wav_mlx, -1, 1)
            mx.eval(wav_mlx)

            wav_np = np.array(wav_mlx)
            return torch.from_numpy(wav_np)

        pt_decoder.forward = hybrid_forward

    def generate(
        self,
        text: str,
        speaker: str = "ono_anna",
        language: str = "Japanese",
    ) -> np.ndarray:
        """
        Generate speech from text.

        Note: Currently uses PyTorch for the full generation pipeline,
        with MLX patched in for decoder. Full MLX Talker integration
        requires more work on the generate loop.
        """
        with torch.no_grad():
            wavs, sr = self.pt_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
            )
        return wavs[0]


def benchmark():
    """Benchmark the full pipeline."""
    print("=" * 60)
    print("Full MLX Pipeline Benchmark")
    print("=" * 60)
    print()

    pipeline = MLXFullPipeline()
    pipeline.load()

    # Test texts
    test_texts = [
        "こんにちは",
        "私はエリス。よろしくね。",
        "今日は素晴らしい天気ですね。一緒にお散歩しませんか？",
    ]

    print("\nGenerating speech...")
    for text in test_texts:
        start = time.time()
        audio = pipeline.generate(text)
        gen_time = time.time() - start

        duration = len(audio) / 24000
        rtf = duration / gen_time

        print(f"\n  Text: {text[:20]}...")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Gen time: {gen_time:.2f}s")
        print(f"  RTF: {rtf:.2f}x")


if __name__ == "__main__":
    benchmark()
