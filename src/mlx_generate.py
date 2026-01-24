#!/usr/bin/env python3
"""
MLX Generate Loop for Qwen3-TTS

Autoregressive generation loop that combines:
- MLX Talker (28 layers)
- MLX CodePredictor (5 layers)
- MLX Sampling
- KV Cache

Architecture:
    Text tokens → Talker → codec_head → codebook[0]
                        ↓
                  hidden_states → CodePredictor → codebook[1-15]
                                               ↓
                                         16 codec codes per step
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig
from mlx_code_predictor import Qwen3TTSCodePredictorMLX, CodePredictorConfig, load_code_predictor_weights
from mlx_sampling import sample_next_token, greedy_sample
from mlx_kv_cache import MultiLayerKVCache


# Quality mode presets
QUALITY_PRESETS = {
    "high": 15,       # Full quality, RTF ~0.82x
    "balanced": 11,   # Good quality, RTF ~1.0x (realtime)
    "fast": 7,        # Acceptable quality, RTF ~1.4x
    "ultra_fast": 3,  # Reduced quality, RTF ~2.0x
}


@dataclass
class GenerateConfig:
    """Configuration for generation."""
    max_new_tokens: int = 2048
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: Optional[int] = 50
    repetition_penalty: float = 1.05
    codec_eos_token_id: int = 2150
    do_sample: bool = True
    # CodePredictor sampling
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: Optional[int] = 50
    subtalker_do_sample: bool = True
    # Quality mode: "high", "balanced", "fast", "ultra_fast"
    # Or set num_acoustic_codebooks directly for fine control
    quality_mode: str = "balanced"
    num_acoustic_codebooks: Optional[int] = None  # If set, overrides quality_mode

    def get_num_acoustic_codebooks(self) -> int:
        """Get actual number of acoustic codebooks based on quality_mode or direct setting."""
        if self.num_acoustic_codebooks is not None:
            return self.num_acoustic_codebooks
        return QUALITY_PRESETS.get(self.quality_mode, 11)


class MLXGenerateLoop:
    """
    MLX-native autoregressive generation for Qwen3-TTS.

    Generates 16 codec codes per step:
    - Talker + codec_head → codebook[0]
    - CodePredictor → codebook[1-15]
    """

    def __init__(
        self,
        talker: Qwen3TTSTalkerMLX,
        code_predictor: Qwen3TTSCodePredictorMLX,
        codec_head_weight: mx.array,
        use_compile: bool = False,  # TODO: Enable after KVCache refactor to array-based
    ):
        """
        Initialize generator.

        Args:
            talker: MLX Talker model
            code_predictor: MLX CodePredictor model
            codec_head_weight: Weight for codec_head (vocab_size=3072, hidden_size=1024)
            use_compile: Whether to use mx.compile() for acceleration
        """
        self.talker = talker
        self.code_predictor = code_predictor
        self.codec_head_weight = codec_head_weight  # (3072, 1024)
        self.use_compile = use_compile

        # Caches
        self.talker_cache = None
        self.cp_cache = None

        # Compiled functions (lazy initialization)
        self._compiled_talker_step = None
        self._compiled_cp_step = None

    def _get_compiled_talker_step(self):
        """Get or create compiled talker step function."""
        if self._compiled_talker_step is None and self.use_compile:
            talker = self.talker
            codec_head_weight = self.codec_head_weight

            @mx.compile
            def talker_step(input_embeds, cache):
                hidden = talker(input_embeds, cache=cache)
                last_hidden = hidden[:, -1:, :]
                logits = last_hidden @ codec_head_weight.T
                return hidden, logits[:, -1, :]

            self._compiled_talker_step = talker_step
        return self._compiled_talker_step

    def _get_compiled_cp_step(self):
        """Get or create compiled code predictor step function."""
        if self._compiled_cp_step is None and self.use_compile:
            code_predictor = self.code_predictor

            @mx.compile
            def cp_step(cp_input, cache, cb_idx):
                cp_hidden = code_predictor(cp_input, cache=cache)
                logits = code_predictor.get_logits(cp_hidden[:, -1:, :], cb_idx)
                return logits[:, -1, :]

            self._compiled_cp_step = cp_step
        return self._compiled_cp_step

    def reset_caches(self):
        """Reset KV caches for new generation."""
        self.talker_cache = MultiLayerKVCache(28)  # Talker has 28 layers
        self.cp_cache = MultiLayerKVCache(5)  # CodePredictor has 5 layers

    def codec_head(self, hidden: mx.array) -> mx.array:
        """
        Apply codec_head to get logits for codebook 0.

        Args:
            hidden: (batch, seq_len, hidden_size)

        Returns:
            logits: (batch, seq_len, vocab_size=3072)
        """
        # Linear: hidden @ weight.T
        return hidden @ self.codec_head_weight.T

    def apply_repetition_penalty(
        self,
        logits: mx.array,
        generated_codes: mx.array,
        penalty: float,
    ) -> mx.array:
        """
        Apply repetition penalty to discourage repeated tokens.

        Note: Simplified implementation - penalty disabled for now for speed testing.
        TODO: Implement efficient vectorized version.

        Args:
            logits: (batch, vocab_size)
            generated_codes: Previously generated codes (batch, seq_len)
            penalty: Penalty factor (>1 discourages repetition)

        Returns:
            Adjusted logits
        """
        # TODO: Implement efficient repetition penalty
        # For now, skip to test speed
        return logits

    def generate_step(
        self,
        input_embeds: mx.array,
        config: GenerateConfig,
        generated_codes: Optional[mx.array] = None,
        debug: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """
        Generate one step: 16 codec codes.

        Args:
            input_embeds: Current input embeddings (batch, 1, hidden_size)
            config: Generation config
            generated_codes: Previously generated codebook[0] tokens for rep penalty
            debug: Print timing info

        Returns:
            Tuple of:
            - codes: Generated codes for all 16 codebooks (batch, 16)
            - hidden: Talker hidden states for next step
        """
        import time
        batch_size = input_embeds.shape[0]

        # Reset CodePredictor cache for each step (it generates independently per step)
        self.cp_cache.reset()

        t0 = time.time()

        # 1. Talker forward + codec head
        compiled_talker = self._get_compiled_talker_step()
        if compiled_talker is not None:
            # Use compiled version
            hidden, logits_0 = compiled_talker(input_embeds, self.talker_cache.get_caches())
            mx.eval(hidden)
            mx.eval(logits_0)
            last_hidden = hidden[:, -1:, :]
        else:
            # Non-compiled fallback
            hidden = self.talker(input_embeds, cache=self.talker_cache.get_caches())
            mx.eval(hidden)
            last_hidden = hidden[:, -1:, :]
            logits_0 = self.codec_head(last_hidden)
            logits_0 = logits_0[:, -1, :]

        t1 = time.time()
        if debug:
            compile_status = "compiled" if compiled_talker else "uncompiled"
            print(f"    Talker forward ({compile_status}): {(t1-t0)*1000:.1f}ms")

        # Apply repetition penalty
        if generated_codes is not None:
            logits_0 = self.apply_repetition_penalty(
                logits_0, generated_codes, config.repetition_penalty
            )

        # Sample codebook[0]
        if config.do_sample:
            code_0 = sample_next_token(
                logits_0,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
        else:
            code_0 = greedy_sample(logits_0)

        mx.eval(code_0)
        t2 = time.time()
        if debug:
            print(f"    Codebook[0] sample: {(t2-t1)*1000:.1f}ms")

        codes = [code_0]

        # 3. CodePredictor for codebook[1-15] (or fewer based on quality_mode)
        # Use the Talker's last hidden state as input to CodePredictor
        cp_input = last_hidden  # (batch, 1, hidden_size)

        num_to_generate = config.get_num_acoustic_codebooks()  # Based on quality_mode

        for cb_idx in range(num_to_generate):  # 0-14 maps to codebook 1-15
            # CodePredictor forward
            cp_hidden = self.code_predictor(cp_input, cache=self.cp_cache.get_caches())

            # Get logits for this codebook
            logits_cb = self.code_predictor.get_logits(cp_hidden[:, -1:, :], cb_idx)
            logits_cb = logits_cb[:, -1, :]  # (batch, 2048)

            # Sample
            if config.subtalker_do_sample:
                code_cb = sample_next_token(
                    logits_cb,
                    temperature=config.subtalker_temperature,
                    top_p=config.subtalker_top_p,
                    top_k=config.subtalker_top_k,
                )
            else:
                code_cb = greedy_sample(logits_cb)

            codes.append(code_cb)

            # Embed this code for next codebook prediction
            code_embed = self.code_predictor.embed_codes(code_cb[:, None], cb_idx)
            cp_input = code_embed

        # Pad with zeros if we generated fewer than 15 acoustic codebooks
        if num_to_generate < 15:
            zero_code = mx.zeros((batch_size,), dtype=mx.int32)
            for _ in range(15 - num_to_generate):
                codes.append(zero_code)

        # Evaluate all CodePredictor codes at once
        for c in codes[1:]:
            mx.eval(c)

        t3 = time.time()
        if debug:
            print(f"    CodePredictor ({num_to_generate} codebooks): {(t3-t2)*1000:.1f}ms")

        # Stack all codes: (batch, 16)
        all_codes = mx.stack(codes, axis=1)

        return all_codes, last_hidden

    def generate(
        self,
        input_embeds: mx.array,
        config: Optional[GenerateConfig] = None,
        debug: bool = False,
    ) -> mx.array:
        """
        Generate codec codes autoregressively.

        Args:
            input_embeds: Initial embeddings (batch, seq_len, hidden_size)
            config: Generation configuration
            debug: Print timing info for each step

        Returns:
            Generated codes (batch, num_steps, 16)
        """
        import time

        if config is None:
            config = GenerateConfig()

        self.reset_caches()

        batch_size = input_embeds.shape[0]
        generated_codes_0 = []  # For repetition penalty (codebook 0 only)
        all_step_codes = []

        # Process initial input
        current_embeds = input_embeds

        for step in range(config.max_new_tokens):
            step_start = time.time()

            if debug:
                print(f"  Step {step + 1}:")

            # Generate one step
            codes, last_hidden = self.generate_step(
                current_embeds,
                config,
                generated_codes=mx.stack(generated_codes_0, axis=1) if generated_codes_0 else None,
                debug=debug,
            )

            all_step_codes.append(codes)
            generated_codes_0.append(codes[:, 0])

            step_end = time.time()
            if debug:
                print(f"    Step total: {(step_end - step_start) * 1000:.1f}ms")

            # Check EOS
            if mx.any(codes[:, 0] == config.codec_eos_token_id):
                if debug:
                    print(f"    EOS detected at step {step + 1}")
                break

            # Prepare next input: embed the generated codebook[0] token
            next_embed = self.talker.codec_embedding(codes[:, 0:1])  # (batch, 1, hidden_size)
            current_embeds = next_embed

        # Stack all steps: (batch, num_steps, 16)
        result = mx.stack(all_step_codes, axis=1)
        mx.eval(result)

        return result


# ============================================================
# Tests
# ============================================================

def test_generate_loop_basic():
    """Test basic generate loop functionality."""
    print("=" * 60)
    print("Generate Loop Basic Test")
    print("=" * 60)
    print()

    # Create models with random weights (no loading)
    print("Creating models...")
    talker_config = TalkerConfig()
    talker = Qwen3TTSTalkerMLX(talker_config)

    cp_config = CodePredictorConfig()
    code_predictor = Qwen3TTSCodePredictorMLX(cp_config)

    # Random codec_head
    codec_head_weight = mx.random.normal((3072, 1024))

    # Create generator
    generator = MLXGenerateLoop(talker, code_predictor, codec_head_weight)

    # Test single step
    print("\nTest: Single generate step")
    batch_size = 1
    seq_len = 4
    input_embeds = mx.random.normal((batch_size, seq_len, 1024))

    config = GenerateConfig(max_new_tokens=1)
    codes = generator.generate(input_embeds, config)
    mx.eval(codes)

    print(f"  Input: {input_embeds.shape}")
    print(f"  Output: {codes.shape}")
    print(f"  Codes: {codes[0, 0, :].tolist()}")

    assert codes.shape == (batch_size, 1, 16), f"Expected (1, 1, 16), got {codes.shape}"

    print("\n✅ Basic test passed!")
    return True


def test_generate_loop_multi_step():
    """Test multi-step generation."""
    print("\n" + "=" * 60)
    print("Generate Loop Multi-Step Test")
    print("=" * 60)
    print()

    # Create models
    print("Creating models...")
    talker_config = TalkerConfig()
    talker = Qwen3TTSTalkerMLX(talker_config)

    cp_config = CodePredictorConfig()
    code_predictor = Qwen3TTSCodePredictorMLX(cp_config)

    codec_head_weight = mx.random.normal((3072, 1024))

    generator = MLXGenerateLoop(talker, code_predictor, codec_head_weight)

    # Test multi-step
    print("\nTest: Multi-step generation (5 steps)")
    batch_size = 1
    seq_len = 4
    input_embeds = mx.random.normal((batch_size, seq_len, 1024))

    config = GenerateConfig(max_new_tokens=5, do_sample=False)  # Greedy for reproducibility
    codes = generator.generate(input_embeds, config)
    mx.eval(codes)

    print(f"  Generated steps: {codes.shape[1]}")
    print(f"  Codes shape: {codes.shape}")

    # Should have 5 steps (unless EOS hit)
    assert codes.shape[1] <= 5
    assert codes.shape[2] == 16

    print("\n✅ Multi-step test passed!")
    return True


def test_generate_loop_with_weights():
    """Test generate loop with actual weights."""
    print("\n" + "=" * 60)
    print("Generate Loop with Weights Test")
    print("=" * 60)
    print()

    weights_dir = Path(__file__).parent.parent

    # Check weights exist
    talker_weights_path = weights_dir / 'talker_weights_mlx.npz'
    cp_weights_path = weights_dir / 'code_predictor_weights_mlx.npz'

    if not talker_weights_path.exists():
        print(f"⚠️ Talker weights not found: {talker_weights_path}")
        return False
    if not cp_weights_path.exists():
        print(f"⚠️ CodePredictor weights not found: {cp_weights_path}")
        return False

    import time

    # Load Talker
    print("Loading Talker...")
    talker_config = TalkerConfig()
    talker = Qwen3TTSTalkerMLX(talker_config)

    talker_weights = dict(np.load(str(talker_weights_path)))
    # Load talker weights (same as test_mlx_talker.py)
    talker.text_embedding.weight = mx.array(talker_weights['text_embedding.weight'])
    talker.codec_embedding.weight = mx.array(talker_weights['codec_embedding.weight'])
    talker.text_projection_fc1.weight = mx.array(talker_weights['text_projection_fc1.weight'])
    talker.text_projection_fc2.weight = mx.array(talker_weights['text_projection_fc2.weight'])

    for i, layer in enumerate(talker.layers):
        prefix = f'layers.{i}'
        layer.self_attn.q_proj.weight = mx.array(talker_weights[f'{prefix}.self_attn.q_proj.weight'])
        layer.self_attn.k_proj.weight = mx.array(talker_weights[f'{prefix}.self_attn.k_proj.weight'])
        layer.self_attn.v_proj.weight = mx.array(talker_weights[f'{prefix}.self_attn.v_proj.weight'])
        layer.self_attn.o_proj.weight = mx.array(talker_weights[f'{prefix}.self_attn.o_proj.weight'])
        layer.self_attn.q_norm.weight = mx.array(talker_weights[f'{prefix}.self_attn.q_norm.weight'])
        layer.self_attn.k_norm.weight = mx.array(talker_weights[f'{prefix}.self_attn.k_norm.weight'])
        layer.mlp.gate_proj.weight = mx.array(talker_weights[f'{prefix}.mlp.gate_proj.weight'])
        layer.mlp.up_proj.weight = mx.array(talker_weights[f'{prefix}.mlp.up_proj.weight'])
        layer.mlp.down_proj.weight = mx.array(talker_weights[f'{prefix}.mlp.down_proj.weight'])
        layer.input_layernorm.weight = mx.array(talker_weights[f'{prefix}.input_layernorm.weight'])
        layer.post_attention_layernorm.weight = mx.array(talker_weights[f'{prefix}.post_attention_layernorm.weight'])

    talker.norm.weight = mx.array(talker_weights['norm.weight'])
    print("  Talker loaded!")

    # Load codec_head
    codec_head_weight = mx.array(talker_weights['codec_head.weight'])
    print(f"  codec_head loaded: {codec_head_weight.shape}")

    # Load CodePredictor
    print("Loading CodePredictor...")
    cp_config = CodePredictorConfig()
    code_predictor = Qwen3TTSCodePredictorMLX(cp_config)
    code_predictor = load_code_predictor_weights(code_predictor, str(cp_weights_path))

    # Create generator
    generator = MLXGenerateLoop(talker, code_predictor, codec_head_weight)

    # Warmup run (JIT compilation)
    print("\nWarmup run (JIT compilation)...")
    warmup_embeds = mx.random.normal((1, 4, 1024))
    warmup_config = GenerateConfig(max_new_tokens=2, do_sample=False)
    _ = generator.generate(warmup_embeds, warmup_config)
    print("  Warmup done!")

    # Test with random input (simulating text embeddings)
    print("\nTest: Generate 10 steps with actual weights (post-warmup)")
    batch_size = 1
    seq_len = 8
    input_embeds = mx.random.normal((batch_size, seq_len, 1024))

    config = GenerateConfig(
        max_new_tokens=10,
        temperature=0.9,
        top_p=1.0,
        do_sample=True,
    )

    start = time.time()
    codes = generator.generate(input_embeds, config, debug=True)
    mx.eval(codes)
    elapsed = time.time() - start

    print(f"\n  Generated steps: {codes.shape[1]}")
    print(f"  Codes shape: {codes.shape}")
    print(f"  Time: {elapsed:.2f}s ({elapsed / codes.shape[1] * 1000:.1f}ms per step)")
    print(f"  First step codes: {codes[0, 0, :].tolist()}")

    # Check for EOS
    eos_found = mx.any(codes[:, :, 0] == config.codec_eos_token_id)
    print(f"  EOS found: {bool(eos_found)}")

    print("\n✅ Weights test passed!")
    return True


if __name__ == "__main__":
    test_generate_loop_basic()
    test_generate_loop_multi_step()
    test_generate_loop_with_weights()
