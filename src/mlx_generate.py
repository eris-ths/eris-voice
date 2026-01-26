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


def get_default_suppress_tokens() -> List[int]:
    """
    Get default suppress tokens for Qwen3-TTS.

    PyTorch suppresses tokens 2048-3071 EXCEPT 2150 (EOS).
    This ensures the model can only generate EOS from the special token range.
    """
    # Suppress 2048-2149 (before EOS) and 2151-3071 (after EOS)
    return list(range(2048, 2150)) + list(range(2151, 3072))


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
    # Suppress tokens: list of token IDs to set to -inf before sampling
    # Default: suppress all tokens 2048-3071 except 2150 (EOS)
    suppress_tokens: Optional[List[int]] = None

    def __post_init__(self):
        """Initialize default suppress_tokens if not provided."""
        if self.suppress_tokens is None:
            self.suppress_tokens = get_default_suppress_tokens()

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

    def prefill_step(
        self,
        input_embeds: mx.array,
        config: GenerateConfig,
        debug: bool = False,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Prefill step: Run Talker on initial input, sample code_0.

        This is the first step of generation. It runs the Talker on the full
        initial input sequence and samples the first code_0.

        Args:
            input_embeds: Initial embeddings (batch, seq_len, hidden_size)
            config: Generation config
            debug: Print timing info

        Returns:
            Tuple of:
            - code_0: First codebook[0] token (batch,)
            - logits_0: Logits for first code_0 (for debugging)
            - past_hidden: Hidden state to pass to CodePredictor (batch, 1, hidden_size)
        """
        import time
        t0 = time.time()

        # Talker forward on full initial input
        hidden = self.talker(input_embeds, cache=self.talker_cache.get_caches())
        mx.eval(hidden)

        # Get last hidden state for CodePredictor
        past_hidden = hidden[:, -1:, :]  # (batch, 1, hidden_size)

        # Get logits for codebook[0]
        logits_0 = self.codec_head(past_hidden)
        logits_0 = logits_0[:, -1, :]  # (batch, vocab_size)

        t1 = time.time()
        if debug:
            print(f"    Prefill Talker: {(t1-t0)*1000:.1f}ms")

        # Apply suppress_tokens
        if hasattr(self, '_suppress_mask') and self._suppress_mask is not None:
            logits_0 = logits_0 + self._suppress_mask
            if debug:
                logits_np = np.array(logits_0[0])
                eos_logit = logits_np[2150]
                print(f"    Suppression: EOS(2150)={eos_logit:.2f}")

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
            print(f"    Prefill sample: {(t2-t1)*1000:.1f}ms")

        return code_0, logits_0, past_hidden

    def generate_step(
        self,
        past_hidden: mx.array,
        prev_code_0: mx.array,
        trailing_embed: mx.array,
        config: GenerateConfig,
        debug: bool = False,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Generate one step: Run CodePredictor then Talker.

        PyTorch architecture (Generate mode):
        1. CodePredictor([past_hidden, embed(prev_code_0)]) → codes[1-15]
        2. Talker(sum_embeds + trailing) → new logits → sample new code_0
        3. Return new past_hidden for next step

        Args:
            past_hidden: Previous Talker hidden state (batch, 1, hidden_size)
            prev_code_0: Previous step's code_0 (batch,) - used for CodePredictor input
            trailing_embed: Text/pad embedding to add (batch, 1, hidden_size)
            config: Generation config
            debug: Print timing info

        Returns:
            Tuple of:
            - codes: All 16 codes for this step (batch, 16)
            - new_code_0: New code_0 sampled from Talker (batch,)
            - new_past_hidden: New past_hidden for next step (batch, 1, hidden_size)
        """
        import time
        batch_size = past_hidden.shape[0]

        # Reset CodePredictor cache for each step
        self.cp_cache.reset()

        t0 = time.time()

        # ==== 1. CodePredictor: Generate codebook[1-15] ====
        # Input: [past_hidden, embed(prev_code_0)] concatenated
        prev_code_0_embed = self.talker.codec_embedding(prev_code_0[:, None])  # (batch, 1, hidden_size)
        cp_initial_input = mx.concatenate([past_hidden, prev_code_0_embed], axis=1)  # (batch, 2, hidden_size)

        # First CodePredictor forward with 2-token input
        cp_hidden = self.code_predictor(cp_initial_input, cache=self.cp_cache.get_caches())
        mx.eval(cp_hidden)

        # Get logits for codebook[1] (cb_idx=0)
        logits_cb = self.code_predictor.get_logits(cp_hidden[:, -1:, :], 0)
        logits_cb = logits_cb[:, -1, :]

        if config.subtalker_do_sample:
            code_cb = sample_next_token(
                logits_cb,
                temperature=config.subtalker_temperature,
                top_p=config.subtalker_top_p,
                top_k=config.subtalker_top_k,
            )
        else:
            code_cb = greedy_sample(logits_cb)

        codes_1_to_15 = [code_cb]

        # Continue CodePredictor for remaining codebooks
        cp_input = self.code_predictor.embed_codes(code_cb[:, None], 0)

        num_to_generate = config.get_num_acoustic_codebooks()

        for cb_idx in range(1, num_to_generate):  # 1-14 maps to codebook 2-15
            cp_hidden = self.code_predictor(cp_input, cache=self.cp_cache.get_caches())

            logits_cb = self.code_predictor.get_logits(cp_hidden[:, -1:, :], cb_idx)
            logits_cb = logits_cb[:, -1, :]

            if config.subtalker_do_sample:
                code_cb = sample_next_token(
                    logits_cb,
                    temperature=config.subtalker_temperature,
                    top_p=config.subtalker_top_p,
                    top_k=config.subtalker_top_k,
                )
            else:
                code_cb = greedy_sample(logits_cb)

            codes_1_to_15.append(code_cb)
            cp_input = self.code_predictor.embed_codes(code_cb[:, None], cb_idx)

        # Pad with zeros if needed
        if num_to_generate < 15:
            zero_code = mx.zeros((batch_size,), dtype=mx.int32)
            for _ in range(15 - num_to_generate):
                codes_1_to_15.append(zero_code)

        # Evaluate all codes
        for c in codes_1_to_15:
            mx.eval(c)

        t1 = time.time()
        if debug:
            print(f"    CodePredictor ({num_to_generate} codebooks): {(t1-t0)*1000:.1f}ms")

        # ==== 2. Build next Talker input ====
        # sum_embeds = embed(prev_code_0) + sum(embed_cp(codes[1-15]))
        sum_embeds = prev_code_0_embed

        for i, code in enumerate(codes_1_to_15[:num_to_generate]):
            cb_embed = self.code_predictor.embed_codes(code[:, None], i)
            sum_embeds = sum_embeds + cb_embed

        # Add trailing text/pad embedding
        talker_input = sum_embeds + trailing_embed  # (batch, 1, hidden_size)

        # ==== 3. Talker forward for new code_0 ====
        hidden = self.talker(talker_input, cache=self.talker_cache.get_caches())
        mx.eval(hidden)

        new_past_hidden = hidden[:, -1:, :]

        # Get logits for new code_0
        logits_0 = self.codec_head(new_past_hidden)
        logits_0 = logits_0[:, -1, :]

        t2 = time.time()
        if debug:
            print(f"    Talker forward: {(t2-t1)*1000:.1f}ms")

        # Apply suppress_tokens
        if hasattr(self, '_suppress_mask') and self._suppress_mask is not None:
            logits_0 = logits_0 + self._suppress_mask
            if debug:
                logits_np = np.array(logits_0[0])
                eos_logit = logits_np[2150]
                print(f"    Suppression: EOS(2150)={eos_logit:.2f}")

        # Sample new code_0
        if config.do_sample:
            new_code_0 = sample_next_token(
                logits_0,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
        else:
            new_code_0 = greedy_sample(logits_0)

        mx.eval(new_code_0)

        t3 = time.time()
        if debug:
            print(f"    Sample code_0: {(t3-t2)*1000:.1f}ms")

        # Build output: [prev_code_0, codes_1_to_15]
        # Note: This step outputs prev_code_0 (from previous step) + codes[1-15] generated now
        all_codes = mx.stack([prev_code_0] + codes_1_to_15, axis=1)  # (batch, 16)

        return all_codes, new_code_0, new_past_hidden

    def generate(
        self,
        input_embeds: mx.array,
        config: Optional[GenerateConfig] = None,
        trailing_text_hidden: Optional[mx.array] = None,
        tts_pad_embed: Optional[mx.array] = None,
        debug: bool = False,
    ) -> mx.array:
        """
        Generate codec codes autoregressively.

        Architecture (matching PyTorch):
        1. Prefill: Talker(initial_embeds) → past_hidden, logits → sample code_0
        2. Generate loop:
           a. CodePredictor([past_hidden, embed(code_0)]) → codes[1-15]
           b. Talker(sum_embeds + trailing) → new past_hidden, logits → sample new code_0
           c. Output: [code_0, codes[1-15]] for this step

        Args:
            input_embeds: Initial embeddings (batch, seq_len, hidden_size)
            config: Generation configuration
            trailing_text_hidden: Text embeddings to add at each step (batch, text_len, hidden_size)
                                 If provided, adds trailing_text_hidden[:, step] to each step's input.
            tts_pad_embed: Padding embedding to use after text exhausted (batch, 1, hidden_size)
            debug: Print timing info for each step

        Returns:
            Generated codes (batch, num_steps, 16)
        """
        import time

        if config is None:
            config = GenerateConfig()

        self.reset_caches()

        # Precompute suppress_tokens mask for efficiency
        if config.suppress_tokens:
            vocab_size = 3072
            mask_np = np.zeros(vocab_size, dtype=np.float32)
            mask_np[config.suppress_tokens] = float('-inf')
            self._suppress_mask = mx.array(mask_np)
        else:
            self._suppress_mask = None

        batch_size = input_embeds.shape[0]
        all_step_codes = []

        # Get text length for trailing_text_hidden
        text_len = trailing_text_hidden.shape[1] if trailing_text_hidden is not None else 0

        # ==== Prefill: First code_0 ====
        if debug:
            print(f"  Prefill:")

        prefill_start = time.time()
        code_0, logits_0, past_hidden = self.prefill_step(input_embeds, config, debug=debug)

        # Check EOS from prefill
        if mx.any(code_0 == config.codec_eos_token_id):
            if debug:
                print(f"    EOS detected in prefill")
            # Return empty if EOS in first token
            return mx.zeros((batch_size, 0, 16), dtype=mx.int32)

        if debug:
            print(f"    Prefill total: {(time.time() - prefill_start) * 1000:.1f}ms")
            print(f"    First code_0: {int(code_0[0])}")

        # ==== Generate loop ====
        current_code_0 = code_0
        current_past_hidden = past_hidden

        for step in range(config.max_new_tokens):
            step_start = time.time()

            if debug:
                print(f"  Step {step + 1}:")

            # Get trailing embedding for this step
            if trailing_text_hidden is not None and step < text_len:
                trailing_embed = trailing_text_hidden[:, step:step+1, :]
            elif tts_pad_embed is not None:
                trailing_embed = tts_pad_embed
            else:
                trailing_embed = mx.zeros((batch_size, 1, input_embeds.shape[2]))

            # Generate one step
            codes, new_code_0, new_past_hidden = self.generate_step(
                current_past_hidden,
                current_code_0,
                trailing_embed,
                config,
                debug=debug,
            )

            all_step_codes.append(codes)

            step_end = time.time()
            if debug:
                print(f"    Step total: {(step_end - step_start) * 1000:.1f}ms")
                print(f"    code_0={int(codes[0, 0])}, new_code_0={int(new_code_0[0])}")

            # Check EOS in new_code_0 (next step's code_0)
            if mx.any(new_code_0 == config.codec_eos_token_id):
                if debug:
                    print(f"    EOS detected at step {step + 1}")
                # Output final step with EOS
                # Need to run CodePredictor one more time for final codes
                final_codes = self._generate_final_step(
                    current_past_hidden, new_code_0, config, debug
                )
                all_step_codes.append(final_codes)
                break

            # Update for next iteration
            current_code_0 = new_code_0
            current_past_hidden = new_past_hidden

        # Stack all steps: (batch, num_steps, 16)
        if len(all_step_codes) == 0:
            return mx.zeros((batch_size, 0, 16), dtype=mx.int32)

        result = mx.stack(all_step_codes, axis=1)
        mx.eval(result)

        return result

    def _generate_final_step(
        self,
        past_hidden: mx.array,
        eos_code_0: mx.array,
        config: GenerateConfig,
        debug: bool = False,
    ) -> mx.array:
        """
        Generate final step when EOS is detected.

        Only runs CodePredictor to get codes[1-15] for the EOS code_0.
        """
        batch_size = past_hidden.shape[0]
        self.cp_cache.reset()

        # Run CodePredictor with EOS code_0
        eos_embed = self.talker.codec_embedding(eos_code_0[:, None])
        cp_input = mx.concatenate([past_hidden, eos_embed], axis=1)

        cp_hidden = self.code_predictor(cp_input, cache=self.cp_cache.get_caches())
        mx.eval(cp_hidden)

        codes_1_to_15 = []
        num_to_generate = config.get_num_acoustic_codebooks()

        # First codebook
        logits_cb = self.code_predictor.get_logits(cp_hidden[:, -1:, :], 0)
        logits_cb = logits_cb[:, -1, :]

        if config.subtalker_do_sample:
            code_cb = sample_next_token(
                logits_cb,
                temperature=config.subtalker_temperature,
                top_p=config.subtalker_top_p,
                top_k=config.subtalker_top_k,
            )
        else:
            code_cb = greedy_sample(logits_cb)

        codes_1_to_15.append(code_cb)
        cp_input = self.code_predictor.embed_codes(code_cb[:, None], 0)

        # Remaining codebooks
        for cb_idx in range(1, num_to_generate):
            cp_hidden = self.code_predictor(cp_input, cache=self.cp_cache.get_caches())
            logits_cb = self.code_predictor.get_logits(cp_hidden[:, -1:, :], cb_idx)
            logits_cb = logits_cb[:, -1, :]

            if config.subtalker_do_sample:
                code_cb = sample_next_token(
                    logits_cb,
                    temperature=config.subtalker_temperature,
                    top_p=config.subtalker_top_p,
                    top_k=config.subtalker_top_k,
                )
            else:
                code_cb = greedy_sample(logits_cb)

            codes_1_to_15.append(code_cb)
            cp_input = self.code_predictor.embed_codes(code_cb[:, None], cb_idx)

        # Pad if needed
        if num_to_generate < 15:
            zero_code = mx.zeros((batch_size,), dtype=mx.int32)
            for _ in range(15 - num_to_generate):
                codes_1_to_15.append(zero_code)

        for c in codes_1_to_15:
            mx.eval(c)

        all_codes = mx.stack([eos_code_0] + codes_1_to_15, axis=1)
        return all_codes

    def generate_streaming(
        self,
        input_embeds: mx.array,
        config: Optional[GenerateConfig] = None,
        trailing_text_hidden: Optional[mx.array] = None,
        tts_pad_embed: Optional[mx.array] = None,
        buffer_size: int = 5,
        debug: bool = False,
    ):
        """
        Generate codec codes with streaming output.

        Yields codes in chunks of buffer_size steps for progressive decoding.

        Args:
            input_embeds: Initial embeddings (batch, seq_len, hidden_size)
            config: Generation configuration
            trailing_text_hidden: Text embeddings for each step
            tts_pad_embed: Padding embedding after text exhausted
            buffer_size: Number of steps to buffer before yielding
            debug: Print timing info

        Yields:
            (codes_chunk, is_final): codes_chunk is (batch, chunk_steps, 16),
                                     is_final indicates if this is the last chunk
        """
        import time

        if config is None:
            config = GenerateConfig()

        self.reset_caches()

        # Precompute suppress_tokens mask
        if config.suppress_tokens:
            vocab_size = 3072
            mask_np = np.zeros(vocab_size, dtype=np.float32)
            mask_np[config.suppress_tokens] = float('-inf')
            self._suppress_mask = mx.array(mask_np)
        else:
            self._suppress_mask = None

        batch_size = input_embeds.shape[0]
        buffered_codes = []

        text_len = trailing_text_hidden.shape[1] if trailing_text_hidden is not None else 0

        # ==== Prefill ====
        prefill_start = time.time()
        code_0, logits_0, past_hidden = self.prefill_step(input_embeds, config, debug=debug)

        if mx.any(code_0 == config.codec_eos_token_id):
            return  # EOS in prefill, nothing to yield

        if debug:
            print(f"  Prefill: {(time.time() - prefill_start) * 1000:.1f}ms")

        # ==== Generate loop with buffered yield ====
        current_code_0 = code_0
        current_past_hidden = past_hidden

        for step in range(config.max_new_tokens):
            # Get trailing embedding
            if trailing_text_hidden is not None and step < text_len:
                trailing_embed = trailing_text_hidden[:, step:step+1, :]
            elif tts_pad_embed is not None:
                trailing_embed = tts_pad_embed
            else:
                trailing_embed = mx.zeros((batch_size, 1, input_embeds.shape[2]))

            # Generate one step
            codes, new_code_0, new_past_hidden = self.generate_step(
                current_past_hidden,
                current_code_0,
                trailing_embed,
                config,
                debug=debug,
            )

            buffered_codes.append(codes)

            # Check for EOS
            is_eos = mx.any(new_code_0 == config.codec_eos_token_id)

            if is_eos:
                # Generate final step
                final_codes = self._generate_final_step(
                    current_past_hidden, new_code_0, config, debug
                )
                buffered_codes.append(final_codes)

                # Yield remaining buffer
                if buffered_codes:
                    result = mx.stack(buffered_codes, axis=1)
                    mx.eval(result)
                    yield result, True
                return

            # Yield when buffer is full
            if len(buffered_codes) >= buffer_size:
                result = mx.stack(buffered_codes, axis=1)
                mx.eval(result)
                yield result, False
                buffered_codes = []

            current_code_0 = new_code_0
            current_past_hidden = new_past_hidden

        # Yield any remaining codes (max tokens reached)
        if buffered_codes:
            result = mx.stack(buffered_codes, axis=1)
            mx.eval(result)
            yield result, True


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
