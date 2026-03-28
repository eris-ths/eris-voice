#!/usr/bin/env python3
"""
MLX Text Encoder for Qwen3-TTS

PyTorch-free text processing: tokenization + embedding + projection + token assembly.
Replaces the PyTorch model monkey-patch approach in mlx_pipeline.py.

Uses:
- transformers.AutoTokenizer (PyTorch-independent)
- Existing MLX Talker weights (text_embedding, text_projection, codec_embedding)
"""

from typing import Tuple, Optional
import mlx.core as mx
import mlx.nn as nn

from transformers import AutoTokenizer


# ─── Token ID Constants ───
# From Qwen3-TTS-12Hz-0.6B-CustomVoice config
TTS_BOS_ID = 151672
TTS_EOS_ID = 151673
TTS_PAD_ID = 151671

CODEC_THINK_ID = 2154
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157
CODEC_PAD_ID = 2148
CODEC_BOS_ID = 2149
CODEC_EOS_TOKEN_ID = 2150

SPEAKER_IDS = {
    "ono_anna": 2873,
    "vivian": 3065,
    "serena": 3066,
    "uncle_fu": 3010,
    "ryan": 3061,
    "aiden": 2861,
    "sohee": 2864,
    "eric": 2875,
    "dylan": 2878,
}

LANGUAGE_IDS = {
    "japanese": 2058,
    "chinese": 2055,
    "english": 2050,
    "korean": 2064,
    "french": 2061,
    "german": 2053,
    "italian": 2070,
    "portuguese": 2071,
    "spanish": 2054,
    "russian": 2069,
}


class MLXTextEncoder:
    """
    PyTorch-free text encoder for Qwen3-TTS.

    Tokenizes text, computes embeddings via MLX Talker weights,
    and assembles the input tensor structure matching PyTorch's
    modeling_qwen3_tts.py:2063-2273 (non-streaming mode).
    """

    def __init__(self, talker, model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                 weights_dir: Optional[str] = None):
        """
        Args:
            talker: MLX Qwen3TTSTalkerMLX instance (weights already loaded)
            model_id: HuggingFace model ID for tokenizer
            weights_dir: Directory containing weight files
        """
        self.talker = talker
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load text_projection biases (not in Talker weights — separate file)
        if weights_dir is None:
            from pathlib import Path
            weights_dir = str(Path(__file__).parent.parent)
        bias_path = f"{weights_dir}/text_projection_bias.npz"
        try:
            biases = dict(mx.load(bias_path))
            self._fc1_bias = biases["fc1_bias"]
            self._fc2_bias = biases["fc2_bias"]
        except FileNotFoundError:
            print(f"Warning: {bias_path} not found. text_projection bias will be zero.")
            self._fc1_bias = mx.zeros((2048,))
            self._fc2_bias = mx.zeros((1024,))

        # Pre-compute special token embeddings
        self._tts_bos_embed = None
        self._tts_eos_embed = None
        self._tts_pad_embed = None

    def _text_project(self, token_ids: mx.array) -> mx.array:
        """
        text_embedding → text_projection (2048 → 1024) with bias.

        Matches PyTorch Qwen3TTSTalkerResizeMLP:
            FC1(2048→2048, bias=True) → SiLU → FC2(2048→1024, bias=True)

        Args:
            token_ids: (1, seq_len) int32
        Returns:
            (1, seq_len, 1024)
        """
        embeds = self.talker.text_embedding(token_ids)  # (1, seq, 2048)
        projected = self.talker.text_projection_fc1(embeds) + self._fc1_bias
        projected = nn.silu(projected)
        projected = self.talker.text_projection_fc2(projected) + self._fc2_bias
        return projected  # (1, seq, 1024)

    def _codec_embed(self, token_ids: mx.array) -> mx.array:
        """
        codec_embedding lookup.

        Args:
            token_ids: (1, seq_len) int32
        Returns:
            (1, seq_len, 1024)
        """
        return self.talker.codec_embedding(token_ids)

    def _get_special_embeds(self):
        """Pre-compute tts_bos/eos/pad embeddings (cached)."""
        if self._tts_bos_embed is None:
            special_ids = mx.array([[TTS_BOS_ID, TTS_EOS_ID, TTS_PAD_ID]])
            special_embeds = self._text_project(special_ids)  # (1, 3, 1024)
            mx.eval(special_embeds)
            self._tts_bos_embed = special_embeds[:, 0:1, :]
            self._tts_eos_embed = special_embeds[:, 1:2, :]
            self._tts_pad_embed = special_embeds[:, 2:3, :]
        return self._tts_bos_embed, self._tts_eos_embed, self._tts_pad_embed

    def encode(
        self,
        text: str,
        speaker: str = "ono_anna",
        language: str = "japanese",
        instruct: str = "",
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Encode text into Talker input tensors (non-streaming mode).

        Replicates modeling_qwen3_tts.py:2063-2273 in pure MLX.

        Args:
            text: Text to synthesize
            speaker: Speaker preset name
            language: Language name
            instruct: Optional style instruction

        Returns:
            Tuple of:
            - inputs_embeds: (1, seq_len, 1024) — Talker prefill input
            - trailing_text_hidden: (1, 1, 1024) — tts_pad_embed (non-streaming)
            - tts_pad_embed: (1, 1, 1024) — padding embedding
        """
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_special_embeds()

        # ─── 1. Tokenize ───
        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        token_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]
        token_ids = mx.array(token_ids)  # (1, seq_len)

        # ─── 2. Role embedding (first 3 tokens: <|im_start|> assistant \n) ───
        role_embed = self._text_project(token_ids[:, :3])  # (1, 3, 1024)

        # ─── 3. Codec prefix ───
        language_id = LANGUAGE_IDS.get(language.lower(), LANGUAGE_IDS["japanese"])
        codec_prefix_ids = [CODEC_THINK_ID, CODEC_THINK_BOS_ID, language_id, CODEC_THINK_EOS_ID]

        speaker_id = SPEAKER_IDS.get(speaker.lower())
        if speaker_id is not None:
            codec_prefix_ids.append(speaker_id)

        codec_prefix_ids.extend([CODEC_PAD_ID, CODEC_BOS_ID])
        codec_prefix = self._codec_embed(mx.array([codec_prefix_ids]))  # (1, 7, 1024)

        # ─── 4. Build _talker_input_embed ───
        # tts_pad * (codec_len - 2) + tts_bos, then add codec_prefix[:-1]
        n_codec = codec_prefix.shape[1]
        pad_expanded = mx.broadcast_to(tts_pad_embed, (1, n_codec - 2, 1024))
        _embed = mx.concatenate([pad_expanded, tts_bos_embed], axis=1)  # (1, n_codec-1, 1024)
        _embed = _embed + codec_prefix[:, :-1, :]

        # ─── 5. Non-streaming: full text expansion ───
        # text_body = text_project(token_ids[3:-5]) + tts_eos
        text_body = self._text_project(token_ids[:, 3:-5])  # (1, text_len, 1024)
        text_with_eos = mx.concatenate([text_body, tts_eos_embed], axis=1)  # (1, text_len+1, 1024)

        # codec_pad repeated for text_len+1
        n_text_eos = text_with_eos.shape[1]
        codec_pad_repeated = self._codec_embed(
            mx.array([[CODEC_PAD_ID] * n_text_eos])
        )  # (1, text_len+1, 1024)
        text_block = text_with_eos + codec_pad_repeated

        # Final: tts_pad + codec_bos
        codec_bos_embed = self._codec_embed(mx.array([[CODEC_BOS_ID]]))  # (1, 1, 1024)
        final_token = tts_pad_embed + codec_bos_embed

        # ─── 6. Assemble ───
        parts = []

        # Optional instruct
        if instruct:
            instruct_prompt = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            instruct_ids = self.tokenizer(instruct_prompt, return_tensors="np")["input_ids"]
            instruct_ids = mx.array(instruct_ids)
            instruct_embed = self._text_project(instruct_ids)
            parts.append(instruct_embed)

        parts.extend([role_embed, _embed, text_block, final_token])
        inputs_embeds = mx.concatenate(parts, axis=1)
        mx.eval(inputs_embeds)

        # Non-streaming: trailing_text_hidden = tts_pad_embed
        trailing_text_hidden = tts_pad_embed

        return inputs_embeds, trailing_text_hidden, tts_pad_embed


# ─── Tests ───

def test_text_encoder():
    """Test MLXTextEncoder output shapes."""
    import numpy as np
    from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig

    print("=" * 50)
    print("MLX TextEncoder Test")
    print("=" * 50)

    # Load talker with real weights
    config = TalkerConfig()
    talker = Qwen3TTSTalkerMLX(config)
    from pathlib import Path
    weights_dir = Path(__file__).parent.parent
    weights = dict(np.load(str(weights_dir / 'talker_weights_mlx.npz')))

    talker.text_embedding.weight = mx.array(weights['text_embedding.weight'])
    talker.codec_embedding.weight = mx.array(weights['codec_embedding.weight'])
    talker.text_projection_fc1.weight = mx.array(weights['text_projection_fc1.weight'])
    talker.text_projection_fc2.weight = mx.array(weights['text_projection_fc2.weight'])
    for i, layer in enumerate(talker.layers):
        prefix = f'layers.{i}'
        layer.self_attn.q_proj.weight = mx.array(weights[f'{prefix}.self_attn.q_proj.weight'])
        layer.self_attn.k_proj.weight = mx.array(weights[f'{prefix}.self_attn.k_proj.weight'])
        layer.self_attn.v_proj.weight = mx.array(weights[f'{prefix}.self_attn.v_proj.weight'])
        layer.self_attn.o_proj.weight = mx.array(weights[f'{prefix}.self_attn.o_proj.weight'])
        layer.self_attn.q_norm.weight = mx.array(weights[f'{prefix}.self_attn.q_norm.weight'])
        layer.self_attn.k_norm.weight = mx.array(weights[f'{prefix}.self_attn.k_norm.weight'])
        layer.mlp.gate_proj.weight = mx.array(weights[f'{prefix}.mlp.gate_proj.weight'])
        layer.mlp.up_proj.weight = mx.array(weights[f'{prefix}.mlp.up_proj.weight'])
        layer.mlp.down_proj.weight = mx.array(weights[f'{prefix}.mlp.down_proj.weight'])
        layer.input_layernorm.weight = mx.array(weights[f'{prefix}.input_layernorm.weight'])
        layer.post_attention_layernorm.weight = mx.array(weights[f'{prefix}.post_attention_layernorm.weight'])
    talker.norm.weight = mx.array(weights['norm.weight'])

    encoder = MLXTextEncoder(talker)

    # Test 1: Basic encode
    print("\nTest 1: Basic encode")
    embeds, trailing, pad = encoder.encode("こんにちは。")
    mx.eval(embeds, trailing, pad)
    print(f"  inputs_embeds: {embeds.shape}")
    print(f"  trailing_text_hidden: {trailing.shape}")
    print(f"  tts_pad_embed: {pad.shape}")
    assert embeds.shape[0] == 1
    assert embeds.shape[2] == 1024
    assert trailing.shape == (1, 1, 1024)
    assert pad.shape == (1, 1, 1024)

    # Test 2: With instruct
    print("\nTest 2: With instruct")
    embeds2, _, _ = encoder.encode(
        "今日は良い天気ね。",
        instruct="A cute young Japanese anime girl voice"
    )
    mx.eval(embeds2)
    print(f"  inputs_embeds (with instruct): {embeds2.shape}")
    assert embeds2.shape[1] > embeds.shape[1]  # instruct adds tokens

    # Test 3: Different speakers
    print("\nTest 3: Different speakers")
    for spk in ["ono_anna", "vivian", "ryan"]:
        e, _, _ = encoder.encode("test", speaker=spk)
        mx.eval(e)
        print(f"  {spk}: {e.shape}")

    print("\n✅ MLXTextEncoder tests passed!")


if __name__ == "__main__":
    test_text_encoder()
