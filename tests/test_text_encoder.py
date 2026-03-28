"""
Tests for MLX TextEncoder
"""

import mlx.core as mx
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig
from mlx_text_encoder import MLXTextEncoder


def _load_talker():
    """Load talker with real weights."""
    config = TalkerConfig()
    talker = Qwen3TTSTalkerMLX(config)
    weights_dir = Path(__file__).parent.parent
    weights = dict(np.load(str(weights_dir / 'talker_weights_mlx.npz')))

    talker.text_embedding.weight = mx.array(weights['text_embedding.weight'])
    talker.codec_embedding.weight = mx.array(weights['codec_embedding.weight'])
    talker.text_projection_fc1.weight = mx.array(weights['text_projection_fc1.weight'])
    talker.text_projection_fc2.weight = mx.array(weights['text_projection_fc2.weight'])
    talker.norm.weight = mx.array(weights['norm.weight'])

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

    return talker


def test_basic_encode():
    """Test basic text encoding shapes."""
    talker = _load_talker()
    encoder = MLXTextEncoder(talker)

    embeds, trailing, pad = encoder.encode("こんにちは。")
    mx.eval(embeds, trailing, pad)

    assert embeds.shape[0] == 1
    assert embeds.shape[2] == 1024
    assert trailing.shape == (1, 1, 1024)
    assert pad.shape == (1, 1, 1024)


def test_instruct_increases_length():
    """Test instruct adds tokens to inputs_embeds."""
    talker = _load_talker()
    encoder = MLXTextEncoder(talker)

    e1, _, _ = encoder.encode("テスト")
    e2, _, _ = encoder.encode("テスト", instruct="A cute anime girl voice")
    mx.eval(e1, e2)

    assert e2.shape[1] > e1.shape[1]


def test_different_speakers():
    """Test different speakers produce different embeddings."""
    talker = _load_talker()
    encoder = MLXTextEncoder(talker)

    e_anna, _, _ = encoder.encode("テスト", speaker="ono_anna")
    e_ryan, _, _ = encoder.encode("テスト", speaker="ryan")
    mx.eval(e_anna, e_ryan)

    # Different speakers → different embeddings
    diff = mx.abs(e_anna - e_ryan).max()
    mx.eval(diff)
    assert float(diff) > 0.1


def test_special_embeds_cached():
    """Test special embeddings are pre-computed and cached."""
    talker = _load_talker()
    encoder = MLXTextEncoder(talker)

    # First call computes
    encoder.encode("a")
    bos1 = encoder._tts_bos_embed

    # Second call should use cache
    encoder.encode("b")
    bos2 = encoder._tts_bos_embed

    assert bos1 is bos2  # Same object (cached)


def test_trailing_is_pad():
    """Test trailing_text_hidden equals tts_pad_embed (non-streaming mode)."""
    talker = _load_talker()
    encoder = MLXTextEncoder(talker)

    _, trailing, pad = encoder.encode("テスト")
    mx.eval(trailing, pad)

    diff = float(mx.abs(trailing - pad).max())
    assert diff == 0.0


if __name__ == "__main__":
    test_basic_encode()
    test_instruct_increases_length()
    test_different_speakers()
    test_special_embeds_cached()
    test_trailing_is_pad()
    print("✅ All TextEncoder tests passed!")
