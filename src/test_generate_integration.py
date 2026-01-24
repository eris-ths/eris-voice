#!/usr/bin/env python3
"""
Test MLX Generate Loop integration with PyTorch model.

This test extracts trailing_text_hidden from PyTorch and uses it with MLX Generate Loop.
"""

import os
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

import mlx.core as mx
import numpy as np
import torch
torch.set_num_threads(8)

from mlx_generate import MLXGenerateLoop, GenerateConfig
from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig
from mlx_code_predictor import Qwen3TTSCodePredictorMLX, CodePredictorConfig, load_code_predictor_weights
from mlx_decoder_v2 import Qwen3TTSDecoderMLX
from mlx_quantizer import SplitResidualVectorQuantizerMLX


def extract_pytorch_inputs(text: str, speaker: str = "ono_anna"):
    """
    Extract inputs from PyTorch model for MLX Generate Loop.

    Uses monkey-patching to intercept internal values from the generate() call.

    Returns:
        - initial_embeds: Initial input embeddings (MLX array)
        - trailing_text_hidden: Text embeddings for each step (MLX array)
        - tts_pad_embed: Padding embedding (MLX array)
        - pt_model: PyTorch model for decoder access
    """
    from qwen_tts import Qwen3TTSModel

    # Load PyTorch model
    print("Loading PyTorch model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    # Capture internal values by patching talker.generate()
    captured = {}
    original_generate = model.model.talker.generate

    def patched_generate(inputs_embeds, trailing_text_hidden, tts_pad_embed, **kwargs):
        captured['inputs_embeds'] = inputs_embeds.detach().clone()
        captured['trailing_text_hidden'] = trailing_text_hidden.detach().clone()
        captured['tts_pad_embed'] = tts_pad_embed.detach().clone()
        # Stop early - we just need the inputs
        raise StopIteration("Captured inputs")

    model.model.talker.generate = patched_generate

    try:
        with torch.no_grad():
            model.generate_custom_voice(
                text=text,
                language="Japanese",
                speaker=speaker,
            )
    except StopIteration:
        pass  # Expected - we stopped early

    # Restore original
    model.model.talker.generate = original_generate

    # Convert to numpy then MLX
    initial_embed_np = captured['inputs_embeds'].cpu().float().numpy()
    trailing_text_hidden_np = captured['trailing_text_hidden'].cpu().float().numpy()
    tts_pad_embed_np = captured['tts_pad_embed'].cpu().float().numpy()

    print(f"  initial_embed shape: {initial_embed_np.shape}")
    print(f"  trailing_text_hidden shape: {trailing_text_hidden_np.shape}")
    print(f"  tts_pad_embed shape: {tts_pad_embed_np.shape}")

    return (
        mx.array(initial_embed_np),
        mx.array(trailing_text_hidden_np),
        mx.array(tts_pad_embed_np),
        model,  # Return model for decoder access
    )


def load_mlx_models(weights_dir: Path):
    """Load MLX models."""
    print("\nLoading MLX models...")

    # Talker
    print("  Loading Talker...")
    talker_config = TalkerConfig()
    talker = Qwen3TTSTalkerMLX(talker_config)

    talker_weights = dict(np.load(str(weights_dir / 'talker_weights_mlx.npz')))
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
    codec_head_weight = mx.array(talker_weights['codec_head.weight'])

    # CodePredictor
    print("  Loading CodePredictor...")
    cp_config = CodePredictorConfig()
    code_predictor = Qwen3TTSCodePredictorMLX(cp_config)
    code_predictor = load_code_predictor_weights(code_predictor, str(weights_dir / 'code_predictor_weights_mlx.npz'))

    # Quantizer
    print("  Loading Quantizer...")
    mlx_quantizer = SplitResidualVectorQuantizerMLX(
        n_q_semantic=1,
        total_quantizers=16,
        codebook_size=2048,
        input_dim=512,
        codebook_dim=256,
    )
    quantizer_weights = dict(mx.load(str(weights_dir / "quantizer_weights_mlx.npz")))
    mlx_quantizer.load_weights(quantizer_weights)

    # Decoder
    print("  Loading Decoder...")
    mlx_decoder = Qwen3TTSDecoderMLX()
    decoder_weights = dict(mx.load(str(weights_dir / "decoder_weights_mlx.npz")))
    mlx_decoder.load_weights(decoder_weights)

    return talker, code_predictor, codec_head_weight, mlx_quantizer, mlx_decoder


def decode_codes_to_audio(codes: mx.array, pt_model, mlx_quantizer, mlx_decoder, eos_token_id: int = 2150):
    """Decode codec codes to audio using hybrid PyTorch + MLX pipeline."""
    # Find EOS and truncate
    codes_np = np.array(codes)
    codebook_0 = codes_np[0, :, 0]
    eos_positions = np.where(codebook_0 == eos_token_id)[0]

    if len(eos_positions) > 0:
        eos_pos = eos_positions[0]
        codes_np = codes_np[:, :eos_pos, :]

    if codes_np.shape[1] == 0:
        return np.zeros(2400, dtype=np.float32)

    # Reshape: (1, num_steps, 16) → (1, 16, num_steps)
    codes = mx.array(np.transpose(codes_np, (0, 2, 1)))

    print(f"  Decoding {codes.shape[2]} steps...")

    # MLX Quantizer decode
    hidden_mlx = mlx_quantizer.decode(codes)
    mx.eval(hidden_mlx)
    hidden_np = np.array(hidden_mlx)

    # PyTorch pre-processing
    pt_decoder = pt_model.model.speech_tokenizer.model.decoder

    with torch.no_grad():
        hidden = torch.from_numpy(hidden_np).to(dtype=torch.bfloat16)

        # Pre-conv
        hidden = pt_decoder.pre_conv(hidden).transpose(1, 2)

        # Pre-transformer
        if pt_decoder.pre_transformer is not None:
            hidden = pt_decoder.pre_transformer(
                inputs_embeds=hidden
            ).last_hidden_state
            hidden = hidden.permute(0, 2, 1)

        # Upsample
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
    return wav_np[0, 0, :]


def test_generate_integration():
    """Test MLX Generate Loop with trailing_text_hidden."""
    print("=" * 60)
    print("MLX Generate Loop Integration Test")
    print("=" * 60)

    weights_dir = Path(__file__).parent.parent

    # Check weights exist
    required = ['talker_weights_mlx.npz', 'code_predictor_weights_mlx.npz',
                'quantizer_weights_mlx.npz', 'decoder_weights_mlx.npz']
    for f in required:
        if not (weights_dir / f).exists():
            print(f"⚠️ Missing weights: {f}")
            return False

    # Test text
    text = "こんにちは。"

    # Extract inputs from PyTorch
    initial_embeds, trailing_text_hidden, tts_pad_embed, pt_model = extract_pytorch_inputs(text)

    # Load MLX models
    talker, code_predictor, codec_head_weight, mlx_quantizer, mlx_decoder = load_mlx_models(weights_dir)

    # Create generator
    generator = MLXGenerateLoop(talker, code_predictor, codec_head_weight)

    # Warmup
    print("\nWarmup run...")
    warmup_config = GenerateConfig(max_new_tokens=2, do_sample=False)
    warmup_embeds = mx.random.normal((1, 4, 1024))
    _ = generator.generate(warmup_embeds, warmup_config)
    print("  Done!")

    # Generate with trailing_text_hidden
    print("\nGenerating with trailing_text_hidden...")
    config = GenerateConfig(
        max_new_tokens=100,  # Limit for testing
        temperature=0.9,
        top_p=1.0,
        do_sample=True,
        quality_mode="balanced",
    )

    start = time.time()
    codes = generator.generate(
        initial_embeds,
        config,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        debug=False,
    )
    mx.eval(codes)
    gen_time = time.time() - start

    print(f"\n  Generated {codes.shape[1]} steps in {gen_time:.2f}s")
    print(f"  Codes shape: {codes.shape}")
    print(f"  First step codes: {codes[0, 0, :].tolist()[:5]}...")

    # Decode to audio
    print("\nDecoding to audio...")
    audio = decode_codes_to_audio(codes, pt_model, mlx_quantizer, mlx_decoder)

    duration = len(audio) / 24000
    rtf = duration / gen_time if gen_time > 0 else 0

    print(f"\n  Audio duration: {duration:.2f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  RTF: {rtf:.2f}x")

    # Save audio for listening
    output_path = weights_dir / "test_integration_output.wav"
    import soundfile as sf
    sf.write(str(output_path), audio, 24000)
    print(f"\n  Saved to: {output_path}")

    print("\n✅ Integration test passed!")
    return True


if __name__ == "__main__":
    test_generate_integration()
