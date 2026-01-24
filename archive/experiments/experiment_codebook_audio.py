#!/usr/bin/env python3
"""
Codebook Reduction Audio Quality Experiment

Generates audio with different codebook counts to test quality vs speed tradeoff.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

import time
import numpy as np
import mlx.core as mx
import soundfile as sf
import torch
torch.set_num_threads(8)

from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig
from mlx_code_predictor import Qwen3TTSCodePredictorMLX, CodePredictorConfig, load_code_predictor_weights
from mlx_generate import MLXGenerateLoop, GenerateConfig
from mlx_decoder_v2 import Qwen3TTSDecoderMLX
from mlx_quantizer import SplitResidualVectorQuantizerMLX


def load_all_models():
    """Load all MLX models."""
    weights_dir = Path(__file__).parent.parent

    # Load Talker
    print("Loading Talker...")
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

    # Load CodePredictor
    print("Loading CodePredictor...")
    cp_config = CodePredictorConfig()
    code_predictor = Qwen3TTSCodePredictorMLX(cp_config)
    code_predictor = load_code_predictor_weights(code_predictor, str(weights_dir / 'code_predictor_weights_mlx.npz'))

    # Load Decoder
    print("Loading Decoder...")
    decoder = Qwen3TTSDecoderMLX()
    decoder_weights = dict(mx.load(str(weights_dir / "decoder_weights_mlx.npz")))
    decoder.load_weights(decoder_weights)

    # Load Quantizer
    print("Loading Quantizer...")
    quantizer = SplitResidualVectorQuantizerMLX(
        n_q_semantic=1,
        total_quantizers=16,
        codebook_size=2048,
        input_dim=512,
        codebook_dim=256,
    )
    quantizer_weights = dict(mx.load(str(weights_dir / "quantizer_weights_mlx.npz")))
    quantizer.load_weights(quantizer_weights)

    # Load PyTorch model for tokenization
    print("Loading PyTorch model (for tokenization)...")
    from qwen_tts import Qwen3TTSModel
    pt_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    return talker, code_predictor, codec_head_weight, decoder, quantizer, pt_model


def prepare_input_embeds(pt_model, text: str, speaker: str = "ono_anna", language: str = "Japanese"):
    """
    Use PyTorch model to prepare input embeddings from text.
    Returns MLX array ready for generation.
    """
    # Get tokenized input using PyTorch model's prepare method
    with torch.no_grad():
        # Use the internal preparation (simplified)
        prompt = f"[{speaker}][{language}]<|startofaudio|>{text}"

        tokenizer = pt_model.tokenizer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = inputs["input_ids"]

        # Get embeddings from talker's text embedding
        # We'll use the MLX talker's embedding
        return input_ids.numpy()


def decode_codes_to_audio(codes: mx.array, quantizer, decoder, pt_model) -> np.ndarray:
    """
    Decode codec codes to audio waveform.

    Args:
        codes: (batch, num_steps, 16) codec codes
        quantizer: MLX quantizer
        decoder: MLX decoder
        pt_model: PyTorch model for pre-processing

    Returns:
        Audio waveform as numpy array
    """
    # Reshape codes for quantizer: (batch, 16, length)
    codes = mx.transpose(codes, (0, 2, 1))

    # Quantizer decode
    hidden = quantizer.decode(codes)
    mx.eval(hidden)

    # Convert to PyTorch for pre-processing
    hidden_np = np.array(hidden)
    hidden_pt = torch.from_numpy(hidden_np).to(dtype=torch.bfloat16)

    pt_decoder = pt_model.model.speech_tokenizer.model.decoder

    with torch.no_grad():
        # PyTorch pre-processing
        hidden_pt = pt_decoder.pre_conv(hidden_pt).transpose(1, 2)
        if pt_decoder.pre_transformer is not None:
            hidden_pt = pt_decoder.pre_transformer(
                inputs_embeds=hidden_pt
            ).last_hidden_state
            hidden_pt = hidden_pt.permute(0, 2, 1)

        for blocks in pt_decoder.upsample:
            for block in blocks:
                hidden_pt = block(hidden_pt)

    # MLX decoder
    hidden_np = hidden_pt.detach().cpu().float().numpy()
    hidden_mlx = mx.array(hidden_np)

    wav_mlx = decoder.decoder_conv0(hidden_mlx)
    for block in decoder.decoder_blocks:
        wav_mlx = block(wav_mlx)
    wav_mlx = decoder.final_act(wav_mlx)
    wav_mlx = decoder.final_conv(wav_mlx)
    wav_mlx = mx.clip(wav_mlx, -1, 1)
    mx.eval(wav_mlx)

    wav_np = np.array(wav_mlx)
    return wav_np[0, 0, :]


def run_audio_experiment():
    """Run audio quality experiment with different codebook counts."""
    print("=" * 60)
    print("Audio Quality Experiment: Codebook Reduction")
    print("=" * 60)
    print()

    # Load models
    talker, code_predictor, codec_head_weight, decoder, quantizer, pt_model = load_all_models()
    generator = MLXGenerateLoop(talker, code_predictor, codec_head_weight)

    # Test configurations
    codebook_counts = [15, 11, 7, 3]
    test_text = "こんにちは、私はエリスです。"

    # Prepare input (use random embeddings for now - real text requires more integration)
    # For a proper test, we need to integrate with the tokenizer properly
    # For now, let's generate with random input to test the pipeline

    print(f"\nTest text: {test_text}")
    print("Note: Using random input for pipeline test")
    print()

    batch_size = 1
    seq_len = 20  # Simulated prompt length
    num_steps = 50  # Generate ~4 seconds of audio (12Hz * 4s ≈ 48 steps)

    input_embeds = mx.random.normal((batch_size, seq_len, 1024))

    output_dir = Path(__file__).parent.parent / "audio_experiments"
    output_dir.mkdir(exist_ok=True)

    results = []

    for num_cb in codebook_counts:
        print(f"\n--- Testing {num_cb} acoustic codebooks ---")

        config = GenerateConfig(
            max_new_tokens=num_steps,
            do_sample=True,
            temperature=0.9,
            num_acoustic_codebooks=num_cb,
        )

        # Warmup
        _ = generator.generate(input_embeds, config, debug=False)

        # Timed generation
        start = time.time()
        codes = generator.generate(input_embeds, config, debug=False)
        mx.eval(codes)
        gen_time = time.time() - start

        print(f"  Generated {codes.shape[1]} steps in {gen_time:.2f}s")

        # Decode to audio
        try:
            audio = decode_codes_to_audio(codes, quantizer, decoder, pt_model)

            # Save audio
            output_path = output_dir / f"test_{num_cb}cb.wav"
            sf.write(str(output_path), audio, 24000)

            duration = len(audio) / 24000
            rtf = duration / gen_time

            print(f"  Audio duration: {duration:.2f}s")
            print(f"  RTF: {rtf:.2f}x")
            print(f"  Saved: {output_path}")

            results.append({
                'num_codebooks': num_cb,
                'gen_time': gen_time,
                'duration': duration,
                'rtf': rtf,
                'path': str(output_path),
            })

        except Exception as e:
            print(f"  ERROR decoding: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Codebooks':<12} {'Gen Time':<12} {'Duration':<12} {'RTF':<10}")
    print("-" * 46)
    for r in results:
        print(f"{r['num_codebooks']:<12} {r['gen_time']:<12.2f} {r['duration']:<12.2f} {r['rtf']:<10.2f}x")

    print(f"\nAudio files saved to: {output_dir}")
    print("Listen and compare quality!")


if __name__ == "__main__":
    run_audio_experiment()
