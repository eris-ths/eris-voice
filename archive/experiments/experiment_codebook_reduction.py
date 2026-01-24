#!/usr/bin/env python3
"""
Codebook Reduction Experiment

Tests the effect of reducing the number of acoustic codebooks (1-15)
on generation speed and audio quality.

Hypothesis: Later codebooks contribute less to perceptual quality,
so we can skip them for faster generation with acceptable quality loss.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import mlx.core as mx
import numpy as np

from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig
from mlx_code_predictor import Qwen3TTSCodePredictorMLX, CodePredictorConfig, load_code_predictor_weights
from mlx_generate import MLXGenerateLoop, GenerateConfig


def load_models():
    """Load all models with weights."""
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

    return talker, code_predictor, codec_head_weight


def run_speed_experiment(generator, num_codebooks_list=[15, 11, 7, 3, 1]):
    """
    Measure generation speed with different codebook counts.
    """
    print("\n" + "=" * 60)
    print("Speed Experiment: Codebook Reduction")
    print("=" * 60)

    # Fixed input for fair comparison
    batch_size = 1
    seq_len = 8
    num_steps = 10
    input_embeds = mx.random.normal((batch_size, seq_len, 1024))

    results = []

    for num_cb in num_codebooks_list:
        print(f"\n--- Testing {num_cb} acoustic codebooks ---")

        config = GenerateConfig(
            max_new_tokens=num_steps,
            do_sample=False,  # Greedy for reproducibility
            num_acoustic_codebooks=num_cb,
        )

        # Warmup
        _ = generator.generate(input_embeds, config, debug=False)

        # Timed run
        start = time.time()
        codes = generator.generate(input_embeds, config, debug=True)
        mx.eval(codes)
        elapsed = time.time() - start

        ms_per_step = elapsed / codes.shape[1] * 1000
        rtf = 83.3 / ms_per_step  # 12Hz TTS = 83.3ms/step for RTF 1.0

        results.append({
            'num_codebooks': num_cb,
            'ms_per_step': ms_per_step,
            'rtf': rtf,
            'total_time': elapsed,
        })

        print(f"  Total: {elapsed:.2f}s, {ms_per_step:.1f}ms/step, RTF {rtf:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Codebooks':<12} {'ms/step':<12} {'RTF':<10} {'Speedup':<10}")
    print("-" * 44)

    baseline_ms = results[0]['ms_per_step']
    for r in results:
        speedup = baseline_ms / r['ms_per_step']
        print(f"{r['num_codebooks']:<12} {r['ms_per_step']:<12.1f} {r['rtf']:<10.2f} {speedup:<10.2f}x")

    return results


def main():
    print("=" * 60)
    print("Codebook Reduction Experiment")
    print("=" * 60)
    print()
    print("Goal: Find minimum codebooks needed for acceptable quality")
    print("      while maximizing speed (targeting RTF > 1.0x)")
    print()

    # Check weights
    weights_dir = Path(__file__).parent.parent
    if not (weights_dir / 'talker_weights_mlx.npz').exists():
        print("ERROR: Weights not found. Run convert_talker_weights.py first.")
        return

    # Load models
    talker, code_predictor, codec_head_weight = load_models()
    generator = MLXGenerateLoop(talker, code_predictor, codec_head_weight)

    # Run speed experiment
    results = run_speed_experiment(generator, [15, 11, 7, 3, 1])

    print("\n" + "=" * 60)
    print("Next Step: Audio Quality Comparison")
    print("=" * 60)
    print("To compare audio quality, run with actual text input and decoder:")
    print("  python experiment_codebook_audio_quality.py")
    print()

    # Find configurations that achieve RTF > 1.0
    realtime_configs = [r for r in results if r['rtf'] >= 1.0]
    if realtime_configs:
        best = max(realtime_configs, key=lambda x: x['num_codebooks'])
        print(f"Best config for RTF >= 1.0: {best['num_codebooks']} codebooks")
        print(f"  -> {best['ms_per_step']:.1f}ms/step, RTF {best['rtf']:.2f}x")
    else:
        print("No configuration achieved RTF >= 1.0")
        print("Consider: mx.compile() or QuantizedLinear optimization")


if __name__ == "__main__":
    main()
