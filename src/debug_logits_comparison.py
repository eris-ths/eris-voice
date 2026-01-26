#!/usr/bin/env python3
"""
Debug script to compare PyTorch vs MLX logits step-by-step.

This will help identify exactly where the divergence happens.
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


def extract_pytorch_step_data(text: str = "こんにちは。", speaker: str = "ono_anna", num_steps: int = 3):
    """
    Extract step-by-step data from PyTorch for comparison.

    Returns:
        Dict with initial inputs and per-step data (logits, codes, hidden states)
    """
    from qwen_tts import Qwen3TTSModel

    print("Loading PyTorch model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    # Capture internal values by patching
    captured = {
        'initial': {},
        'steps': [],
    }
    step_count = [0]

    # Store original generate
    original_talker_generate = model.model.talker.generate

    def patched_talker_generate(inputs_embeds, trailing_text_hidden, tts_pad_embed, **kwargs):
        """Capture inputs and run limited generation."""
        captured['initial']['inputs_embeds'] = inputs_embeds.detach().clone()
        captured['initial']['trailing_text_hidden'] = trailing_text_hidden.detach().clone()
        captured['initial']['tts_pad_embed'] = tts_pad_embed.detach().clone()

        # Run actual generation but capture each step
        # We need to go deeper - patch the talker model itself
        talker = model.model.talker.model
        code_predictor = model.model.talker.code_predictor
        codec_head = model.model.talker.codec_head

        past_key_values = None
        current_input = inputs_embeds
        generation_step = 0

        all_codes = []

        while generation_step < num_steps:
            step_data = {'step': generation_step}

            # Forward through talker
            with torch.no_grad():
                outputs = talker(
                    inputs_embeds=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                hidden_states = outputs.last_hidden_state
                past_key_values = outputs.past_key_values

            # Get logits from codec_head
            last_hidden = hidden_states[:, -1:, :]
            logits_0 = codec_head(last_hidden)
            logits_0 = logits_0[:, -1, :]  # (batch, vocab)

            # Store step data
            step_data['hidden'] = last_hidden.detach().clone()
            step_data['logits_0'] = logits_0.detach().clone()

            # Greedy sample
            code_0 = torch.argmax(logits_0, dim=-1)
            step_data['code_0'] = code_0.item()

            print(f"  PT Step {generation_step}: code_0={code_0.item()}, logits_0 top5={logits_0[0].topk(5).indices.tolist()}")

            # Generate remaining codebooks with code_predictor
            cp_input = last_hidden
            codes_step = [code_0.unsqueeze(-1)]

            for cb_idx in range(15):
                with torch.no_grad():
                    cp_output = code_predictor.model(inputs_embeds=cp_input)
                    cp_hidden = cp_output.last_hidden_state[:, -1:, :]
                    cp_logits = code_predictor.lm_head[cb_idx](cp_hidden)
                    cp_logits = cp_logits[:, -1, :]
                    cp_code = torch.argmax(cp_logits, dim=-1)
                    codes_step.append(cp_code.unsqueeze(-1))

                    # Next input is embedding of this code
                    cp_input = code_predictor.get_input_embeddings()[cb_idx](cp_code.unsqueeze(-1))

            codes_step = torch.cat(codes_step, dim=-1)  # (batch, 16)
            step_data['codes'] = codes_step.detach().clone()
            all_codes.append(codes_step)

            captured['steps'].append(step_data)

            # Prepare next input (matching Qwen3-TTS behavior)
            # inputs_embeds = last_hidden + sum(cp_embeddings) + trailing_text_hidden[step]
            codec_hiddens = [last_hidden]
            for i in range(15):
                embed = code_predictor.get_input_embeddings()[i](codes_step[:, i+1:i+2])
                codec_hiddens.append(embed)
            codec_hiddens = torch.cat(codec_hiddens, dim=1)
            next_input = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                next_input = next_input + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                next_input = next_input + tts_pad_embed

            current_input = next_input
            generation_step += 1

        # Stack all codes
        all_codes = torch.cat([c.unsqueeze(1) for c in all_codes], dim=1)

        # Return something that won't crash
        class FakeResult:
            sequences = all_codes
        return FakeResult()

    model.model.talker.generate = patched_talker_generate

    try:
        with torch.no_grad():
            model.generate_custom_voice(
                text=text,
                language="Japanese",
                speaker=speaker,
            )
    except Exception as e:
        print(f"  (Expected exception: {type(e).__name__})")

    model.model.talker.generate = original_talker_generate

    return captured, model


def run_mlx_steps(initial_embeds, trailing_text_hidden, tts_pad_embed, weights_dir, num_steps=3):
    """
    Run MLX generation step-by-step and capture data.
    """
    print("\nLoading MLX models...")

    # Load Talker
    talker_config = TalkerConfig()
    talker = Qwen3TTSTalkerMLX(talker_config)
    talker_weights = dict(np.load(str(weights_dir / 'talker_weights_mlx.npz')))

    # Load weights
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
    cp_config = CodePredictorConfig()
    code_predictor = Qwen3TTSCodePredictorMLX(cp_config)
    code_predictor = load_code_predictor_weights(code_predictor, str(weights_dir / 'code_predictor_weights_mlx.npz'))

    # Create generator
    generator = MLXGenerateLoop(talker, code_predictor, codec_head_weight)

    # Convert inputs to MLX
    initial_embeds_mlx = mx.array(initial_embeds.cpu().float().numpy())
    trailing_text_hidden_mlx = mx.array(trailing_text_hidden.cpu().float().numpy())
    tts_pad_embed_mlx = mx.array(tts_pad_embed.cpu().float().numpy())

    print(f"  initial_embeds: {initial_embeds_mlx.shape}")
    print(f"  trailing_text_hidden: {trailing_text_hidden_mlx.shape}")

    # Run step-by-step
    generator.reset_caches()

    text_len = trailing_text_hidden_mlx.shape[1]
    current_embeds = initial_embeds_mlx
    config = GenerateConfig(max_new_tokens=num_steps, do_sample=False)

    mlx_steps = []

    for step in range(num_steps):
        step_data = {'step': step}

        # Reset CodePredictor cache for each step
        generator.cp_cache.reset()

        # Talker forward
        hidden = generator.talker(current_embeds, cache=generator.talker_cache.get_caches())
        mx.eval(hidden)

        last_hidden = hidden[:, -1:, :]

        # Codec head
        logits_0 = generator.codec_head(last_hidden)
        logits_0 = logits_0[:, -1, :]  # (batch, vocab)
        mx.eval(logits_0)

        step_data['hidden'] = np.array(last_hidden)
        step_data['logits_0'] = np.array(logits_0)

        # Greedy sample
        code_0 = mx.argmax(logits_0, axis=-1)
        mx.eval(code_0)
        step_data['code_0'] = int(code_0[0])

        # Get top-5 indices
        top5_indices = mx.argsort(logits_0[0])[-5:][::-1]
        mx.eval(top5_indices)
        print(f"  MLX Step {step}: code_0={int(code_0[0])}, logits_0 top5={top5_indices.tolist()}")

        # Generate remaining codebooks
        cp_input = last_hidden
        codes = [code_0[:, None]]  # (batch, 1)

        for cb_idx in range(15):
            cp_hidden = generator.code_predictor(cp_input, cache=generator.cp_cache.get_caches())
            cp_logits = generator.code_predictor.get_logits(cp_hidden[:, -1:, :], cb_idx)
            cp_logits = cp_logits[:, -1, :]
            cp_code = mx.argmax(cp_logits, axis=-1)
            codes.append(cp_code[:, None])

            # Next input
            cp_input = generator.code_predictor.embed_codes(cp_code[:, None], cb_idx)

        all_codes = mx.concatenate(codes, axis=1)
        mx.eval(all_codes)
        step_data['codes'] = np.array(all_codes)

        mlx_steps.append(step_data)

        # Prepare next input
        next_embed = last_hidden
        for i in range(15):
            cb_embed = generator.code_predictor.embed_codes(all_codes[:, i+1:i+2], i)
            next_embed = next_embed + cb_embed

        if step < text_len:
            next_embed = next_embed + trailing_text_hidden_mlx[:, step:step+1, :]
        else:
            next_embed = next_embed + tts_pad_embed_mlx

        mx.eval(next_embed)
        current_embeds = next_embed

    return mlx_steps


def compare_steps(pt_steps, mlx_steps):
    """Compare PyTorch vs MLX step by step."""
    print("\n" + "=" * 60)
    print("Step-by-Step Comparison")
    print("=" * 60)

    for i, (pt, mlx) in enumerate(zip(pt_steps, mlx_steps)):
        print(f"\n--- Step {i} ---")

        # Compare hidden states
        pt_hidden = pt['hidden'].cpu().float().numpy()
        mlx_hidden = mlx['hidden']

        hidden_diff = np.abs(pt_hidden - mlx_hidden).max()
        hidden_mean_diff = np.abs(pt_hidden - mlx_hidden).mean()
        print(f"  Hidden: max_diff={hidden_diff:.6f}, mean_diff={hidden_mean_diff:.6f}")
        print(f"    PT[:5]: {pt_hidden[0, 0, :5]}")
        print(f"    MLX[:5]: {mlx_hidden[0, 0, :5]}")

        # Compare logits
        pt_logits = pt['logits_0'].cpu().float().numpy()
        mlx_logits = mlx['logits_0']

        logits_diff = np.abs(pt_logits - mlx_logits).max()
        logits_mean_diff = np.abs(pt_logits - mlx_logits).mean()
        print(f"  Logits: max_diff={logits_diff:.6f}, mean_diff={logits_mean_diff:.6f}")

        # Top-10 logits comparison
        pt_top10_idx = np.argsort(pt_logits[0])[-10:][::-1]
        mlx_top10_idx = np.argsort(mlx_logits[0])[-10:][::-1]
        print(f"    PT top10 indices: {pt_top10_idx.tolist()}")
        print(f"    MLX top10 indices: {mlx_top10_idx.tolist()}")
        print(f"    PT top10 values: {[f'{pt_logits[0, idx]:.2f}' for idx in pt_top10_idx]}")
        print(f"    MLX top10 values: {[f'{mlx_logits[0, idx]:.2f}' for idx in mlx_top10_idx]}")

        # Compare codes
        print(f"  Code 0: PT={pt['code_0']}, MLX={mlx['code_0']}, match={pt['code_0'] == mlx['code_0']}")

        # Compare full codes
        pt_codes = pt['codes'].cpu().numpy() if hasattr(pt['codes'], 'cpu') else pt['codes']
        mlx_codes = mlx['codes']
        codes_match = np.array_equal(pt_codes, mlx_codes)
        print(f"  All codes match: {codes_match}")
        if not codes_match:
            print(f"    PT codes: {pt_codes[0, :5].tolist()}...")
            print(f"    MLX codes: {mlx_codes[0, :5].tolist()}...")


def main():
    print("=" * 60)
    print("PyTorch vs MLX Logits Comparison Debug")
    print("=" * 60)
    print()

    weights_dir = Path(__file__).parent.parent

    # Check weights exist
    required = ['talker_weights_mlx.npz', 'code_predictor_weights_mlx.npz']
    for f in required:
        if not (weights_dir / f).exists():
            print(f"Missing weights: {f}")
            return

    num_steps = 3

    # Extract PyTorch data
    print("\n--- PyTorch Generation ---")
    pt_data, pt_model = extract_pytorch_step_data(num_steps=num_steps)

    # Run MLX
    print("\n--- MLX Generation ---")
    mlx_steps = run_mlx_steps(
        pt_data['initial']['inputs_embeds'],
        pt_data['initial']['trailing_text_hidden'],
        pt_data['initial']['tts_pad_embed'],
        weights_dir,
        num_steps=num_steps,
    )

    # Compare
    compare_steps(pt_data['steps'], mlx_steps)

    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
