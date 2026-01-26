#!/usr/bin/env python3
"""
Compare MLX vs PyTorch code generation step-by-step.

Uses greedy sampling (do_sample=False) for deterministic comparison.
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


def run_pytorch_generation(text: str, speaker: str = "ono_anna", num_steps: int = 5):
    """Run PyTorch generation and capture step-by-step codes."""
    from qwen_tts import Qwen3TTSModel

    print("Loading PyTorch model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    # Capture internal values
    captured = {
        'initial': {},
        'codes': [],
    }
    original_generate = model.model.talker.generate

    def patched_generate(inputs_embeds, trailing_text_hidden, tts_pad_embed, **kwargs):
        captured['initial']['inputs_embeds'] = inputs_embeds.detach().clone()
        captured['initial']['trailing_text_hidden'] = trailing_text_hidden.detach().clone()
        captured['initial']['tts_pad_embed'] = tts_pad_embed.detach().clone()

        # Run actual generation but limit steps
        talker = model.model.talker.model
        code_predictor = model.model.talker.code_predictor
        codec_head = model.model.talker.codec_head

        past_key_values = None
        current_input = inputs_embeds
        text_len = trailing_text_hidden.shape[1]

        for step in range(num_steps):
            # Talker forward
            with torch.no_grad():
                outputs = talker(
                    inputs_embeds=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                hidden_states = outputs.last_hidden_state
                past_key_values = outputs.past_key_values

            last_hidden = hidden_states[:, -1:, :]
            logits_0 = codec_head(last_hidden)[:, -1, :]

            # Greedy sample
            code_0 = torch.argmax(logits_0, dim=-1)

            # Generate remaining codebooks with code_predictor
            # PyTorch: CodePredictor([past_hidden, embed(code_0)])
            past_hidden = last_hidden
            last_id_hidden = model.model.talker.get_input_embeddings()(code_0.unsqueeze(-1))

            # Concatenate past_hidden + embed(code_0)
            cp_input = torch.cat([past_hidden, last_id_hidden], dim=1)

            codes_step = [code_0.unsqueeze(-1)]

            with torch.no_grad():
                # First CodePredictor forward with 2-token input
                cp_output = code_predictor.model(inputs_embeds=cp_input)
                cp_hidden = cp_output.last_hidden_state[:, -1:, :]
                cp_logits = code_predictor.lm_head[0](cp_hidden)[:, -1, :]
                cp_code = torch.argmax(cp_logits, dim=-1)
                codes_step.append(cp_code.unsqueeze(-1))

                # Continue with remaining codebooks
                cp_next_input = code_predictor.get_input_embeddings()[0](cp_code.unsqueeze(-1))

                for cb_idx in range(1, 15):
                    cp_output = code_predictor.model(inputs_embeds=cp_next_input)
                    cp_hidden = cp_output.last_hidden_state[:, -1:, :]
                    cp_logits = code_predictor.lm_head[cb_idx](cp_hidden)[:, -1, :]
                    cp_code = torch.argmax(cp_logits, dim=-1)
                    codes_step.append(cp_code.unsqueeze(-1))
                    cp_next_input = code_predictor.get_input_embeddings()[cb_idx](cp_code.unsqueeze(-1))

            codes_step = torch.cat(codes_step, dim=-1)  # (batch, 16)
            captured['codes'].append(codes_step.cpu().numpy())

            print(f"  PT Step {step}: code_0={code_0.item()}, codes={codes_step[0, :5].tolist()}")

            # Check EOS
            if code_0.item() == 2150:
                break

            # Prepare next input
            codec_hiddens = [last_id_hidden]
            for i in range(15):
                embed = code_predictor.get_input_embeddings()[i](codes_step[:, i+1:i+2])
                codec_hiddens.append(embed)
            codec_hiddens = torch.cat(codec_hiddens, dim=1)
            next_input = codec_hiddens.sum(1, keepdim=True)

            if step < text_len:
                next_input = next_input + trailing_text_hidden[:, step:step+1]
            else:
                next_input = next_input + tts_pad_embed

            current_input = next_input

        # Return dummy
        class FakeResult:
            sequences = torch.zeros((1, 1, 16), dtype=torch.long)
        return FakeResult()

    model.model.talker.generate = patched_generate

    try:
        with torch.no_grad():
            model.generate_custom_voice(
                text=text,
                language="Japanese",
                speaker=speaker,
            )
    except Exception as e:
        print(f"  (Expected: {type(e).__name__})")

    model.model.talker.generate = original_generate

    return captured, model


def run_mlx_generation(initial_embeds, trailing_text_hidden, tts_pad_embed, weights_dir, num_steps: int = 5):
    """Run MLX generation and capture step-by-step codes."""
    print("\nLoading MLX models...")

    # Load Talker
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
    cp_config = CodePredictorConfig()
    code_predictor = Qwen3TTSCodePredictorMLX(cp_config)
    code_predictor = load_code_predictor_weights(code_predictor, str(weights_dir / 'code_predictor_weights_mlx.npz'))

    # Create generator
    generator = MLXGenerateLoop(talker, code_predictor, codec_head_weight)

    # Convert inputs
    initial_embeds_mlx = mx.array(initial_embeds.cpu().float().numpy())
    trailing_text_hidden_mlx = mx.array(trailing_text_hidden.cpu().float().numpy())
    tts_pad_embed_mlx = mx.array(tts_pad_embed.cpu().float().numpy())

    # Generate with greedy sampling
    config = GenerateConfig(
        max_new_tokens=num_steps,
        do_sample=False,  # Greedy for deterministic comparison
        subtalker_do_sample=False,
        quality_mode="high",  # Full 15 codebooks
    )

    print("\nMLX Generation:")
    codes = generator.generate(
        initial_embeds_mlx,
        config,
        trailing_text_hidden=trailing_text_hidden_mlx,
        tts_pad_embed=tts_pad_embed_mlx,
        debug=True,
    )
    mx.eval(codes)

    return np.array(codes)


def compare_codes(pt_codes, mlx_codes):
    """Compare PyTorch vs MLX codes."""
    print("\n" + "=" * 60)
    print("Code Comparison (Greedy)")
    print("=" * 60)

    num_steps = min(len(pt_codes), mlx_codes.shape[1])

    all_match = True
    for step in range(num_steps):
        pt_step = pt_codes[step][0]  # (16,)
        mlx_step = mlx_codes[0, step]  # (16,)

        match = np.array_equal(pt_step, mlx_step)
        if not match:
            all_match = False
            # Find first mismatch
            for i in range(16):
                if pt_step[i] != mlx_step[i]:
                    print(f"Step {step}: MISMATCH at codebook {i}")
                    print(f"  PT:  {pt_step[:8].tolist()}")
                    print(f"  MLX: {mlx_step[:8].tolist()}")
                    break
        else:
            print(f"Step {step}: ✅ MATCH (code_0={pt_step[0]})")

    if all_match:
        print("\n✅ All codes match perfectly!")
    else:
        print("\n❌ Some codes don't match")

    return all_match


def main():
    print("=" * 60)
    print("MLX vs PyTorch Code Comparison")
    print("=" * 60)
    print()

    weights_dir = Path(__file__).parent.parent
    num_steps = 5

    # PyTorch generation
    print("--- PyTorch Generation ---")
    pt_data, pt_model = run_pytorch_generation(text="こんにちは。", num_steps=num_steps)

    # MLX generation
    print("\n--- MLX Generation ---")
    mlx_codes = run_mlx_generation(
        pt_data['initial']['inputs_embeds'],
        pt_data['initial']['trailing_text_hidden'],
        pt_data['initial']['tts_pad_embed'],
        weights_dir,
        num_steps=num_steps,
    )

    # Compare
    compare_codes(pt_data['codes'], mlx_codes)


if __name__ == "__main__":
    main()
