#!/usr/bin/env python3
"""
Debug CodePredictor specifically - compare each codebook's logits.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ["OMP_NUM_THREADS"] = "8"

import mlx.core as mx
import numpy as np
import torch
torch.set_num_threads(8)

from mlx_code_predictor import Qwen3TTSCodePredictorMLX, CodePredictorConfig, load_code_predictor_weights


def test_code_predictor_comparison():
    """Compare PyTorch vs MLX CodePredictor step by step."""
    print("=" * 60)
    print("CodePredictor Detailed Comparison")
    print("=" * 60)

    # Load PyTorch model
    from qwen_tts import Qwen3TTSModel
    print("\nLoading PyTorch model...")
    pt_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    pt_cp = pt_model.model.talker.code_predictor

    # Load MLX model
    print("\nLoading MLX model...")
    weights_dir = Path(__file__).parent.parent
    cp_config = CodePredictorConfig()
    mlx_cp = Qwen3TTSCodePredictorMLX(cp_config)
    mlx_cp = load_code_predictor_weights(mlx_cp, str(weights_dir / 'code_predictor_weights_mlx.npz'))

    # Create test input (simulating last_hidden from Talker)
    np.random.seed(42)
    test_input_np = np.random.randn(1, 1, 1024).astype(np.float32)

    pt_input = torch.from_numpy(test_input_np).to(torch.bfloat16)
    mlx_input = mx.array(test_input_np)

    print(f"\nTest input shape: {test_input_np.shape}")

    # Compare forward passes WITHOUT cache (single token input)
    print("\n--- Single Token Forward (No Cache) ---")

    # PyTorch
    with torch.no_grad():
        pt_output = pt_cp.model(inputs_embeds=pt_input, use_cache=False)
        pt_hidden = pt_output.last_hidden_state

    # MLX - no cache
    mlx_hidden = mlx_cp(mlx_input, cache=None)
    mx.eval(mlx_hidden)

    pt_hidden_np = pt_hidden.cpu().float().numpy()
    mlx_hidden_np = np.array(mlx_hidden)

    print(f"  PT hidden shape: {pt_hidden_np.shape}")
    print(f"  MLX hidden shape: {mlx_hidden_np.shape}")
    print(f"  Hidden max diff: {np.abs(pt_hidden_np - mlx_hidden_np).max():.6f}")
    print(f"  Hidden mean diff: {np.abs(pt_hidden_np - mlx_hidden_np).mean():.6f}")
    print(f"  PT hidden[:5]: {pt_hidden_np[0, 0, :5]}")
    print(f"  MLX hidden[:5]: {mlx_hidden_np[0, 0, :5]}")

    # Compare logits for each codebook head
    print("\n--- Logits for each codebook head ---")

    for cb_idx in range(min(5, cp_config.num_codebooks)):  # Just first 5 for brevity
        with torch.no_grad():
            pt_logits = pt_cp.lm_head[cb_idx](pt_hidden[:, -1:, :])
            pt_logits = pt_logits[:, -1, :].cpu().float().numpy()

        mlx_logits = mlx_cp.get_logits(mlx_hidden[:, -1:, :], cb_idx)
        mlx_logits = np.array(mlx_logits[:, -1, :])
        mx.eval(mlx_logits)

        logits_diff = np.abs(pt_logits - mlx_logits).max()
        pt_argmax = np.argmax(pt_logits)
        mlx_argmax = np.argmax(mlx_logits)

        print(f"  Codebook {cb_idx}: max_diff={logits_diff:.6f}, PT argmax={pt_argmax}, MLX argmax={mlx_argmax}, match={pt_argmax==mlx_argmax}")

    # Now test iterative generation
    print("\n--- Iterative Generation (15 codebooks) ---")

    # Reset for fresh comparison
    pt_input_iter = torch.from_numpy(test_input_np).to(torch.bfloat16)
    mlx_input_iter = mx.array(test_input_np)

    pt_codes = []
    mlx_codes = []

    # PyTorch - no cache, fresh forward each time
    current_pt_input = pt_input_iter
    for cb_idx in range(15):
        with torch.no_grad():
            pt_output = pt_cp.model(inputs_embeds=current_pt_input, use_cache=False)
            pt_h = pt_output.last_hidden_state[:, -1:, :]
            pt_logits = pt_cp.lm_head[cb_idx](pt_h)[:, -1, :]
            pt_code = torch.argmax(pt_logits, dim=-1)
            pt_codes.append(pt_code.item())

            # Next input is embedding of this code
            current_pt_input = pt_cp.get_input_embeddings()[cb_idx](pt_code.unsqueeze(-1))

    # MLX - also no cache
    current_mlx_input = mlx_input_iter
    for cb_idx in range(15):
        mlx_h = mlx_cp(current_mlx_input, cache=None)
        mx.eval(mlx_h)
        mlx_logits = mlx_cp.get_logits(mlx_h[:, -1:, :], cb_idx)
        mlx_logits = mlx_logits[:, -1, :]
        mx.eval(mlx_logits)
        mlx_code = mx.argmax(mlx_logits, axis=-1)
        mx.eval(mlx_code)
        mlx_codes.append(int(mlx_code[0]))

        # Next input is embedding of this code
        current_mlx_input = mlx_cp.embed_codes(mlx_code[:, None], cb_idx)
        mx.eval(current_mlx_input)

    print(f"  PT codes: {pt_codes}")
    print(f"  MLX codes: {mlx_codes}")
    print(f"  Codes match: {pt_codes == mlx_codes}")

    # Find first divergence
    for i, (pt_c, mlx_c) in enumerate(zip(pt_codes, mlx_codes)):
        if pt_c != mlx_c:
            print(f"\n  First divergence at codebook {i}: PT={pt_c}, MLX={mlx_c}")

            # Debug this specific codebook
            print(f"\n  Debugging codebook {i} divergence...")

            # Re-run up to this point
            current_pt = pt_input_iter
            current_mlx = mlx_input_iter

            for j in range(i):
                # PyTorch
                with torch.no_grad():
                    pt_out = pt_cp.model(inputs_embeds=current_pt, use_cache=False)
                    pt_h = pt_out.last_hidden_state[:, -1:, :]
                    pt_log = pt_cp.lm_head[j](pt_h)[:, -1, :]
                    pt_c = torch.argmax(pt_log, dim=-1)
                    current_pt = pt_cp.get_input_embeddings()[j](pt_c.unsqueeze(-1))

                # MLX
                mlx_h = mlx_cp(current_mlx, cache=None)
                mx.eval(mlx_h)
                mlx_log = mlx_cp.get_logits(mlx_h[:, -1:, :], j)[:, -1, :]
                mx.eval(mlx_log)
                mlx_c = mx.argmax(mlx_log, axis=-1)
                mx.eval(mlx_c)
                current_mlx = mlx_cp.embed_codes(mlx_c[:, None], j)
                mx.eval(current_mlx)

            # Now we're at codebook i
            print(f"    Input to codebook {i}:")
            print(f"      PT input[:5]: {current_pt[0, 0, :5].cpu().float().numpy()}")
            print(f"      MLX input[:5]: {np.array(current_mlx)[0, 0, :5]}")

            input_diff = np.abs(current_pt.cpu().float().numpy() - np.array(current_mlx)).max()
            print(f"      Input max diff: {input_diff:.6f}")

            # Forward through this codebook
            with torch.no_grad():
                pt_out = pt_cp.model(inputs_embeds=current_pt, use_cache=False)
                pt_h = pt_out.last_hidden_state[:, -1:, :]
                pt_log = pt_cp.lm_head[i](pt_h)[:, -1, :]

            mlx_h = mlx_cp(current_mlx, cache=None)
            mx.eval(mlx_h)
            mlx_log = mlx_cp.get_logits(mlx_h[:, -1:, :], i)[:, -1, :]
            mx.eval(mlx_log)

            print(f"    Hidden after forward:")
            print(f"      PT hidden[:5]: {pt_h[0, 0, :5].cpu().float().numpy()}")
            print(f"      MLX hidden[:5]: {np.array(mlx_h)[0, 0, :5]}")
            hidden_diff = np.abs(pt_h.cpu().float().numpy() - np.array(mlx_h)).max()
            print(f"      Hidden max diff: {hidden_diff:.6f}")

            print(f"    Logits for codebook {i}:")
            pt_log_np = pt_log.cpu().float().numpy()
            mlx_log_np = np.array(mlx_log)
            print(f"      PT logits top5 idx: {np.argsort(pt_log_np[0])[-5:][::-1].tolist()}")
            print(f"      MLX logits top5 idx: {np.argsort(mlx_log_np[0])[-5:][::-1].tolist()}")
            print(f"      PT logits top5 val: {np.sort(pt_log_np[0])[-5:][::-1].tolist()}")
            print(f"      MLX logits top5 val: {np.sort(mlx_log_np[0])[-5:][::-1].tolist()}")

            break

    print("\n" + "=" * 60)
    print("Debug complete!")


if __name__ == "__main__":
    test_code_predictor_comparison()
