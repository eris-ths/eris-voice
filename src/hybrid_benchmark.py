"""
MLX Hybrid Pipeline Benchmark for Qwen3-TTS

PyTorch (LLM + quantizer + pre_conv + pre_transformer + upsample)
→ MLX (decoder blocks) のハイブリッド実装のベンチマーク。

Usage:
    python hybrid_benchmark.py [--text "テキスト"]
"""

import torch
import mlx.core as mx
import numpy as np
import soundfile as sf
import time
import os
import argparse

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)


class HybridPipeline:
    """MLX-accelerated TTS pipeline."""

    def __init__(self):
        self.model = None
        self.mlx_decoder = None
        self.original_forward = None
        self.sample_rate = 24000

    def load(self):
        """Load PyTorch model and MLX decoder."""
        print("Loading models...")

        # PyTorch model
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cpu",
            dtype=torch.bfloat16,
        )

        # MLX decoder
        print("Loading MLX decoder...")
        from mlx_decoder_v2 import Qwen3TTSDecoderMLX

        self.mlx_decoder = Qwen3TTSDecoderMLX()
        weights_path = os.path.join(os.path.dirname(__file__), "..", "decoder_weights_mlx.npz")
        weights = dict(mx.load(weights_path))
        self.mlx_decoder.load_weights(weights)

        # Patch decoder forward
        self._patch_decoder()

    def _patch_decoder(self):
        """Monkey-patch decoder to use MLX for decoder blocks."""
        pt_decoder = self.model.model.speech_tokenizer.model.decoder
        self.original_forward = pt_decoder.forward
        mlx_decoder = self.mlx_decoder

        def hybrid_forward(codes):
            """Hybrid forward: PyTorch for early stages, MLX for decoder blocks."""
            with torch.no_grad():
                # 1. Quantizer decode (PyTorch)
                hidden = pt_decoder.quantizer.decode(codes)

                # 2. Pre-conv + transpose (PyTorch)
                hidden = pt_decoder.pre_conv(hidden).transpose(1, 2)

                # 3. Pre-transformer (PyTorch)
                if pt_decoder.pre_transformer is not None:
                    hidden = pt_decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
                    hidden = hidden.permute(0, 2, 1)

                # 4. Upsample blocks (PyTorch)
                for blocks in pt_decoder.upsample:
                    for block in blocks:
                        hidden = block(hidden)

            # === Switch to MLX ===
            hidden_np = hidden.detach().cpu().float().numpy()
            hidden_mlx = mx.array(hidden_np)

            # 5. Main decoder blocks (MLX - the fast part!)
            wav_mlx = mlx_decoder.decoder_conv0(hidden_mlx)
            for block in mlx_decoder.decoder_blocks:
                wav_mlx = block(wav_mlx)
            wav_mlx = mlx_decoder.final_act(wav_mlx)
            wav_mlx = mlx_decoder.final_conv(wav_mlx)
            wav_mlx = mx.clip(wav_mlx, -1, 1)
            mx.eval(wav_mlx)

            # Back to PyTorch
            wav_np = np.array(wav_mlx)
            return torch.from_numpy(wav_np)

        pt_decoder.forward = hybrid_forward

    def restore_original(self):
        """Restore original PyTorch forward."""
        if self.original_forward:
            pt_decoder = self.model.model.speech_tokenizer.model.decoder
            pt_decoder.forward = self.original_forward

    def generate(
        self,
        text: str,
        instruct: str = "Speak with a confident, slightly teasing tone.",
        output_path: str = None,
    ):
        """Generate speech with timing info."""
        start = time.time()

        with torch.no_grad():
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language="Japanese",
                speaker="ono_anna",
                instruct=instruct,
            )

        elapsed = time.time() - start
        wav = wavs[0]
        duration = len(wav) / sr

        if output_path:
            sf.write(output_path, wav, sr)

        return wav, sr, elapsed, duration


def run_benchmark():
    """Run benchmark comparing PyTorch vs MLX Hybrid."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None, help="Custom text to generate")
    args = parser.parse_args()

    pipeline = HybridPipeline()
    pipeline.load()

    # Test cases
    test_cases = [
        ("5文字", "こんにちは"),
        ("28文字", "ふふ...面白いことを言うわね。私が見逃すとでも思った？"),
    ]

    if args.text:
        test_cases = [(f"{len(args.text)}文字", args.text)]

    # Baseline (PyTorch only)
    baseline = {
        "5文字": 16.0,  # Measured earlier
        "28文字": 98.0,
    }

    print("\n" + "=" * 50)
    print("MLX Hybrid Pipeline Benchmark")
    print("=" * 50)

    for name, text in test_cases:
        print(f"\n{name}: \"{text}\"")

        wav, sr, elapsed, duration = pipeline.generate(
            text=text, output_path=f"hybrid_{name}.wav"
        )

        base = baseline.get(name, elapsed)
        speedup = base / elapsed

        print(f"  Time: {elapsed:.1f}秒 (音声 {duration:.2f}秒)")
        print(f"  Realtime: {duration/elapsed:.3f}x")
        if name in baseline:
            print(f"  Speedup: {speedup:.1f}x (vs PyTorch {base:.0f}秒)")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    run_benchmark()
