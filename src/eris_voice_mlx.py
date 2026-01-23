"""
Eris Voice with MLX Acceleration

PyTorch (LLM + pre_transformer) → MLX (Decoder) のハイブリッド実装。
"""

import torch
import mlx.core as mx
import numpy as np
import soundfile as sf
import time
import os

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)


class ErisVoiceMLX:
    """
    MLX-accelerated Eris Voice.

    Uses PyTorch for LLM generation and MLX for audio decoding.
    """

    def __init__(self):
        self.pytorch_model = None
        self.mlx_decoder = None
        self.sample_rate = 24000

        # Eris style instruction
        self.instruct = (
            "Speak with a confident, slightly teasing tone. "
            "The voice should be elegant yet playful, with a hint of mischief."
        )

    def load(self):
        """Load both PyTorch and MLX components."""
        print("Loading Eris Voice (MLX-accelerated)...")

        # Load PyTorch model
        print("  Loading PyTorch model...")
        start = time.time()
        from qwen_tts import Qwen3TTSModel

        self.pytorch_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cpu",
            dtype=torch.bfloat16,
        )
        pytorch_time = time.time() - start
        print(f"    PyTorch model loaded in {pytorch_time:.1f}s")

        # Load MLX decoder
        print("  Loading MLX decoder...")
        start = time.time()
        from mlx_decoder_v2 import Qwen3TTSDecoderMLX

        self.mlx_decoder = Qwen3TTSDecoderMLX()
        weights_path = os.path.join(os.path.dirname(__file__), "decoder_weights_mlx.npz")
        if not os.path.exists(weights_path):
            weights_path = "decoder_weights_mlx.npz"

        weights = dict(mx.load(weights_path))
        self.mlx_decoder.load_weights(weights)
        mlx_time = time.time() - start
        print(f"    MLX decoder loaded in {mlx_time:.1f}s")

        print(f"  Total load time: {pytorch_time + mlx_time:.1f}s")

    def speak(self, text: str, output_path: str = "output.wav"):
        """
        Generate speech from text using hybrid pipeline.

        Args:
            text: Text to speak
            output_path: Output WAV file path
        """
        if self.pytorch_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_total = time.time()

        # Step 1: PyTorch - Generate codes and get hidden states before decoder
        print("  Step 1: PyTorch LLM + pre_transformer...")
        start = time.time()

        decoder = self.pytorch_model.model.speech_tokenizer.model.decoder

        with torch.no_grad():
            # Full generation to get codes
            wavs, sr = self.pytorch_model.generate_custom_voice(
                text=text,
                language="Japanese",
                speaker="ono_anna",
                instruct=self.instruct,
            )

        pytorch_time = time.time() - start

        # For now, use PyTorch output directly
        # TODO: Extract intermediate hidden states for MLX decoder
        wav = wavs[0]

        total_time = time.time() - start_total
        duration = len(wav) / sr

        # Save
        sf.write(output_path, wav, sr)

        realtime_factor = duration / total_time
        print(f"  Generated {duration:.2f}s audio in {total_time:.1f}s ({realtime_factor:.3f}x realtime)")
        print(f"  Saved to: {output_path}")

        return wav, sr


class ErisVoiceMLXDirect:
    """
    Direct MLX pipeline - extracts hidden states and uses MLX decoder.

    This version intercepts the PyTorch pipeline to use MLX for the decoder.
    """

    def __init__(self):
        self.pytorch_model = None
        self.mlx_decoder = None
        self.sample_rate = 24000

        self.instruct = (
            "Speak with a confident, slightly teasing tone. "
            "The voice should be elegant yet playful, with a hint of mischief."
        )

    def load(self):
        """Load components."""
        print("Loading Eris Voice (MLX Direct)...")

        # Load PyTorch model
        print("  Loading PyTorch model...")
        start = time.time()
        from qwen_tts import Qwen3TTSModel

        self.pytorch_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cpu",
            dtype=torch.bfloat16,
        )
        print(f"    Loaded in {time.time() - start:.1f}s")

        # Load MLX decoder
        print("  Loading MLX decoder...")
        start = time.time()
        from mlx_decoder_v2 import Qwen3TTSDecoderMLX

        self.mlx_decoder = Qwen3TTSDecoderMLX()
        weights = dict(mx.load("decoder_weights_mlx.npz"))
        self.mlx_decoder.load_weights(weights)
        print(f"    Loaded in {time.time() - start:.1f}s")

    def _generate_with_mlx_decoder(self, text: str):
        """
        Generate audio using PyTorch for everything except final decoder.

        Returns hidden states after pre_transformer, then uses MLX decoder.
        """
        decoder = self.pytorch_model.model.speech_tokenizer.model.decoder

        with torch.no_grad():
            # Generate codes using the full model
            # We need to intercept the speech_tokenizer.decode call

            # For now, let's manually run the pipeline
            wavs, sr = self.pytorch_model.generate_custom_voice(
                text=text,
                language="Japanese",
                speaker="ono_anna",
                instruct=self.instruct,
            )

            # This still uses PyTorch decoder
            # To truly use MLX decoder, we need to:
            # 1. Get codes from LLM
            # 2. Run quantizer.decode + pre_conv + pre_transformer in PyTorch
            # 3. Run upsample + decoder in MLX

            return wavs[0], sr

    def speak(self, text: str, output_path: str = "output.wav"):
        """Generate speech."""
        if self.pytorch_model is None:
            raise RuntimeError("Call load() first")

        start = time.time()
        wav, sr = self._generate_with_mlx_decoder(text)
        total_time = time.time() - start

        duration = len(wav) / sr
        sf.write(output_path, wav, sr)

        print(f"Generated {duration:.2f}s in {total_time:.1f}s ({duration/total_time:.3f}x realtime)")
        print(f"Saved: {output_path}")

        return wav, sr


if __name__ == "__main__":
    print("=== Eris Voice MLX Test ===\n")

    voice = ErisVoiceMLX()
    voice.load()

    print("\nGenerating speech...")
    text = "ふふ...面白いことを言うわね"
    voice.speak(text, "test_eris_mlx.wav")
