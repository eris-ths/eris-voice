"""
Eris Voice Generator - Qwen3-TTS Optimized for Apple Silicon

Generates Eris's voice using Qwen3-TTS with optimizations for CPU inference.
"""

import torch
import soundfile as sf
import os
import time
from pathlib import Path
from typing import Optional

# Optimize CPU threads
os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))


class ErisVoice:
    """Eris voice generator using Qwen3-TTS."""

    DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    DEFAULT_SPEAKER = "ono_anna"
    DEFAULT_LANGUAGE = "Japanese"
    DEFAULT_INSTRUCT = "Slightly teasing, mischievous but charming tone"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Eris voice generator.

        Args:
            model_id: HuggingFace model ID (default: 0.6B for speed)
            device: "cpu" or "mps" (mps has limitations)
            dtype: torch.bfloat16 recommended for CPU
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model = None

    def load(self) -> "ErisVoice":
        """Load the model. Call this before generate()."""
        from qwen_tts import Qwen3TTSModel

        print(f"Loading {self.model_id}...")
        start = time.time()

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=self.dtype,
        )

        print(f"Model loaded in {time.time() - start:.1f}s")
        return self

    def generate(
        self,
        text: str,
        output_path: Optional[str] = None,
        speaker: str = DEFAULT_SPEAKER,
        language: str = DEFAULT_LANGUAGE,
        instruct: str = DEFAULT_INSTRUCT,
    ) -> tuple[list, int, float]:
        """
        Generate audio from text.

        Args:
            text: Text to speak
            output_path: Optional path to save wav file
            speaker: Speaker name (default: ono_anna)
            language: Language (default: Japanese)
            instruct: Style instruction

        Returns:
            tuple: (audio_data, sample_rate, generation_time)
        """
        if self.model is None:
            self.load()

        start = time.time()

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

        gen_time = time.time() - start
        audio = wavs[0]
        duration = len(audio) / sr

        print(f"Generated {duration:.2f}s audio in {gen_time:.1f}s "
              f"({duration/gen_time:.3f}x realtime)")

        if output_path:
            sf.write(output_path, audio, sr)
            print(f"Saved to: {output_path}")

        return audio, sr, gen_time

    def speak(self, text: str, output_path: Optional[str] = None) -> tuple[list, int, float]:
        """Shortcut for generate() with Eris defaults."""
        return self.generate(
            text=text,
            output_path=output_path,
            speaker=self.DEFAULT_SPEAKER,
            language=self.DEFAULT_LANGUAGE,
            instruct=self.DEFAULT_INSTRUCT,
        )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Eris voice")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("-o", "--output", help="Output wav file path")
    parser.add_argument("--model", default=ErisVoice.DEFAULT_MODEL,
                        help="Model ID")
    parser.add_argument("--speaker", default=ErisVoice.DEFAULT_SPEAKER,
                        help="Speaker name")
    args = parser.parse_args()

    voice = ErisVoice(model_id=args.model)
    voice.load()

    output = args.output or f"eris_output_{int(time.time())}.wav"
    voice.speak(args.text, output_path=output)


if __name__ == "__main__":
    main()
