"""
Streaming TTS Prototype for Qwen3-TTS

ストリーミング音声生成の実験的実装。
Time-to-First-Audio を最小化して体感速度を向上。

## Approaches

### 1. Sentence Chunking (Implemented)
長いテキストを文単位で分割し、各文を順次処理。
最初の文が終わった時点で再生開始できる。

Example:
    Text: "こんにちは。私はエリスよ。よろしくね。"
    → Sentence 1: "こんにちは。" → Audio 1 (再生開始!)
    → Sentence 2: "私はエリスよ。" → Audio 2
    → Sentence 3: "よろしくね。" → Audio 3

### 2. Token-level Streaming (Future)
LLMのコード生成をフックして、N個のコードごとにデコード。
qwen_tts の talker.generate() の内部に介入が必要。

Usage:
    python streaming_prototype.py --text "こんにちは。私はエリスよ。"
    python streaming_prototype.py --text "長文" --mode sentence
"""

import torch
import mlx.core as mx
import numpy as np
import soundfile as sf
import time
import os
import argparse
import re
import subprocess
import sys
from typing import Generator, Tuple, List

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences for Japanese/English."""
    # Japanese sentence endings + English
    pattern = r'([。！？!?]+)'
    parts = re.split(pattern, text)

    sentences = []
    current = ""
    for part in parts:
        current += part
        if re.match(pattern, part):
            if current.strip():
                sentences.append(current.strip())
            current = ""

    # Add remaining text
    if current.strip():
        sentences.append(current.strip())

    return sentences if sentences else [text]


class StreamingTTSPipeline:
    """Streaming TTS with sentence-level chunking."""

    def __init__(self):
        self.model = None
        self.mlx_decoder = None
        self.mlx_quantizer = None
        self.sample_rate = 24000
        self._loaded = False

    def load(self):
        """Load all models."""
        if self._loaded:
            return

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
        decoder_weights_path = os.path.join(
            os.path.dirname(__file__), "..", "decoder_weights_mlx.npz"
        )
        decoder_weights = dict(mx.load(decoder_weights_path))
        self.mlx_decoder.load_weights(decoder_weights)

        # MLX quantizer
        print("Loading MLX quantizer...")
        from mlx_quantizer import SplitResidualVectorQuantizerMLX

        self.mlx_quantizer = SplitResidualVectorQuantizerMLX(
            n_q_semantic=1,
            total_quantizers=16,
            codebook_size=2048,
            input_dim=512,
            codebook_dim=256,
        )
        quantizer_weights_path = os.path.join(
            os.path.dirname(__file__), "..", "quantizer_weights_mlx.npz"
        )
        quantizer_weights = dict(mx.load(quantizer_weights_path))
        self.mlx_quantizer.load_weights(quantizer_weights)

        # Patch decoder for MLX
        self._patch_decoder()

        self._loaded = True
        print("Models loaded!")

    def _patch_decoder(self):
        """Monkey-patch decoder to use MLX."""
        pt_decoder = self.model.model.speech_tokenizer.model.decoder
        mlx_decoder = self.mlx_decoder
        mlx_quantizer = self.mlx_quantizer

        def hybrid_forward(codes):
            with torch.no_grad():
                # MLX quantizer
                codes_np = codes.detach().cpu().numpy()
                codes_mlx = mx.array(codes_np)
                hidden_mlx = mlx_quantizer.decode(codes_mlx)
                mx.eval(hidden_mlx)
                hidden_np = np.array(hidden_mlx)
                hidden = torch.from_numpy(hidden_np).to(dtype=torch.bfloat16)

                # PyTorch pre-processing
                hidden = pt_decoder.pre_conv(hidden).transpose(1, 2)
                if pt_decoder.pre_transformer is not None:
                    hidden = pt_decoder.pre_transformer(
                        inputs_embeds=hidden
                    ).last_hidden_state
                    hidden = hidden.permute(0, 2, 1)

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
            return torch.from_numpy(wav_np)

        pt_decoder.forward = hybrid_forward

    def generate_sentence(self, text: str, speaker: str = "ono_anna") -> np.ndarray:
        """Generate audio for a single sentence."""
        with torch.no_grad():
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language="Japanese",
                speaker=speaker,
            )
        return wavs[0]

    def generate_streaming(
        self,
        text: str,
        speaker: str = "ono_anna",
        play_immediately: bool = False,
    ) -> Generator[Tuple[np.ndarray, str, float], None, None]:
        """
        Generate audio sentence by sentence.

        Args:
            text: Full text to synthesize
            speaker: Speaker preset name
            play_immediately: If True, play each chunk as it's generated

        Yields:
            Tuple of (audio_chunk, sentence_text, time_to_first_audio)
        """
        sentences = split_sentences(text)
        start_time = time.time()

        for i, sentence in enumerate(sentences):
            sentence_start = time.time()
            audio = self.generate_sentence(sentence, speaker)
            elapsed = time.time() - start_time

            if play_immediately and i == 0:
                # Play first chunk immediately
                self._play_audio_async(audio)

            yield audio, sentence, elapsed

    def _play_audio_async(self, audio: np.ndarray):
        """Play audio asynchronously using afplay."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.sample_rate)
            subprocess.Popen(
                ["afplay", f.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def demo_streaming():
    """Demonstrate streaming output."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="こんにちは。私はエリスよ。よろしくね。",
        help="Text to generate"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio as it's generated"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare streaming vs non-streaming"
    )
    args = parser.parse_args()

    pipeline = StreamingTTSPipeline()
    pipeline.load()

    sentences = split_sentences(args.text)
    print(f"\n{'='*60}")
    print("Sentence-Level Streaming TTS Demo")
    print(f"{'='*60}")
    print(f"Text: \"{args.text}\"")
    print(f"Sentences: {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. \"{s}\"")
    print(f"{'='*60}\n")

    # Streaming mode
    print("🎵 Streaming Mode (sentence-by-sentence):")
    all_chunks = []
    first_audio_time = None

    for audio, sentence, elapsed in pipeline.generate_streaming(
        args.text, play_immediately=args.play
    ):
        if first_audio_time is None:
            first_audio_time = elapsed

        duration = len(audio) / pipeline.sample_rate
        all_chunks.append(audio)
        print(f"  ✓ \"{sentence[:20]}...\" → {duration:.2f}s audio at t={elapsed:.2f}s")

    total_audio = np.concatenate(all_chunks)
    total_duration = len(total_audio) / pipeline.sample_rate

    print(f"\n  ⏱️  Time-to-First-Audio: {first_audio_time:.2f}s")
    print(f"  📊 Total audio: {total_duration:.2f}s")

    # Save streaming output
    sf.write("streaming_output.wav", total_audio, pipeline.sample_rate)
    print(f"  💾 Saved to: streaming_output.wav")

    # Compare with non-streaming
    if args.compare:
        print(f"\n{'='*60}")
        print("🎵 Non-Streaming Mode (full text at once):")
        start = time.time()
        full_audio = pipeline.generate_sentence(args.text)
        full_elapsed = time.time() - start
        full_duration = len(full_audio) / pipeline.sample_rate

        print(f"  ⏱️  Time-to-First-Audio: {full_elapsed:.2f}s")
        print(f"  📊 Total audio: {full_duration:.2f}s")

        # Comparison
        print(f"\n{'='*60}")
        print("📈 Comparison:")
        print(f"  Streaming TTFA:     {first_audio_time:.2f}s")
        print(f"  Non-Streaming TTFA: {full_elapsed:.2f}s")
        if full_elapsed > first_audio_time:
            speedup = full_elapsed / first_audio_time
            print(f"  🚀 Streaming is {speedup:.1f}x faster to first audio!")

    print()


if __name__ == "__main__":
    demo_streaming()
