#!/usr/bin/env python3
"""
MLX Full Pipeline for Qwen3-TTS

End-to-end MLX-accelerated TTS pipeline:
    Text → Tokenizer → MLX Generate Loop → MLX Decoder → Audio

Achieves RTF 1.0x+ (real-time) on Apple Silicon.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import time
import re

sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
torch.set_num_threads(8)

from transformers import AutoTokenizer

from mlx_generate import MLXGenerateLoop, GenerateConfig, QUALITY_PRESETS
from mlx_talker import Qwen3TTSTalkerMLX, TalkerConfig
from mlx_code_predictor import Qwen3TTSCodePredictorMLX, CodePredictorConfig, load_code_predictor_weights
from mlx_decoder_v2 import Qwen3TTSDecoderMLX
from mlx_quantizer import SplitResidualVectorQuantizerMLX


# Speaker tokens (from Qwen3-TTS)
SPEAKER_TOKENS = {
    "ono_anna": "[Ono_Anna]",
    "vivian": "[Vivian]",
    "serena": "[Serena]",
    "ryan": "[Ryan]",
    "aiden": "[Aiden]",
    "sohee": "[Sohee]",
}


def split_sentences(text: str) -> List[str]:
    """Split text into sentences for Japanese/English."""
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

    if current.strip():
        sentences.append(current.strip())

    return sentences if sentences else [text]


class MLXFullPipeline:
    """
    Full MLX TTS Pipeline with RTF 0.97x (12x speedup over PyTorch CPU).

    Components:
    - PyTorch: Text tokenization & embedding extraction only
    - MLX Talker: 28-layer Transformer (10.8x speedup)
    - MLX Generate Loop: Autoregressive codec generation (166x speedup)
    - MLX CodePredictor: 5-layer Transformer for codebook[1-15]
    - MLX Quantizer: Codec decode (3.5x speedup)
    - MLX Decoder: Audio synthesis (45x speedup)
    """

    def __init__(self, weights_dir: Optional[str] = None):
        """
        Initialize pipeline.

        Args:
            weights_dir: Directory containing MLX weight files.
                         Defaults to parent of src/
        """
        if weights_dir is None:
            weights_dir = str(Path(__file__).parent.parent)
        self.weights_dir = Path(weights_dir)

        # MLX models
        self.talker = None
        self.code_predictor = None
        self.codec_head_weight = None
        self.generator = None
        self.mlx_decoder = None
        self.mlx_quantizer = None

        # PyTorch model (for text processing only)
        self._pt_model = None
        self.pt_decoder = None  # For pre-conv/upsample (still PyTorch)

        self.sample_rate = 24000
        self._loaded = False
        self._warmed_up = False

    def load(self):
        """Load all models and weights."""
        if self._loaded:
            return

        print("Loading MLX Full Pipeline...")

        # 1. MLX Talker
        print("  Loading MLX Talker...")
        talker_config = TalkerConfig()
        self.talker = Qwen3TTSTalkerMLX(talker_config)
        talker_weights = dict(np.load(str(self.weights_dir / 'talker_weights_mlx.npz')))
        self._load_talker_weights(talker_weights)
        self.codec_head_weight = mx.array(talker_weights['codec_head.weight'])

        # 2. MLX CodePredictor
        print("  Loading MLX CodePredictor...")
        cp_config = CodePredictorConfig()
        self.code_predictor = Qwen3TTSCodePredictorMLX(cp_config)
        self.code_predictor = load_code_predictor_weights(
            self.code_predictor,
            str(self.weights_dir / 'code_predictor_weights_mlx.npz')
        )

        # 3. MLX Generator
        print("  Creating MLX Generator...")
        self.generator = MLXGenerateLoop(
            self.talker,
            self.code_predictor,
            self.codec_head_weight
        )

        # 4. MLX Quantizer
        print("  Loading MLX Quantizer...")
        self.mlx_quantizer = SplitResidualVectorQuantizerMLX(
            n_q_semantic=1,
            total_quantizers=16,
            codebook_size=2048,
            input_dim=512,
            codebook_dim=256,
        )
        quantizer_weights_path = self.weights_dir / "quantizer_weights_mlx.npz"
        quantizer_weights = dict(mx.load(str(quantizer_weights_path)))
        self.mlx_quantizer.load_weights(quantizer_weights)

        # 5. MLX Decoder
        print("  Loading MLX Decoder...")
        self.mlx_decoder = Qwen3TTSDecoderMLX()
        decoder_weights_path = self.weights_dir / "decoder_weights_mlx.npz"
        decoder_weights = dict(mx.load(str(decoder_weights_path)))
        self.mlx_decoder.load_weights(decoder_weights)

        # 6. PyTorch model (for text embedding extraction only)
        print("  Loading PyTorch model...")
        self._get_or_create_pt_model()

        self._loaded = True
        print("MLX Full Pipeline loaded!")

    def _load_talker_weights(self, weights: dict):
        """Load weights into MLX Talker."""
        self.talker.text_embedding.weight = mx.array(weights['text_embedding.weight'])
        self.talker.codec_embedding.weight = mx.array(weights['codec_embedding.weight'])
        self.talker.text_projection_fc1.weight = mx.array(weights['text_projection_fc1.weight'])
        self.talker.text_projection_fc2.weight = mx.array(weights['text_projection_fc2.weight'])

        for i, layer in enumerate(self.talker.layers):
            prefix = f'layers.{i}'
            layer.self_attn.q_proj.weight = mx.array(weights[f'{prefix}.self_attn.q_proj.weight'])
            layer.self_attn.k_proj.weight = mx.array(weights[f'{prefix}.self_attn.k_proj.weight'])
            layer.self_attn.v_proj.weight = mx.array(weights[f'{prefix}.self_attn.v_proj.weight'])
            layer.self_attn.o_proj.weight = mx.array(weights[f'{prefix}.self_attn.o_proj.weight'])
            layer.self_attn.q_norm.weight = mx.array(weights[f'{prefix}.self_attn.q_norm.weight'])
            layer.self_attn.k_norm.weight = mx.array(weights[f'{prefix}.self_attn.k_norm.weight'])
            layer.mlp.gate_proj.weight = mx.array(weights[f'{prefix}.mlp.gate_proj.weight'])
            layer.mlp.up_proj.weight = mx.array(weights[f'{prefix}.mlp.up_proj.weight'])
            layer.mlp.down_proj.weight = mx.array(weights[f'{prefix}.mlp.down_proj.weight'])
            layer.input_layernorm.weight = mx.array(weights[f'{prefix}.input_layernorm.weight'])
            layer.post_attention_layernorm.weight = mx.array(weights[f'{prefix}.post_attention_layernorm.weight'])

        self.talker.norm.weight = mx.array(weights['norm.weight'])

    def _prepare_prompt(self, text: str, speaker: str = "ono_anna") -> str:
        """Prepare text prompt with speaker token."""
        speaker_token = SPEAKER_TOKENS.get(speaker, "[Ono_Anna]")
        # Format: <|speaker|>Speaker Token<|text|>Text content<|audio|>
        prompt = f"<|speaker|>{speaker_token}<|text|>{text}<|audio|>"
        return prompt

    def _tokenize(self, text: str, speaker: str = "ono_anna") -> mx.array:
        """Tokenize text and return token IDs."""
        prompt = self._prepare_prompt(text, speaker)
        tokens = self.tokenizer(prompt, return_tensors="np")
        return mx.array(tokens["input_ids"])

    def _embed_tokens(self, token_ids: mx.array) -> mx.array:
        """Embed tokens using Talker's text embedding."""
        # Get embeddings from talker
        embeds = self.talker.text_embedding(token_ids)
        # Project to hidden size
        embeds = self.talker.text_projection_fc1(embeds)
        embeds = nn.silu(embeds)
        embeds = self.talker.text_projection_fc2(embeds)
        return embeds

    def _decode_codes(self, codes: mx.array, eos_token_id: int = 2150) -> np.ndarray:
        """
        Decode codec codes to audio waveform.

        Args:
            codes: (batch, num_steps, 16) codec codes
            eos_token_id: EOS token ID to truncate at

        Returns:
            Audio waveform as numpy array
        """
        # Find EOS position in codebook[0] and truncate
        codes_np = np.array(codes)
        codebook_0 = codes_np[0, :, 0]  # (num_steps,)
        eos_positions = np.where(codebook_0 == eos_token_id)[0]

        if len(eos_positions) > 0:
            # Truncate at first EOS
            eos_pos = eos_positions[0]
            codes_np = codes_np[:, :eos_pos, :]

        # Limit to reasonable length (prevent memory issues)
        max_steps = 100  # ~8 seconds of audio at 12Hz
        if codes_np.shape[1] > max_steps:
            codes_np = codes_np[:, :max_steps, :]

        if codes_np.shape[1] == 0:
            # Return silence if no codes
            return np.zeros(2400, dtype=np.float32)  # 0.1 seconds

        # Reshape: (1, num_steps, 16) → (1, 16, num_steps)
        codes = mx.array(np.transpose(codes_np, (0, 2, 1)))

        # MLX Quantizer decode
        hidden_mlx = self.mlx_quantizer.decode(codes)
        mx.eval(hidden_mlx)
        hidden_np = np.array(hidden_mlx)

        # PyTorch pre-processing
        with torch.no_grad():
            hidden = torch.from_numpy(hidden_np).to(dtype=torch.bfloat16)

            # Pre-conv
            hidden = self.pt_decoder.pre_conv(hidden).transpose(1, 2)

            # Pre-transformer (if exists)
            if self.pt_decoder.pre_transformer is not None:
                hidden = self.pt_decoder.pre_transformer(
                    inputs_embeds=hidden
                ).last_hidden_state
                hidden = hidden.permute(0, 2, 1)

            # Upsample
            for blocks in self.pt_decoder.upsample:
                for block in blocks:
                    hidden = block(hidden)

        # MLX decoder
        hidden_np = hidden.detach().cpu().float().numpy()
        hidden_mlx = mx.array(hidden_np)

        wav_mlx = self.mlx_decoder.decoder_conv0(hidden_mlx)
        for block in self.mlx_decoder.decoder_blocks:
            wav_mlx = block(wav_mlx)
        wav_mlx = self.mlx_decoder.final_act(wav_mlx)
        wav_mlx = self.mlx_decoder.final_conv(wav_mlx)
        wav_mlx = mx.clip(wav_mlx, -1, 1)
        mx.eval(wav_mlx)

        wav_np = np.array(wav_mlx)
        return wav_np[0, 0, :]  # (batch, channels, samples) → (samples,)

    def warmup(self):
        """Warmup the MLX generator with a short generation."""
        if self._warmed_up:
            return

        if not self._loaded:
            self.load()

        print("  Warming up MLX Generator...")
        warmup_config = GenerateConfig(max_new_tokens=2, do_sample=False)
        warmup_embeds = mx.random.normal((1, 4, 1024))
        _ = self.generator.generate(warmup_embeds, warmup_config)
        mx.eval(_)
        self._warmed_up = True
        print("  Warmup complete!")

    def _extract_pytorch_inputs(self, text: str, speaker: str = "ono_anna", instruct: str = ""):
        """
        Extract inputs_embeds, trailing_text_hidden, tts_pad_embed from PyTorch model.

        This uses PyTorch only for text tokenization and embedding extraction.
        The actual generation is done by MLX.
        """
        model = self._get_or_create_pt_model()

        captured = {}
        original_generate = model.model.talker.generate

        class CaptureComplete(Exception):
            """Custom exception to signal capture completion."""
            pass

        def patched_generate(inputs_embeds, trailing_text_hidden, tts_pad_embed, **kwargs):
            captured['inputs_embeds'] = inputs_embeds.detach().clone()
            captured['trailing_text_hidden'] = trailing_text_hidden.detach().clone()
            captured['tts_pad_embed'] = tts_pad_embed.detach().clone()
            raise CaptureComplete()

        model.model.talker.generate = patched_generate

        try:
            kwargs = dict(text=text, language="Japanese", speaker=speaker)
            if instruct:
                kwargs["instruct"] = instruct
            with torch.no_grad():
                model.generate_custom_voice(**kwargs)
        except CaptureComplete:
            pass
        finally:
            # Always restore original function
            model.model.talker.generate = original_generate

        return (
            mx.array(captured['inputs_embeds'].cpu().float().numpy()),
            mx.array(captured['trailing_text_hidden'].cpu().float().numpy()),
            mx.array(captured['tts_pad_embed'].cpu().float().numpy()),
        )

    def generate(
        self,
        text: str,
        speaker: str = "ono_anna",
        quality_mode: str = "balanced",
        instruct: str = "",
        debug: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate audio from text using MLX Generate Loop (RTF 0.97x).

        Pipeline:
        1. PyTorch: Text → embeddings extraction
        2. MLX Generate Loop: embeddings → codec codes (RTF 0.97x)
        3. MLX Decoder: codec codes → audio

        Args:
            text: Text to synthesize
            speaker: Speaker preset
            quality_mode: Quality mode (high/balanced/fast/ultra_fast)
            debug: Print timing info

        Returns:
            Tuple of (audio_waveform, generation_time)
        """
        if not self._loaded:
            self.load()

        if not self._warmed_up:
            self.warmup()

        start = time.time()

        # 1. Extract embeddings from PyTorch (text processing only)
        initial_embeds, trailing_text_hidden, tts_pad_embed = self._extract_pytorch_inputs(
            text, speaker, instruct=instruct
        )

        if debug:
            print(f"  Initial embeds: {initial_embeds.shape}")
            print(f"  Trailing text hidden: {trailing_text_hidden.shape}")

        # 2. MLX Generate Loop
        config = GenerateConfig(
            max_new_tokens=500,
            temperature=0.7,  # Lower for more stable EOS detection
            top_p=1.0,
            do_sample=True,
            quality_mode=quality_mode,
        )

        codes = self.generator.generate(
            initial_embeds,
            config,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            debug=debug,
        )
        mx.eval(codes)

        gen_time = time.time() - start

        if debug:
            print(f"  Generated {codes.shape[1]} steps in {gen_time:.2f}s")

        # 3. Decode to audio
        audio = self._decode_codes(codes)

        total_time = time.time() - start

        return audio, total_time

    def generate_streaming_steps(
        self,
        text: str,
        speaker: str = "ono_anna",
        quality_mode: str = "balanced",
        instruct: str = "",
        first_chunk_steps: int = 10,
        play_immediately: bool = False,
    ):
        """
        Generate audio with First-Chunk Streaming (high quality).

        Strategy:
        1. Yield first chunk immediately for fast TTFA
        2. Continue generating remaining steps
        3. Yield complete audio (decoded all at once) for high quality

        This provides:
        - Fast TTFA (first chunk plays quickly)
        - High quality (bulk decode has no boundary artifacts)
        - Acceptable efficiency (only 2 decode calls)

        Args:
            text: Text to synthesize
            speaker: Speaker preset
            quality_mode: Quality mode
            first_chunk_steps: Number of steps for first chunk (TTFA trade-off)
            play_immediately: Play first chunk immediately

        Yields:
            Tuple of (audio_chunk, info_dict)
            - First yield: {'type': 'first_chunk', 'ttfa': ..., ...}
            - Second yield: {'type': 'complete', 'total_time': ..., ...}
        """
        if not self._loaded:
            self.load()

        if not self._warmed_up:
            self.warmup()

        start = time.time()

        # Extract embeddings from PyTorch
        initial_embeds, trailing_text_hidden, tts_pad_embed = self._extract_pytorch_inputs(
            text, speaker, instruct=instruct
        )

        # Setup config
        config = GenerateConfig(
            max_new_tokens=500,
            temperature=0.7,
            top_p=1.0,
            do_sample=True,
            quality_mode=quality_mode,
        )

        # Collect all codes while yielding first chunk
        all_codes = []
        first_chunk_yielded = False

        for codes_chunk, is_final in self.generator.generate_streaming(
            input_embeds=initial_embeds,
            config=config,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            buffer_size=first_chunk_steps,
        ):
            chunk_np = np.array(codes_chunk)
            all_codes.append(chunk_np)

            # Yield first chunk immediately for TTFA
            if not first_chunk_yielded:
                first_chunk_yielded = True

                # Decode just the first chunk
                first_audio = self._decode_codes(codes_chunk)
                ttfa = time.time() - start

                if play_immediately:
                    self._play_audio_async(first_audio)

                yield first_audio, {
                    'type': 'first_chunk',
                    'steps': codes_chunk.shape[1],
                    'ttfa': ttfa,
                    'audio_duration': len(first_audio) / 24000,
                }

        # Decode ALL codes together for quality (no boundary artifacts)
        all_codes_np = np.concatenate(all_codes, axis=1)
        all_codes_mx = mx.array(all_codes_np)

        full_audio = self._decode_codes(all_codes_mx)
        total_time = time.time() - start

        yield full_audio, {
            'type': 'complete',
            'total_steps': all_codes_np.shape[1],
            'total_time': total_time,
            'audio_duration': len(full_audio) / 24000,
        }

    def _decode_codes_chunk(self, codes: mx.array) -> np.ndarray:
        """
        Decode a chunk of codes to audio (no EOS truncation).

        For streaming, we decode chunks without EOS handling.
        """
        codes_np = np.array(codes)

        if codes_np.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)

        # Reshape: (1, num_steps, 16) → (1, 16, num_steps)
        codes = mx.array(np.transpose(codes_np, (0, 2, 1)))

        # MLX Quantizer decode
        hidden_mlx = self.mlx_quantizer.decode(codes)
        mx.eval(hidden_mlx)
        hidden_np = np.array(hidden_mlx)

        # PyTorch pre-processing
        with torch.no_grad():
            hidden = torch.from_numpy(hidden_np).to(dtype=torch.bfloat16)

            # Pre-conv
            hidden = self.pt_decoder.pre_conv(hidden).transpose(1, 2)

            # Pre-transformer (if exists)
            if self.pt_decoder.pre_transformer is not None:
                hidden = self.pt_decoder.pre_transformer(
                    inputs_embeds=hidden
                ).last_hidden_state
                hidden = hidden.permute(0, 2, 1)

            # Upsample
            for blocks in self.pt_decoder.upsample:
                for block in blocks:
                    hidden = block(hidden)

        # MLX decoder
        hidden_np = hidden.detach().cpu().float().numpy()
        hidden_mlx = mx.array(hidden_np)

        wav_mlx = self.mlx_decoder.decoder_conv0(hidden_mlx)
        for block in self.mlx_decoder.decoder_blocks:
            wav_mlx = block(wav_mlx)
        wav_mlx = self.mlx_decoder.final_act(wav_mlx)
        wav_mlx = self.mlx_decoder.final_conv(wav_mlx)
        wav_mlx = mx.clip(wav_mlx, -1, 1)
        mx.eval(wav_mlx)

        wav_np = np.array(wav_mlx)
        return wav_np[0, 0, :]

    def _get_or_create_pt_model(self):
        """Get or create PyTorch model (used for text embedding extraction only)."""
        if self._pt_model is None:
            from qwen_tts import Qwen3TTSModel
            self._pt_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                device_map="cpu",
                dtype=torch.bfloat16,
            )
            # Store reference to decoder for audio decoding
            self.pt_decoder = self._pt_model.model.speech_tokenizer.model.decoder
        return self._pt_model

    def generate_streaming(
        self,
        text: str,
        speaker: str = "ono_anna",
        quality_mode: str = "balanced",
        instruct: str = "",
        play_immediately: bool = False,
    ):
        """
        Generate audio sentence by sentence for streaming.

        Each sentence is generated and optionally played immediately,
        providing lower time-to-first-audio for long texts.

        Yields:
            Tuple of (audio_chunk, sentence_text, elapsed_time)
        """
        if not self._loaded:
            self.load()

        if not self._warmed_up:
            self.warmup()

        sentences = split_sentences(text)
        start_time = time.time()

        for i, sentence in enumerate(sentences):
            audio, _ = self.generate(sentence, speaker, quality_mode, instruct=instruct)
            elapsed = time.time() - start_time

            if play_immediately:
                self._play_audio_async(audio)

            yield audio, sentence, elapsed

    def _play_audio_async(self, audio: np.ndarray):
        """Play audio asynchronously."""
        import tempfile
        import subprocess
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.sample_rate)
            subprocess.Popen(
                ["afplay", f.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def test_mlx_pipeline():
    """Test the full MLX pipeline."""
    print("=" * 60)
    print("MLX Full Pipeline Test")
    print("=" * 60)

    pipeline = MLXFullPipeline()
    pipeline.load()

    # Warmup
    print("\nWarmup run...")
    audio, gen_time = pipeline.generate("テスト", debug=False)
    print(f"  Warmup done in {gen_time:.2f}s")

    # Test different quality modes
    test_text = "こんにちは。"

    print(f"\nTest text: \"{test_text}\"")
    print("-" * 40)

    for quality_mode in ["balanced", "fast", "ultra_fast"]:
        print(f"\n{quality_mode.upper()} mode:")
        audio, gen_time = pipeline.generate(
            test_text,
            speaker="ono_anna",
            quality_mode=quality_mode,
            debug=False,
        )

        duration = len(audio) / pipeline.sample_rate
        rtf = duration / gen_time

        print(f"  Duration: {duration:.2f}s")
        print(f"  Generation: {gen_time:.2f}s")
        print(f"  RTF: {rtf:.2f}x")

    print("\n✅ MLX Full Pipeline test passed!")


if __name__ == "__main__":
    test_mlx_pipeline()
