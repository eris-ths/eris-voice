#!/usr/bin/env python3
"""
Eris Voice MCP Server (Direct Pipeline)

MCP server that loads MLXFullPipeline directly in-process.
No HTTP server needed — eliminates network overhead for ~2x faster generation.

Architecture:
    Claude Code → MCP Server (this file, holds MLXFullPipeline in memory)

Trade-offs vs HTTP version:
    + ~2x faster (no HTTP roundtrip, no JSON serialization)
    - MCP startup takes ~40s (model loading + warmup)
    - ~3GB memory in MCP process
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from typing import Optional
from enum import Enum
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

import yaml
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("eris_voice_mcp")

# Voice presets (loaded from YAML)
_PRESETS_PATH = Path(__file__).parent.parent / "voice_presets.yaml"
_voice_presets: dict = {}


def _load_presets() -> dict:
    """Load voice presets from YAML. Reloads on each call for hot-reload."""
    global _voice_presets
    if _PRESETS_PATH.exists():
        with open(_PRESETS_PATH) as f:
            _voice_presets = yaml.safe_load(f) or {}
    return _voice_presets


def _resolve_instruct(voice_mood: str, instruct: str) -> str:
    """Resolve instruct from voice_mood preset or direct instruct string."""
    if instruct:
        return instruct
    if voice_mood:
        presets = _load_presets()
        preset = presets.get(voice_mood)
        if preset:
            return preset.get("instruct", "")
    return ""


# Global pipeline instance (loaded lazily on first call)
_pipeline = None
_warmup_done = False


def _get_pipeline():
    """Get or create the TTS pipeline (lazy loading)."""
    global _pipeline
    if _pipeline is None:
        import torch
        torch.set_num_threads(8)

        from mlx_pipeline import MLXFullPipeline
        _pipeline = MLXFullPipeline()
        _pipeline.load()
    return _pipeline


def _ensure_warmed_up():
    """Ensure pipeline is warmed up."""
    global _warmup_done
    if _warmup_done:
        return
    pipeline = _get_pipeline()
    pipeline.warmup()
    # One real generation to stabilize
    pipeline.generate("テスト", speaker="ono_anna", quality_mode="fast")
    _warmup_done = True


def _play_audio(audio_data, sample_rate: int = 24000):
    """Play audio via afplay (macOS)."""
    import numpy as np
    import soundfile as sf

    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(temp_path, audio_data, sample_rate)
    subprocess.Popen(
        ["afplay", temp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return temp_path


# Enums
class Speaker(str, Enum):
    ONO_ANNA = "ono_anna"
    VIVIAN = "vivian"
    SERENA = "serena"
    RYAN = "ryan"
    AIDEN = "aiden"
    SOHEE = "sohee"


class OutputMode(str, Enum):
    PLAY = "play"
    FILE = "file"
    BOTH = "both"


class QualityMode(str, Enum):
    HIGH = "high"
    BALANCED = "balanced"
    FAST = "fast"
    ULTRA_FAST = "ultra_fast"


# Input Models
class SpeakInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    text: str = Field(
        ...,
        description="Text to synthesize into speech (e.g., 'こんにちは', 'Hello world')",
        min_length=1,
        max_length=500,
    )
    speaker: Speaker = Field(
        default=Speaker.ONO_ANNA,
        description="Speaker preset. ono_anna is recommended for Japanese.",
    )
    output_mode: OutputMode = Field(
        default=OutputMode.PLAY,
        description="Output mode: 'play' to play immediately, 'file' to return path, 'both' for both",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save audio file. Uses temp directory if not specified.",
    )
    quality_mode: QualityMode = Field(
        default=QualityMode.HIGH,
        description="Quality mode: 'high' (best quality, default), 'balanced' (realtime but lower quality), 'fast' (faster), 'ultra_fast' (fastest)",
    )
    voice_mood: str = Field(
        default="default",
        description="Voice mood preset from voice_presets.yaml (e.g., 'default', 'playful', 'calm', 'gentle', 'serious', 'whisper'). Ignored if instruct is set.",
    )
    instruct: str = Field(
        default="",
        description="Direct voice style instruction in English. Overrides voice_mood if set.",
        max_length=200,
    )


class StreamingSpeakInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    text: str = Field(
        ...,
        description="Text to synthesize (can be multiple sentences for streaming)",
        min_length=1,
        max_length=2000,
    )
    speaker: Speaker = Field(
        default=Speaker.ONO_ANNA,
        description="Speaker preset",
    )
    play_as_generated: bool = Field(
        default=True,
        description="Play each sentence as it's generated (streaming mode)",
    )
    quality_mode: QualityMode = Field(
        default=QualityMode.HIGH,
        description="Quality mode for generation",
    )
    voice_mood: str = Field(
        default="default",
        description="Voice mood preset from voice_presets.yaml",
    )
    instruct: str = Field(
        default="",
        description="Direct voice style instruction. Overrides voice_mood if set.",
        max_length=200,
    )


# Tools
@mcp.tool(
    name="eris_speak",
    annotations={
        "title": "Eris TTS Speak",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def eris_speak(params: SpeakInput) -> str:
    """
    Generate speech from text using MLX-accelerated Qwen3-TTS (direct pipeline).

    Runs MLXFullPipeline in-process for maximum speed.
    First call triggers model loading (~40s), subsequent calls are fast.

    Args:
        params (SpeakInput): Input parameters containing:
            - text (str): Text to synthesize (1-500 characters)
            - speaker (Speaker): Voice preset (default: ono_anna)
            - output_mode (OutputMode): play/file/both
            - quality_mode (QualityMode): high/balanced/fast/ultra_fast

    Returns:
        str: JSON response with generation info
    """
    try:
        _ensure_warmed_up()
        pipeline = _get_pipeline()

        resolved_instruct = _resolve_instruct(params.voice_mood, params.instruct)

        audio, generation_time = pipeline.generate(
            text=params.text,
            speaker=params.speaker.value,
            quality_mode=params.quality_mode.value,
            instruct=resolved_instruct,
        )

        duration = len(audio) / 24000
        rtf = duration / generation_time

        audio_path = None

        # Save if needed
        if params.output_mode in (OutputMode.FILE, OutputMode.BOTH):
            import soundfile as sf
            if params.output_dir:
                os.makedirs(params.output_dir, exist_ok=True)
                audio_path = os.path.join(params.output_dir, f"eris_{int(time.time())}.wav")
            else:
                fd, audio_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
            sf.write(audio_path, audio, 24000)

        # Play if needed
        if params.output_mode in (OutputMode.PLAY, OutputMode.BOTH):
            if audio_path:
                subprocess.Popen(
                    ["afplay", audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                _play_audio(audio)

        return json.dumps({
            "success": True,
            "text": params.text,
            "speaker": params.speaker.value,
            "quality_mode": params.quality_mode.value,
            "voice_mood": params.voice_mood,
            "instruct": resolved_instruct,
            "duration_seconds": round(duration, 2),
            "generation_time_seconds": round(generation_time, 2),
            "realtime_factor": round(rtf, 2),
            "audio_path": audio_path,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        }, ensure_ascii=False)


@mcp.tool(
    name="eris_speak_streaming",
    annotations={
        "title": "Eris TTS Streaming",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def eris_speak_streaming(params: StreamingSpeakInput) -> str:
    """
    Generate speech with sentence-level streaming for faster TTFA.

    Splits text into sentences and generates each sequentially.
    When play_as_generated is True, plays audio as soon as each sentence is ready.
    """
    try:
        _ensure_warmed_up()
        pipeline = _get_pipeline()

        import numpy as np
        import soundfile as sf

        sentences = []
        all_audio = []
        start_time = time.time()
        ttfa = None

        resolved_instruct = _resolve_instruct(params.voice_mood, params.instruct)

        for audio, sentence, elapsed in pipeline.generate_streaming(
            params.text,
            speaker=params.speaker.value,
            quality_mode=params.quality_mode.value,
            instruct=resolved_instruct,
            play_immediately=params.play_as_generated,
        ):
            duration = len(audio) / 24000
            sentences.append({
                "text": sentence,
                "duration_seconds": round(duration, 2),
                "elapsed_seconds": round(elapsed, 2),
            })
            all_audio.append(audio)

            if ttfa is None:
                ttfa = elapsed

        total_time = time.time() - start_time

        # Save combined audio
        combined = np.concatenate(all_audio)
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(audio_path, combined, 24000)

        total_duration = len(combined) / 24000

        return json.dumps({
            "success": True,
            "sentences": sentences,
            "total_duration_seconds": round(total_duration, 2),
            "time_to_first_audio": round(ttfa, 2) if ttfa else None,
            "total_generation_time": round(total_time, 2),
            "realtime_factor": round(total_duration / total_time, 2),
            "audio_path": audio_path,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        }, ensure_ascii=False)


@mcp.tool(
    name="eris_list_voice_moods",
    annotations={
        "title": "List Voice Mood Presets",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def eris_list_voice_moods() -> str:
    """List available voice mood presets from voice_presets.yaml (hot-reloaded)."""
    presets = _load_presets()
    result = []
    for key, val in presets.items():
        result.append({
            "mood": key,
            "description": val.get("description", ""),
            "instruct": val.get("instruct", ""),
        })
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="eris_list_speakers",
    annotations={
        "title": "List Eris Voice Speakers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def eris_list_speakers() -> str:
    """List all available speaker presets for Eris Voice."""
    speakers = [
        {"id": "ono_anna", "name": "Ono Anna", "language": "Japanese", "description": "Playful Japanese female voice"},
        {"id": "vivian", "name": "Vivian", "language": "Chinese", "description": "Bright, edgy young female voice"},
        {"id": "serena", "name": "Serena", "language": "Chinese", "description": "Warm, gentle young female voice"},
        {"id": "ryan", "name": "Ryan", "language": "English", "description": "Dynamic male voice"},
        {"id": "aiden", "name": "Aiden", "language": "English", "description": "Sunny American male voice"},
        {"id": "sohee", "name": "Sohee", "language": "Korean", "description": "Warm female voice with rich emotion"},
    ]
    return json.dumps(speakers, ensure_ascii=False)


@mcp.tool(
    name="eris_status",
    annotations={
        "title": "Eris Voice Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def eris_status() -> str:
    """Get the current status of the Eris Voice pipeline (direct mode)."""
    return json.dumps({
        "mode": "direct",
        "loaded": _pipeline is not None,
        "warmed_up": _warmup_done,
        "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "optimizations": [
            "MLX Audio Decoder (45x speedup)",
            "MLX Quantizer (3.5x speedup)",
            "MLX Generate Loop (166x speedup)",
            "Pre-allocated KV Cache (O(1) per step)",
            "mx.fast SDPA (Metal-accelerated)",
            "Codebook pre-computation",
            "Direct pipeline (no HTTP overhead)",
        ],
        "sample_rate": 24000,
    }, ensure_ascii=False)


if __name__ == "__main__":
    print("=" * 50)
    print("Eris Voice MCP Server (Direct Pipeline)")
    print("=" * 50)
    print()
    print("No HTTP server needed — pipeline runs in-process.")
    print("First call will load models (~40s).")
    print()
    print("Starting MCP server...")
    mcp.run()
