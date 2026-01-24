#!/usr/bin/env python3
"""
Eris Voice MCP Server

MLX-accelerated Text-to-Speech for Claude Code integration.
Provides tools for generating and playing speech using Qwen3-TTS with MLX optimization.

Usage:
    # Run as MCP server (stdio)
    python eris_voice_mcp.py

    # Test with MCP Inspector
    npx @modelcontextprotocol/inspector python eris_voice_mcp.py
"""

import os
import sys
import json
import subprocess
import tempfile
import time
from typing import Optional, List
from enum import Enum
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("eris_voice_mcp")

# Global pipeline instance (lazy loaded)
_pipeline = None


def get_pipeline():
    """Lazy load the TTS pipeline."""
    global _pipeline
    if _pipeline is None:
        import torch
        torch.set_num_threads(8)

        from streaming_prototype import StreamingTTSPipeline
        _pipeline = StreamingTTSPipeline()
        _pipeline.load()
    return _pipeline


# Enums
class Speaker(str, Enum):
    """Available speaker presets."""
    ONO_ANNA = "ono_anna"
    VIVIAN = "vivian"
    SERENA = "serena"
    RYAN = "ryan"
    AIDEN = "aiden"
    SOHEE = "sohee"


class OutputMode(str, Enum):
    """Output mode for generated audio."""
    PLAY = "play"  # Play immediately
    FILE = "file"  # Return file path only
    BOTH = "both"  # Play and return file path


# Input Models
class SpeakInput(BaseModel):
    """Input for speak tool."""
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


class StreamingSpeakInput(BaseModel):
    """Input for streaming speak tool."""
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
    Generate speech from text using MLX-accelerated Qwen3-TTS.

    This tool converts text to speech using the Eris Voice pipeline with
    14.8x speedup over CPU PyTorch. Audio can be played immediately,
    saved to a file, or both.

    Args:
        params (SpeakInput): Input parameters containing:
            - text (str): Text to synthesize (1-500 characters)
            - speaker (Speaker): Voice preset (default: ono_anna)
            - output_mode (OutputMode): play/file/both
            - output_dir (Optional[str]): Directory for output file

    Returns:
        str: JSON response with generation info:
        {
            "success": true,
            "text": "synthesized text",
            "speaker": "ono_anna",
            "duration_seconds": 2.5,
            "generation_time_seconds": 1.2,
            "realtime_factor": 2.08,
            "audio_path": "/path/to/audio.wav" (if file mode)
        }

    Examples:
        - Generate and play: eris_speak(text="こんにちは")
        - Save to file: eris_speak(text="Hello", output_mode="file")
        - Japanese voice: eris_speak(text="私はエリスよ", speaker="ono_anna")
    """
    import numpy as np
    import soundfile as sf

    try:
        pipeline = get_pipeline()

        start_time = time.time()
        audio = pipeline.generate_sentence(params.text, speaker=params.speaker.value)
        generation_time = time.time() - start_time

        duration = len(audio) / pipeline.sample_rate
        realtime_factor = duration / generation_time if generation_time > 0 else 0

        # Determine output path
        audio_path = None
        if params.output_mode in (OutputMode.FILE, OutputMode.BOTH):
            if params.output_dir:
                output_dir = Path(params.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(tempfile.gettempdir())

            timestamp = int(time.time() * 1000)
            audio_path = str(output_dir / f"eris_voice_{timestamp}.wav")
            sf.write(audio_path, audio, pipeline.sample_rate)

        # Play audio
        if params.output_mode in (OutputMode.PLAY, OutputMode.BOTH):
            if audio_path:
                subprocess.Popen(
                    ["afplay", audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # Create temp file for playback
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, pipeline.sample_rate)
                    subprocess.Popen(
                        ["afplay", f.name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

        response = {
            "success": True,
            "text": params.text,
            "speaker": params.speaker.value,
            "duration_seconds": round(duration, 2),
            "generation_time_seconds": round(generation_time, 2),
            "realtime_factor": round(realtime_factor, 2),
        }

        if audio_path:
            response["audio_path"] = audio_path

        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "text": params.text,
        }, indent=2, ensure_ascii=False)


@mcp.tool(
    name="eris_speak_streaming",
    annotations={
        "title": "Eris TTS Streaming Speak",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def eris_speak_streaming(params: StreamingSpeakInput) -> str:
    """
    Generate speech with sentence-level streaming for faster time-to-first-audio.

    This tool splits the input text into sentences and generates audio for
    each sentence sequentially. When play_as_generated is True, audio starts
    playing as soon as the first sentence is ready (2.7x faster TTFA).

    Args:
        params (StreamingSpeakInput): Input parameters containing:
            - text (str): Text to synthesize (can be multiple sentences)
            - speaker (Speaker): Voice preset
            - play_as_generated (bool): Stream playback as generated

    Returns:
        str: JSON response with streaming generation info:
        {
            "success": true,
            "sentences": [
                {"text": "sentence1", "duration": 1.5, "time": 2.1},
                {"text": "sentence2", "duration": 2.0, "time": 4.5}
            ],
            "total_duration_seconds": 3.5,
            "time_to_first_audio": 2.1,
            "total_generation_time": 4.5,
            "audio_path": "/path/to/combined.wav"
        }

    Examples:
        - Streaming: eris_speak_streaming(text="こんにちは。私はエリスよ。よろしくね。")
        - No streaming: eris_speak_streaming(text="Hello.", play_as_generated=False)
    """
    import numpy as np
    import soundfile as sf

    try:
        pipeline = get_pipeline()

        sentences_info = []
        all_audio = []
        first_audio_time = None

        for audio, sentence, elapsed in pipeline.generate_streaming(
            params.text,
            speaker=params.speaker.value,
            play_immediately=params.play_as_generated,
        ):
            if first_audio_time is None:
                first_audio_time = elapsed

            duration = len(audio) / pipeline.sample_rate
            all_audio.append(audio)

            sentences_info.append({
                "text": sentence,
                "duration_seconds": round(duration, 2),
                "elapsed_seconds": round(elapsed, 2),
            })

        # Combine and save
        combined_audio = np.concatenate(all_audio)
        total_duration = len(combined_audio) / pipeline.sample_rate

        timestamp = int(time.time() * 1000)
        audio_path = str(Path(tempfile.gettempdir()) / f"eris_streaming_{timestamp}.wav")
        sf.write(audio_path, combined_audio, pipeline.sample_rate)

        response = {
            "success": True,
            "sentences": sentences_info,
            "total_duration_seconds": round(total_duration, 2),
            "time_to_first_audio_seconds": round(first_audio_time, 2) if first_audio_time else None,
            "total_generation_time_seconds": round(sentences_info[-1]["elapsed_seconds"], 2) if sentences_info else 0,
            "audio_path": audio_path,
        }

        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "text": params.text,
        }, indent=2, ensure_ascii=False)


@mcp.tool(
    name="eris_list_speakers",
    annotations={
        "title": "List Available Speakers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def eris_list_speakers() -> str:
    """
    List all available speaker presets for Eris Voice.

    Returns information about each speaker including their name,
    recommended language, and description.

    Returns:
        str: JSON array of speaker information:
        [
            {
                "id": "ono_anna",
                "name": "Ono Anna",
                "language": "Japanese",
                "description": "Playful Japanese female voice"
            },
            ...
        ]
    """
    speakers = [
        {
            "id": "ono_anna",
            "name": "Ono Anna",
            "language": "Japanese",
            "description": "Playful Japanese female voice (recommended for Japanese text)",
            "recommended": True,
        },
        {
            "id": "vivian",
            "name": "Vivian",
            "language": "Chinese",
            "description": "Bright, edgy young female voice",
            "recommended": False,
        },
        {
            "id": "serena",
            "name": "Serena",
            "language": "Chinese",
            "description": "Warm, gentle young female voice",
            "recommended": False,
        },
        {
            "id": "ryan",
            "name": "Ryan",
            "language": "English",
            "description": "Dynamic male voice",
            "recommended": False,
        },
        {
            "id": "aiden",
            "name": "Aiden",
            "language": "English",
            "description": "Sunny American male voice",
            "recommended": False,
        },
        {
            "id": "sohee",
            "name": "Sohee",
            "language": "Korean",
            "description": "Warm female voice with rich emotion",
            "recommended": False,
        },
    ]

    return json.dumps(speakers, indent=2, ensure_ascii=False)


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
    """
    Get the current status of the Eris Voice pipeline.

    Returns information about whether the model is loaded,
    available optimizations, and system info.

    Returns:
        str: JSON status object:
        {
            "loaded": true,
            "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "optimizations": ["MLX Decoder (45x)", "MLX Quantizer (3.5x)"],
            "sample_rate": 24000
        }
    """
    global _pipeline

    status = {
        "loaded": _pipeline is not None,
        "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "optimizations": [
            "MLX Audio Decoder (45x speedup)",
            "MLX Quantizer (3.5x speedup)",
            "Sentence-level Streaming (2.7x faster TTFA)",
        ],
        "sample_rate": 24000,
        "supported_languages": ["Japanese", "Chinese", "English", "Korean"],
    }

    if _pipeline is not None:
        status["ready"] = True
    else:
        status["ready"] = False
        status["note"] = "Model will be loaded on first use"

    return json.dumps(status, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
