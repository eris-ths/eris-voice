#!/usr/bin/env python3
"""
Eris Voice MCP Server

MCP server that connects to the persistent HTTP server for low-latency TTS.
The HTTP server keeps models loaded, eliminating cold-start overhead.

Architecture:
    Claude Code → MCP Server (this file) → HTTP Server (eris_voice_server.py)
                                          ↓
                                    Pre-loaded models

Usage:
    # First, start the HTTP server in a separate terminal:
    python eris_voice_server.py

    # Then use this MCP server normally
    # (configured in ~/.claude.json)
"""

import os
import sys
import json
import subprocess
import time
from typing import Optional
from enum import Enum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("eris_voice_mcp")

# HTTP server configuration
HTTP_SERVER_URL = os.environ.get("ERIS_VOICE_SERVER_URL", "http://localhost:8765")
HTTP_TIMEOUT = 120.0  # seconds


async def check_server_status() -> bool:
    """Check if HTTP server is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HTTP_SERVER_URL}/status", timeout=5.0)
            return response.status_code == 200
    except:
        return False


async def ensure_server_running():
    """Ensure HTTP server is running, provide instructions if not."""
    if not await check_server_status():
        return {
            "error": "HTTP server not running",
            "message": "Please start the Eris Voice server first:",
            "command": "cd /Users/hirohashi/Develop/three_hearts_space/_nao/work/qwen3_tts_eris_voice/src && python eris_voice_server.py",
            "hint": "Run this in a separate terminal window",
        }
    return None


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
    PLAY = "play"
    FILE = "file"
    BOTH = "both"


class QualityMode(str, Enum):
    """Quality mode for generation speed/quality tradeoff."""
    HIGH = "high"           # 15 codebooks, RTF ~0.82x, best quality
    BALANCED = "balanced"   # 11 codebooks, RTF ~1.0x, good quality (default)
    FAST = "fast"           # 7 codebooks, RTF ~1.4x, acceptable quality
    ULTRA_FAST = "ultra_fast"  # 3 codebooks, RTF ~2.0x, reduced quality


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
    quality_mode: QualityMode = Field(
        default=QualityMode.BALANCED,
        description="Quality mode: 'high' (best quality, slower), 'balanced' (default, realtime), 'fast' (faster), 'ultra_fast' (fastest)",
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
    quality_mode: QualityMode = Field(
        default=QualityMode.BALANCED,
        description="Quality mode for generation",
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
        - Fast mode: eris_speak(text="速い", quality_mode="fast")
    """
    # Check server status
    error = await ensure_server_running()
    if error:
        return json.dumps(error, ensure_ascii=False)

    try:
        # Prepare request
        request_data = {
            "text": params.text,
            "speaker": params.speaker.value,
            "play": params.output_mode in (OutputMode.PLAY, OutputMode.BOTH),
            "save": params.output_mode in (OutputMode.FILE, OutputMode.BOTH),
            "quality_mode": params.quality_mode.value,
        }

        # Call HTTP server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{HTTP_SERVER_URL}/speak",
                json=request_data,
                timeout=HTTP_TIMEOUT,
            )
            result = response.json()

        return json.dumps(result, ensure_ascii=False)

    except httpx.TimeoutException:
        return json.dumps({
            "success": False,
            "error": "Request timed out",
            "hint": "The text may be too long, try shorter input",
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
    # Check server status
    error = await ensure_server_running()
    if error:
        return json.dumps(error, ensure_ascii=False)

    try:
        request_data = {
            "text": params.text,
            "speaker": params.speaker.value,
            "play_as_generated": params.play_as_generated,
            "quality_mode": params.quality_mode.value,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{HTTP_SERVER_URL}/stream",
                json=request_data,
                timeout=HTTP_TIMEOUT,
            )
            result = response.json()

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        }, ensure_ascii=False)


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
            "description": "Playful Japanese female voice",
        },
        {
            "id": "vivian",
            "name": "Vivian",
            "language": "Chinese",
            "description": "Bright, edgy young female voice",
        },
        {
            "id": "serena",
            "name": "Serena",
            "language": "Chinese",
            "description": "Warm, gentle young female voice",
        },
        {
            "id": "ryan",
            "name": "Ryan",
            "language": "English",
            "description": "Dynamic male voice",
        },
        {
            "id": "aiden",
            "name": "Aiden",
            "language": "English",
            "description": "Sunny American male voice",
        },
        {
            "id": "sohee",
            "name": "Sohee",
            "language": "Korean",
            "description": "Warm female voice with rich emotion",
        },
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
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HTTP_SERVER_URL}/status", timeout=5.0)
            if response.status_code == 200:
                result = response.json()
                result["server_running"] = True
                return json.dumps(result, ensure_ascii=False)
    except:
        pass

    return json.dumps({
        "server_running": False,
        "loaded": False,
        "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "optimizations": [
            "MLX Audio Decoder (45x speedup)",
            "MLX Quantizer (3.5x speedup)",
            "MLX Generate Loop (166x speedup)",
            "Codebook Reduction (quality_mode)",
            "Sentence-level Streaming (2.7x faster TTFA)",
        ],
        "quality_presets": {
            "high": "15 codebooks, RTF ~0.82x, best quality",
            "balanced": "11 codebooks, RTF ~1.0x (default)",
            "fast": "7 codebooks, RTF ~1.4x",
            "ultra_fast": "3 codebooks, RTF ~2.0x",
        },
        "sample_rate": 24000,
        "ready": False,
        "note": "Start the HTTP server first: python eris_voice_server.py",
    }, ensure_ascii=False)


if __name__ == "__main__":
    print("=" * 50)
    print("Eris Voice MCP Server")
    print("=" * 50)
    print()
    print("This MCP server connects to a persistent HTTP server.")
    print("Make sure to start the HTTP server first:")
    print()
    print("  python eris_voice_server.py")
    print()
    print("Starting MCP server...")
    mcp.run()
