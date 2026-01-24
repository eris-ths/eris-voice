#!/usr/bin/env python3
"""
Eris Voice HTTP Server

Persistent TTS server with pre-warmed models for low-latency responses.
Eliminates cold-start overhead by keeping models loaded in memory.

Usage:
    # Start server
    python eris_voice_server.py

    # Generate speech
    curl -X POST http://localhost:8765/speak \
        -H "Content-Type: application/json" \
        -d '{"text": "こんにちは", "speaker": "ono_anna"}'
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

os.environ["OMP_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Global pipeline instance
_pipeline = None
_warmup_done = False


def get_pipeline():
    """Get or create the TTS pipeline."""
    global _pipeline
    if _pipeline is None:
        from streaming_prototype import StreamingTTSPipeline
        _pipeline = StreamingTTSPipeline()
        _pipeline.load()
    return _pipeline


def warmup_pipeline():
    """Warmup the pipeline with a short generation."""
    global _warmup_done
    if _warmup_done:
        return

    print("Warming up pipeline...")
    pipeline = get_pipeline()
    start = time.time()

    # Generate a short phrase to warm up all components
    _ = pipeline.generate_sentence("テスト", speaker="ono_anna")

    print(f"Warmup complete in {time.time() - start:.2f}s")
    _warmup_done = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup/shutdown."""
    # Startup: load and warmup
    print("Starting Eris Voice Server...")
    warmup_pipeline()
    print("Server ready!")
    yield
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Eris Voice Server",
    description="MLX-accelerated TTS for Apple Silicon",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response models
class SpeakRequest(BaseModel):
    """Request for speech generation."""
    text: str = Field(..., min_length=1, max_length=500, description="Text to synthesize")
    speaker: str = Field(default="ono_anna", description="Speaker preset")
    play: bool = Field(default=False, description="Play audio immediately")
    save: bool = Field(default=True, description="Save to file and return path")


class SpeakResponse(BaseModel):
    """Response from speech generation."""
    success: bool
    text: str
    speaker: str
    duration_seconds: float
    generation_time_seconds: float
    realtime_factor: float
    audio_path: Optional[str] = None


class StreamRequest(BaseModel):
    """Request for streaming speech generation."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    speaker: str = Field(default="ono_anna", description="Speaker preset")
    play_as_generated: bool = Field(default=True, description="Play each sentence as generated")


class StreamResponse(BaseModel):
    """Response from streaming generation."""
    success: bool
    sentences: list
    total_duration_seconds: float
    time_to_first_audio: float
    total_generation_time: float
    audio_path: Optional[str] = None


class StatusResponse(BaseModel):
    """Server status response."""
    loaded: bool
    warmed_up: bool
    model: str = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    optimizations: list
    sample_rate: int = 24000


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Eris Voice Server", "status": "running"}


@app.get("/status", response_model=StatusResponse)
async def status():
    """Get server status."""
    return StatusResponse(
        loaded=_pipeline is not None,
        warmed_up=_warmup_done,
        optimizations=[
            "MLX Audio Decoder (45x speedup)",
            "MLX Quantizer (3.5x speedup)",
            "Persistent Server (no cold start)",
        ],
    )


@app.post("/speak", response_model=SpeakResponse)
async def speak(request: SpeakRequest):
    """Generate speech from text."""
    try:
        pipeline = get_pipeline()

        start = time.time()
        audio = pipeline.generate_sentence(request.text, speaker=request.speaker)
        generation_time = time.time() - start

        duration = len(audio) / 24000
        rtf = duration / generation_time

        audio_path = None
        if request.save:
            # Save to temp file
            fd, audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(audio_path, audio, 24000)

        if request.play:
            # Play audio
            if audio_path:
                subprocess.Popen(
                    ["afplay", audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(temp_path, audio, 24000)
                subprocess.Popen(
                    ["afplay", temp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        return SpeakResponse(
            success=True,
            text=request.text,
            speaker=request.speaker,
            duration_seconds=duration,
            generation_time_seconds=generation_time,
            realtime_factor=rtf,
            audio_path=audio_path,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream", response_model=StreamResponse)
async def stream(request: StreamRequest):
    """Generate speech with sentence-level streaming."""
    try:
        pipeline = get_pipeline()

        sentences = []
        all_audio = []
        total_start = time.time()
        ttfa = None

        for audio, sentence_text, elapsed in pipeline.generate_streaming(
            request.text,
            speaker=request.speaker,
            play_immediately=request.play_as_generated,
        ):
            if ttfa is None:
                ttfa = elapsed

            duration = len(audio) / 24000
            sentences.append({
                "text": sentence_text,
                "duration": duration,
                "time": elapsed,
            })
            all_audio.append(audio)

        total_time = time.time() - total_start

        # Combine all audio
        combined = np.concatenate(all_audio)
        total_duration = len(combined) / 24000

        # Save combined audio
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(audio_path, combined, 24000)

        return StreamResponse(
            success=True,
            sentences=sentences,
            total_duration_seconds=total_duration,
            time_to_first_audio=ttfa or 0,
            total_generation_time=total_time,
            audio_path=audio_path,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers")
async def list_speakers():
    """List available speakers."""
    return {
        "speakers": [
            {"id": "ono_anna", "name": "Ono Anna", "language": "Japanese", "description": "Playful female"},
            {"id": "vivian", "name": "Vivian", "language": "Chinese", "description": "Bright, edgy female"},
            {"id": "serena", "name": "Serena", "language": "Chinese", "description": "Warm, gentle female"},
            {"id": "ryan", "name": "Ryan", "language": "English", "description": "Dynamic male"},
            {"id": "aiden", "name": "Aiden", "language": "English", "description": "Sunny American male"},
            {"id": "sohee", "name": "Sohee", "language": "Korean", "description": "Warm female"},
        ]
    }


if __name__ == "__main__":
    print("=" * 50)
    print("Eris Voice Server")
    print("=" * 50)
    print()
    print("Starting on http://localhost:8765")
    print("Endpoints:")
    print("  POST /speak   - Generate speech")
    print("  POST /stream  - Streaming generation")
    print("  GET  /status  - Server status")
    print("  GET  /speakers - List speakers")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
