from setuptools import setup, find_packages

setup(
    name="eris-voice",
    version="0.1.0",
    description="Eris voice generator using Qwen3-TTS optimized for Apple Silicon",
    author="Eris @ Three Hearts Space",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "qwen-tts>=0.1.0",
        "torch>=2.0.0",
        "soundfile>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "eris-voice=src.eris_voice:main",
        ],
    },
)
