"""
Sound Recognition Module

A reusable module for audio identification and recognition, including:
- Silence detection and audio buffering
- MFCC-based word matching (TFF - Temporal Feature Fingerprinting)
- Speech-to-text transcription integration

This module is designed to be self-contained and suitable for distribution
as a standalone PyPI package.

Usage:
    from sound_recognition import SoundBuffer, WordMatcher, transcribe_audio
"""

from .silence import SoundBuffer
from .matching import WordMatcher
from .recogniser import Recogniser
from .transcription import (
    transcribe_audio,
    resolve_stt_ip,
    STT_HOSTNAME,
    STT_PORT,
)

__version__ = "0.1.0"
__all__ = [
    "SoundBuffer",
    "WordMatcher",
    "Recogniser",
    "transcribe_audio",
    "resolve_stt_ip",
    "STT_HOSTNAME",
    "STT_PORT",
]
