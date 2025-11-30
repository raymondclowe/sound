"""
Sound Recognition Module

A Python module for wake word detection and speech transcription.

This module provides:
- Real-time audio capture and silence detection (SoundBuffer)
- MFCC-based word matching for wake word detection (WordMatcher)
- Integration with external speech-to-text engines (transcribe_audio)
- Audio utility functions (play_confirmation_chime, list_audio_devices)
"""

from .sound_buffer import SoundBuffer
from .word_matcher import WordMatcher
from .transcriber import (
    transcribe_audio,
    resolve_stt_ip,
    create_stt_session,
    benchmark_dns,
    benchmark_post,
    DEFAULT_STT_HOSTNAME,
    DEFAULT_STT_PORT,
)
from .audio_utils import (
    play_confirmation_chime,
    list_audio_devices,
)

__all__ = [
    # Core classes
    'SoundBuffer',
    'WordMatcher',
    # Transcription functions
    'transcribe_audio',
    'resolve_stt_ip',
    'create_stt_session',
    'benchmark_dns',
    'benchmark_post',
    # Audio utilities
    'play_confirmation_chime',
    'list_audio_devices',
    # Configuration constants
    'DEFAULT_STT_HOSTNAME',
    'DEFAULT_STT_PORT',
]

__version__ = '0.1.0'
