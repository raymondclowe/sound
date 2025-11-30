# Sound Wake Word Detection Project

## Project Status (as of November 27, 2025)

This project is a Python-based wake word detection and speech transcription tool. It uses:
- Real-time audio capture and silence detection
- MFCC-based word matching for wake word detection
- Integration with external speech-to-text (STT) engines (mini_transcriber and Whisper.cpp)

## Module Structure

The core functionality is provided by the `sound_recognition` module:

```
sound_recognition/
├── __init__.py          # Public API exports
├── sound_buffer.py      # SoundBuffer class for audio capture and silence detection
├── word_matcher.py      # WordMatcher class for MFCC-based word matching
├── transcriber.py       # STT transcription functions
└── audio_utils.py       # Audio utility functions (chimes, device listing)
```

### Using the Module

```python
from sound_recognition import (
    SoundBuffer,           # Audio capture and silence detection
    WordMatcher,           # MFCC-based word matching
    transcribe_audio,      # Speech-to-text transcription
    resolve_stt_ip,        # DNS resolution helper
    create_stt_session,    # HTTP session factory
    play_confirmation_chime,  # Audio feedback
    list_audio_devices,    # Device enumeration
)
```

### Demo Application

The `main.py` file provides a proof-of-concept demonstration of the module's capabilities.

### Current Features
- **Audio Buffering & Silence Detection**: Efficiently captures and segments audio for wake word detection.
- **MFCC Word Matching**: Uses Mel-Frequency Cepstral Coefficients to match spoken words against a reference.
- **STT Integration**: Supports both mini_transcriber and Whisper.cpp endpoints for transcription.
- **Network Optimization**: Now uses a cached IP address for `.local` hostnames to avoid repeated slow mDNS lookups.
- **Session Reuse**: HTTP requests use a persistent session for faster repeated calls.
- **Benchmarking Tools**: Includes utilities to benchmark DNS resolution and POST request times.

### Key Findings
- `.local` hostnames (mDNS) can cause significant delays (2+ seconds per lookup) if not cached. Using a static IP or hosts file is much faster.
- Python's `requests.Session` improves HTTP request speed by reusing connections.
- The code now resolves the STT server IP once at startup and uses it for all requests, eliminating repeated DNS delays.

### How to Use
1. Configure your reference word and run the main script.
2. The system will listen for the wake word and confirm with a chime when detected.
3. Transcription is performed using the selected STT backend.

### Next Steps
- Further optimize audio preprocessing and detection.
- Add more robust error handling and logging.
- Explore additional STT backends or on-device models.

---

For more details, see the code and comments in `main.py`.
