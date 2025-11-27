# Sound Wake Word Detection Project

## Project Status (as of November 27, 2025)

This project is a Python-based wake word detection and speech transcription tool. It uses:
- Real-time audio capture and silence detection
- MFCC-based word matching for wake word detection
- Integration with external speech-to-text (STT) engines (mini_transcriber and Whisper.cpp)

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
