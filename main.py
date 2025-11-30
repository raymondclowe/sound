"""
Sound Wake Word Detection - Proof of Concept

This POC demonstrates the sound_recognition module capabilities including:
- Real-time audio capture and silence detection
- MFCC-based word matching for wake word detection
- Integration with external speech-to-text (STT) engines

The core identification and recognition logic is in the sound_recognition module.
"""




import numpy as np
import sounddevice as sd
import sys
from sound_recognition.recogniser import Recogniser

# Resolve STT IP at startup for this POC

# Create a session for connection reuse (for benchmark functions)



def play_confirmation_chime():
    sample_rate = 44100
    duration1 = 0.08
    duration2 = 0.12
    gap = 0.02
    t1 = np.linspace(0, duration1, int(sample_rate * duration1))
    freq1 = 880
    tone1 = np.sin(2 * np.pi * freq1 * t1)
    envelope1 = np.exp(-3 * t1 / duration1)
    tone1 = tone1 * envelope1 * 0.3
    silence = np.zeros(int(sample_rate * gap))
    t2 = np.linspace(0, duration2, int(sample_rate * duration2))
    freq2 = 659.25
    tone2 = np.sin(2 * np.pi * freq2 * t2)
    envelope2 = np.exp(-2.5 * t2 / duration2)
    tone2 = tone2 * envelope2 * 0.35
    chime = np.concatenate([tone1, silence, tone2])
    sd.play(chime, sample_rate)
    sd.wait()



def test_microphone_level(device_id, duration=3):
    print(f"\nTesting device {device_id} for {duration} seconds...")
    print("Please speak or make noise!")
    print("Level: ", end="", flush=True)
    recorded_data = []
    def callback(indata, frames, time, status):
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        recorded_data.append(audio.copy())
        rms = np.sqrt(np.mean(audio**2))
        max_val = np.max(np.abs(audio))
        bar_length = int(rms * 100)
        print(f"\rLevel: {'█' * min(bar_length, 50)} RMS={rms:.4f} Max={max_val:.4f}", end="", flush=True)
    try:
        dev_info = sd.query_devices(device_id)
        # Use .get for dict access to avoid type errors
        if isinstance(dev_info, dict):
            channels = int(dev_info.get('max_input_channels', 1))
        else:
            channels = int(getattr(dev_info, 'max_input_channels', 1))
        with sd.InputStream(samplerate=16000, channels=channels, callback=callback, device=device_id):
            sd.sleep(int(duration * 1000))
        print()
        if recorded_data:
            all_audio = np.concatenate(recorded_data)
            print(f"\nPlayback of recorded audio ({len(all_audio)/16000:.1f}s)...")
            sd.play(all_audio, 16000)
            sd.wait()
            print("Playback complete.\n")
            return all_audio
    except Exception as e:
        print(f"\nError testing device: {e}")
        return None



def list_audio_devices():
    print("\nAvailable audio input devices:")
    print("="*60)
    devices = sd.query_devices()
    found = False
    for i, device in enumerate(devices):
        try:
            max_in_val = int(device.get('max_input_channels', 0)) if isinstance(device, dict) else int(getattr(device, 'max_input_channels', 0))
        except Exception:
            continue
        if max_in_val > 0:
            found = True
            name = device.get('name', str(i)) if isinstance(device, dict) else getattr(device, 'name', str(i))
            print(f"{i}: {name} (channels: {max_in_val})")
    if not found:
        print("[DEBUG] No input devices found with >0 channels. Full device list:")
        for i, device in enumerate(devices):
            print(f"{i}: {device}")
    print("="*60 + "\n")
    return devices





def main():
    # list_audio_devices()
    # try:
    #     device_input = input("Enter microphone device number (or 't' to test, or Enter for default): ").strip()
    #     if device_input.lower() == 't':
    #         test_device = input("Enter device number to test: ").strip()
    #         if test_device:
    #             test_audio = test_microphone_level(int(test_device), duration=5)
    #             response = input("Use this device? (y/n): ").strip().lower()
    #             if response == 'y':
    #                 MICROPHONE_DEVICE = int(test_device)
    #                 print(f"Using device {MICROPHONE_DEVICE}\n")
    #             else:
    #                 MICROPHONE_DEVICE = None
    #                 print("Using default microphone\n")
    #     elif device_input:
    #         MICROPHONE_DEVICE = int(device_input)
    #         print(f"Testing device {MICROPHONE_DEVICE}...")
    #         test_microphone_level(MICROPHONE_DEVICE, duration=3)
    #         print(f"Using device {MICROPHONE_DEVICE}\n")
    #     else:
    #         MICROPHONE_DEVICE = None
    #         print("Using default microphone\n")
    # except ValueError:
    #     MICROPHONE_DEVICE = None
    #     print("Invalid input, using default microphone\n")

    # Create recogniser object
    recogniser = Recogniser(
        wakewordreferenceaudio=["reference_word.wav", "reference_word_male.wav"],  # List of reference audio files
        wakewordstring="computer",  # Target wake word (required for logic)
        threshold=75,  # MFCC similarity threshold (0-100)
        device=int(1),  # Audio input device index
        debug='--debug' in sys.argv,  # Enable debug output if '--debug' in args
        debug_playback=False,  # Play back detected audio for debugging
        min_silence_before=1.0,  # Minimum silence before word (seconds)
        min_sound=0.5,           # Minimum sound duration (seconds)
        max_sound=1.5,           # Maximum sound duration (seconds)
        min_trailing_silence=0.5, # Minimum silence after word (seconds)
        max_audio_duration=3.0,  # Maximum allowed audio duration (seconds)
        padding=0.05,            # Padding (seconds) before/after detected word
        # Acceptable min/max number of words in phrase
        allowed_other_words=[    # List of allowed extra words (case-insensitive, alphanumeric only)
            "ok", "okay", "activate", "please", "hey", "hello", "hi", "greetings",
            "excuse", "pardon", "listen", "attention", "wake", "start",
            "ready", "go", "begin", "now", "dear", "good"
        ]
        # whisperurl=None,       # (Optional) URL for STT server
        # whispermodel="tiny",  # (Optional) Whisper model name
        # languages=None,        # (Optional) List of allowed languages
        # timeout=30,            # (Optional) STT timeout in seconds
    )

    print("Listening for wake word... (Ctrl+C to exit)")
    while True:
        result = recogniser.waitforit()
        if result is not None:
            play_confirmation_chime()


if __name__ == "__main__":
    main()
