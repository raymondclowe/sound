"""
Sound Wake Word Detection - Proof of Concept

This POC demonstrates the easywakeword module capabilities including:
- Real-time audio capture and silence detection
- MFCC-based word matching for wake word detection
- Integration with external speech-to-text (STT) engines

The core identification and recognition logic is in the easywakeword module.
"""




import numpy as np
import sounddevice as sd
import sys
import os
from easywakeword import Recogniser

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



from easywakeword.silence import list_audio_devices, test_microphone_level



# list_audio_devices moved to easywakeword.silence





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

    # Try to find the real WAV files in the project root (not examples/)
    wav_files = [
        "example_computer_male.wav",
        "example_computer_female.wav",
        "example_computer_male_teen.wav"
    ]
    wav_paths = [os.path.join(os.path.dirname(__file__), f) for f in wav_files]
    missing = [p for p in wav_paths if not os.path.isfile(p)]
    if missing:
        print("[DEMO ERROR] One or more required WAV files are missing:")
        for p in missing:
            print(f"  Missing: {p}")
        print("Please copy the real example_computer_*.wav files to the project root.")
        sys.exit(1)

    recogniser = Recogniser(
        wakewordstrings=["computer", "computer", "computer"],
        wakewordreferenceaudios=wav_paths,
        threshold=75,
        device=int(1),
        debug='--debug' in sys.argv,
        debug_playback=False,
        allowed_other_words=["ok", "okay", "activate", "please", "hey", "hello", "hi"],
        auto_sound_from_reference=True,
    )

    print("Listening for wake word... (Ctrl+C to exit)")
    while True:
        result = recogniser.waitforit()
        if result is not None:
            play_confirmation_chime()


if __name__ == "__main__":
    main()
