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



from sound_recognition.silence import list_audio_devices, test_microphone_level



# list_audio_devices moved to sound_recognition.silence





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
        wakewordstrings=["computer", "computer"],
        wakewordreferenceaudios=["example_computer_male.wav", "example_computer_female..wav"],
        threshold=75,
        device=int(1),
        debug='--debug' in sys.argv,
        debug_playback=False
    )

    print("Listening for wake word... (Ctrl+C to exit)")
    while True:
        result = recogniser.waitforit()
        if result is not None:
            play_confirmation_chime()


if __name__ == "__main__":
    main()
