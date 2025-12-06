"""
Generate example wave files for the project (silent example files for testing).
This script creates three example wavs under `examples/`:
- example_computer_male.wav
- example_computer_female.wav
- example_computer_male_teen.wav
"""
import os
import numpy as np
import soundfile as sf


def generate_sine_wave(filename: str, duration=1.0, sample_rate=16000, freq=440.0):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.1 * np.sin(2 * np.pi * freq * t)
    sf.write(filename, tone, sample_rate)


def main():
    os.makedirs("examples", exist_ok=True)
    generate_sine_wave("examples/example_computer_male.wav", duration=1.0, freq=220.0)
    generate_sine_wave("examples/example_computer_female.wav", duration=1.0, freq=440.0)
    generate_sine_wave("examples/example_computer_male_teen.wav", duration=1.0, freq=330.0)
    print("Created example WAVs in examples/")


if __name__ == "__main__":
    main()
