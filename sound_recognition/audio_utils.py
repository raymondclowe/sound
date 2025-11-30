"""Audio utility functions."""

import numpy as np
import sounddevice as sd


def play_confirmation_chime():
    """Play a pleasant two-tone confirmation sound (Star Trek computer-style)."""
    sample_rate = 44100
    duration1 = 0.08  # First tone duration
    duration2 = 0.12  # Second tone duration
    gap = 0.02  # Gap between tones
    
    # Generate first tone (higher frequency)
    t1 = np.linspace(0, duration1, int(sample_rate * duration1))
    freq1 = 880  # A5 note
    tone1 = np.sin(2 * np.pi * freq1 * t1)
    # Apply envelope to avoid clicks
    envelope1 = np.exp(-3 * t1 / duration1)
    tone1 = tone1 * envelope1 * 0.3
    
    # Gap
    silence = np.zeros(int(sample_rate * gap))
    
    # Generate second tone (lower frequency)
    t2 = np.linspace(0, duration2, int(sample_rate * duration2))
    freq2 = 659.25  # E5 note
    tone2 = np.sin(2 * np.pi * freq2 * t2)
    # Apply envelope
    envelope2 = np.exp(-2.5 * t2 / duration2)
    tone2 = tone2 * envelope2 * 0.35
    
    # Combine tones
    chime = np.concatenate([tone1, silence, tone2])
    
    # Play the chime
    sd.play(chime, sample_rate)
    sd.wait()


def list_audio_devices():
    """
    List all available audio input devices.
    
    Returns:
        List of device dictionaries from sounddevice
    """
    print("\nAvailable audio input devices:")
    print("="*60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (channels: {device['max_input_channels']})")
    print("="*60 + "\n")
    return devices
