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
import time
import os
import socket
import requests

# Import core functionality from the sound_recognition module
from sound_recognition import (
    SoundBuffer,
    WordMatcher,
    transcribe_audio,
    resolve_stt_ip,
    STT_HOSTNAME,
    STT_PORT,
)

# Resolve STT IP at startup for this POC
STT_IP = resolve_stt_ip()
STT_URL = f"http://{STT_IP}:{STT_PORT}"

# Create a session for connection reuse (for benchmark functions)
stt_session = requests.Session()


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


def test_microphone_level(device_id, duration=3):
    """Test microphone and show live level meter."""
    print(f"\nTesting device {device_id} for {duration} seconds...")
    print("Please speak or make noise!")
    print("Level: ", end="", flush=True)
    
    recorded_data = []
    
    def callback(indata, frames, time, status):
        # Take only first channel from multi-channel input
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        recorded_data.append(audio.copy())
        rms = np.sqrt(np.mean(audio**2))
        max_val = np.max(np.abs(audio))
        # Show visual level meter
        bar_length = int(rms * 100)
        print(f"\rLevel: {'█' * min(bar_length, 50)} RMS={rms:.4f} Max={max_val:.4f}", end="", flush=True)
    
    try:
        dev_info = sd.query_devices(device_id)
        channels = dev_info['max_input_channels']
        with sd.InputStream(samplerate=16000, channels=channels, callback=callback, device=device_id):
            sd.sleep(int(duration * 1000))
        print()  # newline
        
        # Play back what was recorded
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
    """List all available audio input devices."""
    print("\nAvailable audio input devices:")
    print("="*60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (channels: {device['max_input_channels']})")
    print("="*60 + "\n")
    return devices


def benchmark_dns(hostname):
    start = time.time()
    ip = socket.gethostbyname(hostname)
    elapsed = (time.time() - start) * 1000
    print(f"DNS resolution for {hostname}: {ip} in {elapsed:.2f}ms")
    return ip, elapsed


def benchmark_post(stt_url, files, data, use_session=True):
    if use_session:
        sess = stt_session
    else:
        sess = requests
    start = time.time()
    response = sess.post(stt_url, files=files, data=data, timeout=10)
    elapsed = (time.time() - start) * 1000
    print(f"POST to {stt_url} with{'out' if not use_session else ''} session: {elapsed:.2f}ms, status {response.status_code}")
    return response, elapsed


def main():
    # Configuration
    REFERENCE_FILE = "reference_word.wav"
    SIMILARITY_THRESHOLD = 75
    RECORD_REFERENCE = False
    TARGET_WORD = None
    MICROPHONE_DEVICE = None  # None = default, or set to device index
    DEBUG_PLAYBACK = False  # Set to True to hear captured audio before STT
    
    # List available microphones
    list_audio_devices()
    
    # Optionally select microphone
    try:
        device_input = input("Enter microphone device number (or 't' to test, or Enter for default): ").strip()
        if device_input.lower() == 't':
            # Test mode
            test_device = input("Enter device number to test: ").strip()
            if test_device:
                test_audio = test_microphone_level(int(test_device), duration=5)
                response = input("Use this device? (y/n): ").strip().lower()
                if response == 'y':
                    MICROPHONE_DEVICE = int(test_device)
                    print(f"Using device {MICROPHONE_DEVICE}\n")
                else:
                    print("Using default microphone\n")
        elif device_input:
            MICROPHONE_DEVICE = int(device_input)
            print(f"Testing device {MICROPHONE_DEVICE}...")
            test_microphone_level(MICROPHONE_DEVICE, duration=3)
            print(f"Using device {MICROPHONE_DEVICE}\n")
        else:
            print("Using default microphone\n")
    except ValueError:
        print("Invalid input, using default microphone\n")
    
    soundBuffer = SoundBuffer(seconds=10, device=MICROPHONE_DEVICE)
    matcher = WordMatcher(sample_rate=SoundBuffer.FREQUENCY)
    
    # Load or prompt for reference word
    if os.path.exists(REFERENCE_FILE) and not RECORD_REFERENCE:
        matcher.load_reference_from_file(REFERENCE_FILE)
        
        # Transcribe the reference to know what word we're looking for
        print("Transcribing reference word...")
        import soundfile as sf
        ref_audio, _ = sf.read(REFERENCE_FILE)
        TARGET_WORD = transcribe_audio(ref_audio, SoundBuffer.FREQUENCY)
        if TARGET_WORD:
            # Clean up the transcription
            TARGET_WORD = TARGET_WORD.strip().lower().rstrip('.,!?;:')
            print(f"Target word: '{TARGET_WORD}'")
        else:
            print("Warning: Could not transcribe reference word")
        print("Listening for matches...\n")
    else:
        print("\n" + "="*60)
        print("RECORDING REFERENCE WORD")
        print("="*60)
        print("\nInstructions:")
        print("1. Wait for silence detection (you'll see 'Ready to record')")
        print("2. Say your target word clearly")
        print("3. Wait for silence again")
        print("4. You'll hear playback and can accept or retry")
        print("\nWaiting for buffer to fill (10 seconds)...")
        
        # Wait for buffer to fill
        while soundBuffer.samples_collected < soundBuffer.buffer_length:
            sd.sleep(100)
        
        print("\nBuffer ready! Please be quiet for 1 second...")
        
        # Record reference word
        reference_recorded = False
        temp_state = 'waiting'
        temp_silence_start = None
        temp_sound_start = None
        temp_sound_end = None
        
        while not reference_recorded:
            sd.sleep(100)
                
            is_silent = soundBuffer.is_silent()
            current_time = time.time()
            
            if temp_state == 'waiting':
                if is_silent:
                    temp_state = 'in_silence'
                    temp_silence_start = current_time
                    
            elif temp_state == 'in_silence':
                silence_duration = current_time - temp_silence_start
                if silence_duration >= 1.0 and silence_duration < 1.2:
                    print("\n>>> Ready to record! Say your word now...")
                    
                if not is_silent:
                    silence_duration = current_time - temp_silence_start
                    if silence_duration >= 1.0:
                        temp_state = 'in_sound'
                        temp_sound_start = current_time
                        print(">>> Recording...")
                    else:
                        print("  (Not enough silence, waiting...)")
                        temp_state = 'waiting'
                        
            elif temp_state == 'in_sound':
                if is_silent:
                    sound_duration = current_time - temp_sound_start
                    if 0.5 <= sound_duration <= 1.5:
                        temp_state = 'after_sound'
                        temp_sound_end = current_time
                        print(f">>> Word recorded ({sound_duration:.2f}s), verifying silence...")
                    else:
                        print(f"  Word too {'short' if sound_duration < 0.5 else 'long'} ({sound_duration:.2f}s), try again")
                        temp_state = 'waiting'
                elif current_time - temp_sound_start > 1.5:
                    print("  Word too long (>1.5s), try again")
                    temp_state = 'waiting'
                    
            elif temp_state == 'after_sound':
                if is_silent:
                    trailing_silence = current_time - temp_sound_end
                    if trailing_silence >= 1.0:
                        # Extract reference word
                        padding = 0.1
                        extract_start = temp_sound_start - current_time - padding
                        extract_end = temp_sound_end - current_time + padding
                        
                        word_samples_with_padding = soundBuffer.return_last_n_seconds(abs(extract_start))
                        word_end_idx = int((abs(extract_end)) * SoundBuffer.FREQUENCY)
                        reference_audio = word_samples_with_padding[:len(word_samples_with_padding) - word_end_idx]
                        
                        # Play back for confirmation
                        print("\n" + "="*60)
                        print("PLAYBACK - Is this a good sample?")
                        print("="*60)
                        sd.play(reference_audio, SoundBuffer.FREQUENCY)
                        sd.wait()
                        
                        # Ask for confirmation
                        response = input("\nAccept this recording? (y/n): ").strip().lower()
                        if response == 'y':
                            matcher.set_reference(reference_audio, "target_word")
                            
                            # Save reference
                            import soundfile as sf
                            sf.write(REFERENCE_FILE, reference_audio, SoundBuffer.FREQUENCY)
                            
                            # Get transcription
                            print("Getting transcription...")
                            TARGET_WORD = transcribe_audio(reference_audio, SoundBuffer.FREQUENCY)
                            if TARGET_WORD:
                                # Clean up the transcription
                                TARGET_WORD = TARGET_WORD.strip().lower().rstrip('.,!?;:')
                                print(f"Target word: '{TARGET_WORD}'")
                            else:
                                print("Warning: Could not transcribe reference word")
                            
                            reference_recorded = True
                        else:
                            temp_state = 'waiting'
                else:
                    temp_state = 'waiting'
        
        print("="*60)
        print("Listening for matches...")
        print("="*60 + "\n")
    
    # State machine for word detection
    # Start in 'in_silence' state if already silent, otherwise 'waiting'
    # Wait for buffer to be full first
    while soundBuffer.samples_collected < soundBuffer.buffer_length:
        sd.sleep(100)
    
    # Check initial state
    if soundBuffer.is_silent():
        state = 'in_silence'
        silence_start_time = time.time()
    else:
        state = 'waiting'
        silence_start_time = None
        
    sound_start_time = None
    sound_end_time = None
    
    while True:
        sd.sleep(100)
            
        is_currently_silent = soundBuffer.is_silent()
        current_time = time.time()
        
        if state == 'waiting':
            if is_currently_silent:
                state = 'in_silence'
                silence_start_time = current_time
                silence_start_time = current_time
                
        elif state == 'in_silence':
            if not is_currently_silent:
                silence_duration = current_time - silence_start_time
                if silence_duration >= 1.0:
                    state = 'in_sound'
                    sound_start_time = current_time
                else:
                    state = 'waiting'
                    
        elif state == 'in_sound':
            if not is_currently_silent:
                sound_duration = current_time - sound_start_time
                if sound_duration > 1.5:
                    state = 'waiting'
            else:
                sound_duration = current_time - sound_start_time
                if 0.5 <= sound_duration <= 1.5:
                    state = 'after_sound'
                    sound_end_time = current_time
                else:
                    state = 'waiting'
                    
        elif state == 'after_sound':
            if is_currently_silent:
                trailing_silence_duration = current_time - sound_end_time
                if trailing_silence_duration >= 0.5:  # Reduced from 1.0s to 0.5s
                    # Extract the word
                    extract_start_time = time.time()
                    padding = 0.05  # Reduced padding to avoid capturing too much silence
                    extract_start = sound_start_time - current_time - padding
                    extract_end = sound_end_time - current_time + padding
                    
                    word_samples_with_padding = soundBuffer.return_last_n_seconds(abs(extract_start))
                    word_end_idx = int((abs(extract_end)) * SoundBuffer.FREQUENCY)
                    word_audio = word_samples_with_padding[:len(word_samples_with_padding) - word_end_idx]
                    
                    # Skip if audio is too long (likely noise/hallucination)
                    audio_duration = len(word_audio) / SoundBuffer.FREQUENCY
                    if audio_duration > 3.0:  # Skip clips longer than 3 seconds
                        print(f"[Skipped: {audio_duration:.1f}s too long]")
                        state = 'waiting'
                        continue
                    
                    extract_time = time.time() - extract_start_time
                    
                    # Check similarity
                    mfcc_start_time = time.time()
                    matches, similarity = matcher.matches(word_audio, threshold=SIMILARITY_THRESHOLD)
                    mfcc_time = time.time() - mfcc_start_time
                    
                    # Calculate audio stats for debugging
                    audio_rms = np.sqrt(np.mean(word_audio**2))
                    audio_max = np.max(np.abs(word_audio))
                    
                    print(f"[Timing] Extract: {extract_time*1000:.1f}ms, MFCC: {mfcc_time*1000:.1f}ms", end="")
                    print(f" [Audio: RMS={audio_rms:.4f}, Max={audio_max:.4f}, Sim={similarity:.1f}]", end="")
                    
                    if matches:
                        # Optional debug playback
                        if DEBUG_PLAYBACK:
                            print(f" - PLAYING ({audio_duration:.2f}s)...", end="", flush=True)
                            sd.play(word_audio, SoundBuffer.FREQUENCY)
                            sd.wait()
                            print(" DONE", end="")
                        
                        # Send to STT for confirmation
                        stt_start_time = time.time()
                        prompt = f"Wake word: {TARGET_WORD}" if TARGET_WORD else "Wake word: computer"
                        transcription = transcribe_audio(word_audio, SoundBuffer.FREQUENCY, prompt=prompt, model="tiny")
                        stt_time = time.time() - stt_start_time
                        
                        print(f", STT: {stt_time*1000:.1f}ms")
                        
                        if transcription:
                            # Clean up transcription for comparison
                            transcription_clean = transcription.strip().lower().rstrip('.,!?;:')
                            
                            # Check if target word appears in the transcription (allows "ok computer", "computer activate", etc)
                            if TARGET_WORD and TARGET_WORD in transcription_clean.split():
                                print(f"✓ CONFIRMED: '{transcription}'")
                                play_confirmation_chime()
                            else:
                                print(f"? DETECTED: '{transcription}' - expected '{TARGET_WORD}'")
                        else:
                            print(f"? MATCH - STT failed")
                    else:
                        # Optional debug playback for non-matches
                        if DEBUG_PLAYBACK:
                            print(f" - PLAYING NON-MATCH ({audio_duration:.2f}s)...", end="", flush=True)
                            sd.play(word_audio, SoundBuffer.FREQUENCY)
                            sd.wait()
                        print()  # Newline for timing output
                    
                    state = 'waiting'
            else:
                state = 'waiting'


if __name__ == "__main__":
    main()
