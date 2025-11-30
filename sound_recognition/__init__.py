"""
Sound Recognition Module

Core identification and recognition functionality for wake word detection.
Includes:
- SoundBuffer: Audio buffering and silence detection
- WordMatcher: MFCC-based word matching
- transcribe_audio: Speech-to-text integration
"""

import numpy as np
import sounddevice as sd
import time
import librosa
from scipy.spatial.distance import cosine
import base64
import io
import requests
import socket

# --- Static configuration ---
STT_HOSTNAME = "tc3.local"
STT_PORT = 8085


def resolve_stt_ip(hostname=None):
    """Resolve hostname to IP at startup (internal DNS cache).
    
    Args:
        hostname: The hostname to resolve. If None, uses STT_HOSTNAME.
        
    Returns:
        The resolved IP address, or the original hostname if resolution fails.
    """
    if hostname is None:
        hostname = STT_HOSTNAME
    try:
        ip = socket.gethostbyname(hostname)
        print(f"Resolved {hostname} to {ip}")
        return ip
    except Exception as e:
        print(f"Failed to resolve {hostname}: {e}")
        return hostname  # fallback to hostname if resolution fails


# Default STT configuration
_stt_ip = None
_stt_url = None
_stt_session = None


def get_stt_session():
    """Get or create the STT session."""
    global _stt_session
    if _stt_session is None:
        _stt_session = requests.Session()
    return _stt_session


def get_stt_url():
    """Get the STT URL, resolving hostname if needed."""
    global _stt_ip, _stt_url
    if _stt_url is None:
        _stt_ip = resolve_stt_ip()
        _stt_url = f"http://{_stt_ip}:{STT_PORT}"
    return _stt_url


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


class SoundBuffer:
    """
    Circular audio buffer with silence detection.
    
    Provides real-time audio buffering with dynamic silence threshold adjustment.
    """
    
    frame_size = 0
    buffer_length = 0  # in samples
    buffer_seconds = 0
    sd_stream = None
    data = None
    pointer = 0  # current position in buffer (in samples)
    silence_threshold = 0.01
    FREQUENCY = 16000
    MIN_THRESHOLD = 0.005  # Minimum threshold to prevent going to 0
    samples_collected = 0
    
    def __init__(self, seconds=10, device=None):
        self.buffer_seconds = seconds
        self.buffer_length = self.buffer_seconds * SoundBuffer.FREQUENCY
        self.data = np.zeros(self.buffer_length)
        self.samples_collected = 0
        # Determine number of channels for the selected device, with fallback
        channels = 1
        selected_device = device
        if device is not None:
            try:
                dev_info = sd.query_devices(device)
                channels = dev_info['max_input_channels']
                if channels < 1:
                    channels = 1
            except Exception as e:
                print(f"Warning: Could not query device {device}: {e}. Falling back to default device.")
                selected_device = None
                try:
                    dev_info = sd.query_devices(None, 'input')
                    channels = dev_info['max_input_channels']
                    if channels < 1:
                        channels = 1
                except Exception as e2:
                    print(f"Warning: Could not query default input device: {e2}. Using 1 channel.")
                    channels = 1
        self.sd_stream = sd.InputStream(
            samplerate=16000,
            channels=channels,
            callback=self.add_sound_to_buffer,
            device=selected_device
        )
        self.sd_stream.start()

    def stop(self):
        self.sd_stream.stop()
        
    def start(self):
        self.sd_stream.start()
        
    def silent_frames(self):
        """Returns all the frames which by current definitions are silent."""
        if self.frame_size == 0 or len(self.data) < self.frame_size:
            return []
        silent_frames = []
        num_frames = len(self.data) // self.frame_size
        for i in range(num_frames):
            frame = self.data[i*self.frame_size:(i+1)*self.frame_size]
            rms = np.sqrt(np.mean(frame**2))
            if rms < self.silence_threshold:
                silent_frames.append(i)
        return silent_frames
        
    def add_sound_to_buffer(self, indata, frames, time, status):
        # Take only first channel if multi-channel input
        if indata.ndim > 1:
            new_data = indata[:, 0].flatten()
        else:
            new_data = np.array(indata).flatten()
        if self.frame_size == 0:
            self.frame_size = len(new_data)
        
        # Add new data to circular buffer
        for sample in new_data:
            self.data[self.pointer] = sample
            self.pointer = (self.pointer + 1) % self.buffer_length
            if self.samples_collected < self.buffer_length:
                self.samples_collected += 1
            
        # Only start adjusting threshold after we have enough data
        if self.samples_collected < self.buffer_length:
            return
            
        # adjust silence threshold dynamically by setting it to just above the average of silent_frames()
        silent_frames = self.silent_frames()
        if len(silent_frames) > 0:
            silent_values = []
            for i in silent_frames:
                frame = self.data[i*self.frame_size:(i+1)*self.frame_size]
                rms = np.sqrt(np.mean(frame**2))
                silent_values.append(rms)
            new_threshold = np.mean(silent_values) * 1.5  # set threshold to 150% of average silent frame value
            self.silence_threshold = max(new_threshold, self.MIN_THRESHOLD)  # Ensure minimum threshold
        else:
            # If no silent frames found, use a percentile of all frames
            num_frames = len(self.data) // self.frame_size
            all_rms = []
            for i in range(num_frames):
                frame = self.data[i*self.frame_size:(i+1)*self.frame_size]
                rms = np.sqrt(np.mean(frame**2))
                all_rms.append(rms)
            if len(all_rms) > 0:
                # Use 25th percentile as silence threshold
                new_threshold = np.percentile(all_rms, 25) * 1.2
                self.silence_threshold = max(new_threshold, self.MIN_THRESHOLD)
            
    def is_silent(self):
        """Check if the current audio is silent.
        
        Returns:
            bool: True if the current audio level is below the silence threshold,
                  False otherwise.
        """
        if len(self.data) == 0 or self.frame_size == 0:
            return True
        # Get the most recent frame
        recent_samples = self.return_last_n_seconds(0.1)  # Check last 100ms
        if len(recent_samples) == 0:
            return True
        rms = np.sqrt(np.mean(recent_samples**2))
        return rms < self.silence_threshold
    
    def return_last_n_seconds(self, n):
        """Return the last n seconds of audio with wrap around if needed.
        
        Args:
            n: Number of seconds of audio to return (float).
            
        Returns:
            numpy.ndarray: Array of audio samples from the last n seconds.
        """
        n_samples = int(n * SoundBuffer.FREQUENCY)
        if n_samples > len(self.data):
            n_samples = len(self.data)
        if n_samples == 0:
            return np.array([])
        
        start_index = (self.pointer - n_samples) % self.buffer_length
        if start_index < self.pointer:
            return self.data[start_index:self.pointer]
        else:
            return np.concatenate((self.data[start_index:], self.data[:self.pointer]))


class WordMatcher:
    """Matches audio clips using MFCC (Mel-Frequency Cepstral Coefficients) similarity."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.reference_mfcc_mean = None
        self.reference_mfcc_std = None
        self.reference_word = None
        
    def extract_mfcc(self, audio):
        """Extract MFCC features from audio with more discriminative power."""
        # Extract 20 MFCC coefficients (more detailed)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=512, hop_length=160)
        
        # Use both mean and std deviation for better discrimination
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return mfcc_mean, mfcc_std
    
    def set_reference(self, audio, word_name="target"):
        """Set the reference word to match against."""
        self.reference_word = word_name
        self.reference_mfcc_mean, self.reference_mfcc_std = self.extract_mfcc(audio)
        print(f"Reference word '{word_name}' set with MFCC shape: {self.reference_mfcc_mean.shape}")
        
    def load_reference_from_file(self, filepath, word_name="target"):
        """Load reference word from audio file."""
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        self.set_reference(audio, word_name)
        
    def save_reference(self, filepath, audio):
        """Save reference audio to file."""
        import soundfile as sf
        sf.write(filepath, audio, self.sample_rate)
        print(f"Reference saved to {filepath}")
        
    def calculate_similarity(self, audio):
        """
        Calculate similarity between audio and reference word.
        Returns similarity score (0-100, higher is more similar).
        Uses both mean and std of MFCCs for better discrimination.
        """
        if self.reference_mfcc_mean is None:
            raise ValueError("No reference word set. Call set_reference() first.")
        
        # Extract MFCC from candidate audio
        candidate_mfcc_mean, candidate_mfcc_std = self.extract_mfcc(audio)
        
        # Calculate cosine similarity for both mean and std
        sim_mean = 1 - cosine(self.reference_mfcc_mean, candidate_mfcc_mean)
        sim_std = 1 - cosine(self.reference_mfcc_std, candidate_mfcc_std)
        
        # Combine similarities (mean is more important)
        combined_similarity = (sim_mean * 0.7 + sim_std * 0.3)
        
        # Scale to 0-100 for better readability and apply non-linear scaling
        # This spreads out the scores to create more separation
        similarity_percent = combined_similarity * 100
        
        # Apply exponential scaling to amplify differences
        # This makes high scores higher and low scores lower
        scaled_similarity = (similarity_percent ** 1.5) / (100 ** 0.5)
        
        return scaled_similarity
    
    def matches(self, audio, threshold=75):
        """
        Check if audio matches reference word.
        
        Args:
            audio: Audio samples to check
            threshold: Similarity threshold (0-100). Default 75 for good separation.
                      Typical good matches: 85-100
                      Typical non-matches: 60-80
        
        Returns:
            (matches: bool, similarity: float)
        """
        similarity = self.calculate_similarity(audio)
        return similarity >= threshold, similarity


def transcribe_audio(audio_samples, sample_rate=16000, stt_url=None, prompt=None, model="tiny"):
    """
    Send audio to STT engine and get transcription.
    
    Args:
        audio_samples: numpy array of audio samples
        sample_rate: sample rate of audio
        stt_url: URL of the transcription service
        prompt: Optional hint text to guide transcription
        model: Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        transcription text or None if failed
    """
    if stt_url is None:
        stt_url = get_stt_url()
    
    stt_session = get_stt_session()
    
    try:
        prep_start = time.time()
        
        # Normalize and boost audio before sending
        # Remove DC offset
        audio_samples = audio_samples - np.mean(audio_samples)
        
        # Normalize to use full dynamic range
        max_val = np.max(np.abs(audio_samples))
        if max_val > 0:
            audio_samples = audio_samples / max_val
            
        # Apply moderate boost (increase volume by 50%)
        audio_samples = audio_samples * 1.5
        
        # Clip to prevent distortion
        audio_samples = np.clip(audio_samples, -1.0, 1.0)
        
        # Convert audio to WAV format in memory
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio_samples, sample_rate, format='WAV')
        buffer.seek(0)
        
        prep_time = time.time() - prep_start
        
        # Use multipart/form-data for mini_transcriber
        files = {'file': ('audio.wav', buffer, 'audio/wav')}
        data = {
            'model': model,
            'language': 'en'
        }
        if prompt:
            data['initial_prompt'] = prompt
        
        # Send to STT API
        network_start = time.time()
        response = stt_session.post(
            f"{stt_url}/transcribe",
            files=files,
            data=data,
            timeout=10
        )
        network_time = time.time() - network_start
        
        print(f" [Prep: {prep_time*1000:.1f}ms, Network: {network_time*1000:.1f}ms]", end="")
        
        if response.status_code == 200:
            result = response.json()
            # Server reports its own duration
            server_duration = result.get('duration_s', 0)
            if server_duration > 0:
                print(f" [Server model: {server_duration*1000:.1f}ms]", end="")
            text = result.get('text', '').strip()
            if not text:
                print("\n[DEBUG] STT server response had empty or missing 'text' field:")
                print(result)
            return text
        else:
            print(f"STT API error: {response.status_code}")
            if response.text:
                print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None


def benchmark_dns(hostname):
    """Benchmark DNS resolution time."""
    start = time.time()
    ip = socket.gethostbyname(hostname)
    elapsed = (time.time() - start) * 1000
    print(f"DNS resolution for {hostname}: {ip} in {elapsed:.2f}ms")
    return ip, elapsed


def benchmark_post(stt_url, files, data, use_session=True):
    """Benchmark HTTP POST request time."""
    if use_session:
        sess = get_stt_session()
    else:
        sess = requests
    start = time.time()
    response = sess.post(stt_url, files=files, data=data, timeout=10)
    elapsed = (time.time() - start) * 1000
    print(f"POST to {stt_url} with{'out' if not use_session else ''} session: {elapsed:.2f}ms, status {response.status_code}")
    return response, elapsed


# Public API
__all__ = [
    'SoundBuffer',
    'WordMatcher',
    'transcribe_audio',
    'play_confirmation_chime',
    'test_microphone_level',
    'list_audio_devices',
    'resolve_stt_ip',
    'get_stt_url',
    'get_stt_session',
    'benchmark_dns',
    'benchmark_post',
    'STT_HOSTNAME',
    'STT_PORT',
]
