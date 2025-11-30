"""Transcriber module for speech-to-text functionality."""

import numpy as np
import time
import io
import requests
import socket

# --- Static configuration ---
DEFAULT_STT_HOSTNAME = "tc3.local"
DEFAULT_STT_PORT = 8085


def resolve_stt_ip(hostname=DEFAULT_STT_HOSTNAME):
    """
    Resolve hostname to IP at startup (internal DNS cache).
    
    Args:
        hostname: Hostname to resolve
        
    Returns:
        IP address string or hostname if resolution fails
    """
    try:
        ip = socket.gethostbyname(hostname)
        print(f"Resolved {hostname} to {ip}")
        return ip
    except Exception as e:
        print(f"Failed to resolve {hostname}: {e}")
        return hostname  # fallback to hostname if resolution fails


def create_stt_session():
    """
    Create a requests session for connection reuse.
    
    Returns:
        requests.Session object
    """
    return requests.Session()


def transcribe_audio(audio_samples, sample_rate=16000, stt_url=None, prompt=None, model="tiny", session=None):
    """
    Send audio to STT engine and get transcription.
    
    Args:
        audio_samples: numpy array of audio samples
        sample_rate: sample rate of audio
        stt_url: URL of the transcription service
        prompt: Optional hint text to guide transcription
        model: Whisper model to use (tiny, base, small, medium, large)
        session: Optional requests.Session for connection reuse
        
    Returns:
        Transcription text or None if failed
    """
    if session is None:
        session = requests
        
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
        response = session.post(
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
            return result.get('text', '').strip()
        else:
            print(f"STT API error: {response.status_code}")
            if response.text:
                print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None


def benchmark_dns(hostname):
    """
    Benchmark DNS resolution time.
    
    Args:
        hostname: Hostname to resolve
        
    Returns:
        Tuple of (ip_address, elapsed_time_ms)
    """
    start = time.time()
    ip = socket.gethostbyname(hostname)
    elapsed = (time.time() - start) * 1000
    print(f"DNS resolution for {hostname}: {ip} in {elapsed:.2f}ms")
    return ip, elapsed


def benchmark_post(stt_url, files, data, session=None):
    """
    Benchmark POST request time.
    
    Args:
        stt_url: URL to POST to
        files: Files to upload
        data: Form data
        session: Optional requests.Session for connection reuse
        
    Returns:
        Tuple of (response, elapsed_time_ms)
    """
    if session is None:
        sess = requests
    else:
        sess = session
    start = time.time()
    response = sess.post(stt_url, files=files, data=data, timeout=10)
    elapsed = (time.time() - start) * 1000
    use_session = session is not None
    print(f"POST to {stt_url} with{'out' if not use_session else ''} session: {elapsed:.2f}ms, status {response.status_code}")
    return response, elapsed
