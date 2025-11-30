"""
Transcription Module

Provides speech-to-text transcription functionality with support for
external STT services.
"""

import io
import time
import socket
from typing import Optional

import numpy as np
import requests
import soundfile as sf

# --- Static configuration ---
STT_HOSTNAME = "tc3.local"
STT_PORT = 8085


def resolve_stt_ip(hostname: Optional[str] = None) -> str:
    """
    Resolve hostname to IP at startup (internal DNS cache).
    
    Args:
        hostname (str): Hostname to resolve. Defaults to STT_HOSTNAME.
        
    Returns:
        str: Resolved IP address or original hostname if resolution fails.
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


# Create a session for connection reuse
_stt_session = requests.Session()


def transcribe_audio(
    audio_samples: np.ndarray,
    sample_rate: int = 16000,
    stt_url: Optional[str] = None,
    prompt: Optional[str] = None,
    model: str = "tiny"
) -> Optional[str]:
    """
    Send audio to STT engine and get transcription.
    
    Args:
        audio_samples: numpy array of audio samples
        sample_rate (int): sample rate of audio. Default is 16000.
        stt_url (str): URL of the transcription service. If None, uses default.
        prompt (str): Optional hint text to guide transcription.
        model (str): Whisper model to use (tiny, base, small, medium, large).
        
    Returns:
        str: Transcription text or None if failed.
        
    Example:
        >>> audio, sr = librosa.load("speech.wav", sr=16000)
        >>> text = transcribe_audio(audio, sample_rate=sr, model="base")
        >>> print(text)
    """
    if stt_url is None:
        stt_ip = resolve_stt_ip()
        stt_url = f"http://{stt_ip}:{STT_PORT}"
    
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
        response = _stt_session.post(
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
