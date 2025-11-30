"""
Silence Detection Module

Provides audio buffering and silence detection functionality using a circular
buffer approach with dynamic threshold adjustment.
"""

from typing import Any, List, Optional

import numpy as np
import sounddevice as sd


class SoundBuffer:
    """
    A circular audio buffer with real-time silence detection.
    
    This class manages continuous audio capture from an input device and provides
    methods to detect silence periods, extract audio segments, and dynamically
    adjust silence thresholds based on ambient noise levels.
    
    Attributes:
        FREQUENCY (int): Sample rate in Hz (16000)
        frame_size (int): Size of each audio frame in samples
        buffer_length (int): Total buffer length in samples
        buffer_seconds (float): Buffer duration in seconds
        silence_threshold (float): Current silence threshold value
        MIN_THRESHOLD (float): Minimum allowed threshold to prevent false positives
    
    Example:
        >>> buffer = SoundBuffer(seconds=10, device=None)
        >>> # Wait for some audio to be captured
        >>> if buffer.is_silent():
        ...     print("Currently silent")
        >>> audio = buffer.return_last_n_seconds(2.0)
        >>> buffer.stop()
    """
    
    frame_size = 0
    buffer_length = 0  # in samples
    buffer_seconds = 0
    sd_stream = None
    data = None
    pointer = 0  # current position in buffer (in samples)
    silence_threshold = 0.02  # Higher initial threshold for more robust detection
    FREQUENCY = 16000
    MIN_THRESHOLD = 0.005  # Minimum threshold to prevent going to 0
    samples_collected = 0
    
    def __init__(self, seconds: int = 10, device: Optional[int] = None) -> None:
        """
        Initialize the sound buffer.
        Prefill buffer with a low-level noise to simulate silence and set a reasonable initial threshold.
        Args:
            seconds (int): Duration of the circular buffer in seconds. Default is 10.
            device: Audio input device ID. None for default device.
        """
        self.buffer_seconds = seconds
        self.buffer_length = self.buffer_seconds * SoundBuffer.FREQUENCY
        # Prefill buffer with low-level noise (simulate silence, e.g., RMS ~0.01)
        self.data = np.random.normal(0, 0.01, self.buffer_length)
        self.samples_collected = self.buffer_length  # Pretend buffer is full for thresholding
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

    def stop(self) -> None:
        """Stop the audio input stream."""
        self.sd_stream.stop()
        
    def start(self) -> None:
        """Start the audio input stream."""
        self.sd_stream.start()
        
    def silent_frames(self) -> List[int]:
        """
        Get indices of frames that are below the silence threshold.
        
        Returns:
            list: List of frame indices that are considered silent.
        """
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
        
    def add_sound_to_buffer(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """
        Callback function for the audio input stream.
        
        This method is called automatically by sounddevice when new audio data
        is available. It adds the data to the circular buffer and dynamically
        adjusts the silence threshold.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time info
            status: Status flags
        """
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
            
    def is_silent(self) -> bool:
        """
        Check if the current audio input is silent.
        Prints debug info about RMS and threshold.
        Returns:
            bool: True if the current audio level is below the silence threshold.
        """
        if len(self.data) == 0 or self.frame_size == 0:
            print("[DEBUG is_silent] No data or frame size 0, returning True")
            return True
        # Get the most recent frame
        recent_samples = self.return_last_n_seconds(0.1)  # Check last 100ms
        if len(recent_samples) == 0:
            print("[DEBUG is_silent] No recent samples, returning True")
            return True
        rms = np.sqrt(np.mean(recent_samples**2))
        print(f"[DEBUG is_silent] RMS: {rms:.6f}, Threshold: {self.silence_threshold:.6f}")
        return rms < self.silence_threshold
    
    def return_last_n_seconds(self, n: float) -> np.ndarray:
        """
        Return the last n seconds of audio from the buffer.
        
        Args:
            n (float): Number of seconds to return.
            
        Returns:
            numpy.ndarray: Audio samples from the last n seconds.
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
