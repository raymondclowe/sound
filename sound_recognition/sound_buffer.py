"""Sound buffer module for audio capture and silence detection."""

import numpy as np
import sounddevice as sd


class SoundBuffer:
    """Circular buffer for capturing and analyzing audio in real-time."""
    
    FREQUENCY = 16000
    MIN_THRESHOLD = 0.005  # Minimum threshold to prevent going to 0
    
    def __init__(self, seconds=10, device=None):
        """
        Initialize the sound buffer.
        
        Args:
            seconds: Length of the buffer in seconds
            device: Audio input device (None for default)
        """
        self.frame_size = 0
        self.buffer_seconds = seconds
        self.buffer_length = self.buffer_seconds * SoundBuffer.FREQUENCY
        self.data = np.zeros(self.buffer_length)
        self.pointer = 0  # current position in buffer (in samples)
        self.silence_threshold = 0.01
        self.samples_collected = 0
        self.sd_stream = sd.InputStream(
            samplerate=16000, 
            channels=1, 
            callback=self.add_sound_to_buffer,
            device=device
        )
        self.sd_stream.start()

    def stop(self):
        """Stop the audio stream."""
        self.sd_stream.stop()
        
    def start(self):
        """Start the audio stream."""
        self.sd_stream.start()
        
    def silent_frames(self):
        """
        Get all frames that are currently silent.
        
        Returns:
            List of frame indices that have RMS values below the silence threshold.
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
        
    def add_sound_to_buffer(self, indata, frames, time, status):
        """
        Callback for adding audio data to the buffer.
        
        This is called by sounddevice when new audio data is available.
        """
        new_data = np.array(indata).flatten()  # flatten to 1D array as we have mono data here
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
        """
        Check if the current audio is silent.
        
        Returns:
            True if the recent audio is below the silence threshold.
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
        """
        Get the last n seconds of audio from the buffer.
        
        Args:
            n: Number of seconds to retrieve
            
        Returns:
            numpy array containing the audio samples
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
