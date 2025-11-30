"""Word matcher module for MFCC-based audio similarity matching."""

import numpy as np
import librosa
from scipy.spatial.distance import cosine


class WordMatcher:
    """Matches audio clips using MFCC (Mel-Frequency Cepstral Coefficients) similarity."""
    
    def __init__(self, sample_rate=16000):
        """
        Initialize the word matcher.
        
        Args:
            sample_rate: Sample rate of audio to analyze (default 16000)
        """
        self.sample_rate = sample_rate
        self.reference_mfcc_mean = None
        self.reference_mfcc_std = None
        self.reference_word = None
        
    def extract_mfcc(self, audio):
        """
        Extract MFCC features from audio with more discriminative power.
        
        Args:
            audio: Audio samples as numpy array
            
        Returns:
            Tuple of (mfcc_mean, mfcc_std) arrays
        """
        # Extract 20 MFCC coefficients (more detailed)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=512, hop_length=160)
        
        # Use both mean and std deviation for better discrimination
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return mfcc_mean, mfcc_std
    
    def set_reference(self, audio, word_name="target"):
        """
        Set the reference word to match against.
        
        Args:
            audio: Audio samples as numpy array
            word_name: Name/label for the reference word
        """
        self.reference_word = word_name
        self.reference_mfcc_mean, self.reference_mfcc_std = self.extract_mfcc(audio)
        print(f"Reference word '{word_name}' set with MFCC shape: {self.reference_mfcc_mean.shape}")
        
    def load_reference_from_file(self, filepath, word_name="target"):
        """
        Load reference word from audio file.
        
        Args:
            filepath: Path to the audio file
            word_name: Name/label for the reference word
        """
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        self.set_reference(audio, word_name)
        
    def save_reference(self, filepath, audio):
        """
        Save reference audio to file.
        
        Args:
            filepath: Path to save the audio file
            audio: Audio samples as numpy array
        """
        import soundfile as sf
        sf.write(filepath, audio, self.sample_rate)
        print(f"Reference saved to {filepath}")
        
    def calculate_similarity(self, audio):
        """
        Calculate similarity between audio and reference word.
        
        Uses both mean and std of MFCCs for better discrimination.
        
        Args:
            audio: Audio samples to compare
            
        Returns:
            Similarity score (0-100, higher is more similar)
            
        Raises:
            ValueError: If no reference word has been set
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
            Tuple of (matches: bool, similarity: float)
        """
        similarity = self.calculate_similarity(audio)
        return similarity >= threshold, similarity
