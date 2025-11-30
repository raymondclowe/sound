"""
Word Matching Module

Provides MFCC-based word matching functionality for comparing audio clips
against a reference word (Temporal Feature Fingerprinting).
"""

from typing import Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine


class WordMatcher:
    """
    Matches audio clips using MFCC (Mel-Frequency Cepstral Coefficients) similarity.
    
    This class provides functionality to compare audio clips against a reference
    word using MFCC features. It's useful for wake word detection and simple
    speech command recognition.
    
    Attributes:
        sample_rate (int): Audio sample rate in Hz
        reference_mfcc_mean: Mean MFCC features of the reference word
        reference_mfcc_std: Standard deviation of MFCC features
        reference_word (str): Name/label of the reference word
        
    Example:
        >>> matcher = WordMatcher(sample_rate=16000)
        >>> matcher.load_reference_from_file("reference.wav", "hello")
        >>> matches, similarity = matcher.matches(audio_samples, threshold=75)
        >>> if matches:
        ...     print(f"Match found with {similarity:.1f}% similarity")
    """
    
    def __init__(self, sample_rate: int = 16000) -> None:
        """
        Initialize the WordMatcher for multi-profile support.
        Args:
            sample_rate (int): Audio sample rate in Hz. Default is 16000.
        """
        self.sample_rate = sample_rate
        self.references = []  # List of dicts: {name, mfcc_mean, mfcc_std}
        
    def extract_mfcc(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract MFCC features from audio with more discriminative power.
        
        Args:
            audio: Audio samples as numpy array.
            
        Returns:
            tuple: (mfcc_mean, mfcc_std) - Mean and standard deviation of MFCC features.
        """
        # Extract 20 MFCC coefficients (more detailed)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=512, hop_length=160)
        
        # Use both mean and std deviation for better discrimination
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return mfcc_mean, mfcc_std
    
    def add_reference(self, audio: np.ndarray, word_name: str = "target") -> None:
        """
        Add a reference profile for matching.
        Args:
            audio: Audio samples as numpy array.
            word_name (str): Name/label for the reference word.
        """
        mfcc_mean, mfcc_std = self.extract_mfcc(audio)
        self.references.append({
            'name': word_name,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std
        })
        print(f"Reference word '{word_name}' added with MFCC shape: {mfcc_mean.shape}")
        
    def load_reference_from_file(self, filepath: str, word_name: str = "target") -> None:
        """
        Load reference word from audio file and add as a profile.
        Args:
            filepath (str): Path to the audio file.
            word_name (str): Name/label for the reference word.
        """
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        self.add_reference(audio, word_name)
        
    def save_reference(self, filepath: str, audio: np.ndarray) -> None:
        """
        Save reference audio to file.
        
        Args:
            filepath (str): Path to save the audio file.
            audio: Audio samples as numpy array.
        """
        sf.write(filepath, audio, self.sample_rate)
        print(f"Reference saved to {filepath}")
        
    def calculate_similarity(self, audio: np.ndarray, ref) -> float:
        """
        Calculate similarity between audio and a reference profile.
        Args:
            audio: Audio samples as numpy array.
            ref: Reference dict with 'mfcc_mean' and 'mfcc_std'.
        Returns:
            float: Similarity score (0-100, higher is more similar).
        """
        candidate_mfcc_mean, candidate_mfcc_std = self.extract_mfcc(audio)
        sim_mean = 1 - cosine(ref['mfcc_mean'], candidate_mfcc_mean)
        sim_std = 1 - cosine(ref['mfcc_std'], candidate_mfcc_std)
        combined_similarity = (sim_mean * 0.7 + sim_std * 0.3)
        similarity_percent = combined_similarity * 100
        scaled_similarity = (similarity_percent ** 1.5) / (100 ** 0.5)
        return scaled_similarity
    
    def best_match(self, audio: np.ndarray, threshold: float = 75):
        """
        Compare audio to all references, return (best_name, best_similarity, all_scores)
        Args:
            audio: Audio samples to check.
            threshold (float): Similarity threshold (0-100).
        Returns:
            tuple: (best_name, best_score, all_scores)
        """
        if not self.references:
            raise ValueError("No reference profiles loaded.")
        best_name = None
        best_score = -float('inf')
        all_scores = []
        for ref in self.references:
            score = self.calculate_similarity(audio, ref)
            all_scores.append((ref['name'], score))
            if score > best_score:
                best_score = score
                best_name = ref['name']
        return best_name, best_score, all_scores
    def clear_references(self):
        """Remove all reference profiles."""
        self.references = []
