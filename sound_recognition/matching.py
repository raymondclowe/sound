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
        Initialize the WordMatcher.
        Args:
            sample_rate (int): Audio sample rate in Hz. Default is 16000.
        """
        self.sample_rate = sample_rate
        self.references = []  # List of dicts: {"mean": ..., "std": ..., "label": ..., "filename": ...}
        
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
    
    def add_reference(self, audio: np.ndarray, word_name: str = "target", filename: str = None) -> None:
        """
        Add a reference sample to the matcher.
        Args:
            audio: Audio samples as numpy array.
            word_name (str): Name/label for the reference word.
            filename (str): Optional filename for traceability.
        """
        mfcc_mean, mfcc_std = self.extract_mfcc(audio)
        self.references.append({
            "mean": mfcc_mean,
            "std": mfcc_std,
            "label": word_name,
            "filename": filename
        })
        print(f"Added reference '{word_name}' (file: {filename}) with MFCC shape: {mfcc_mean.shape}")
        
    def load_reference_from_file(self, filepath: str, word_name: str = "target") -> None:
        """
        Load a single reference word from audio file.
        Args:
            filepath (str): Path to the audio file.
            word_name (str): Name/label for the reference word.
        """
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        self.add_reference(audio, word_name, filename=filepath)

    def load_references_from_files(self, filepaths: list, word_name: str = "target") -> None:
        """
        Load multiple reference samples from a list of audio files.
        Args:
            filepaths (list): List of file paths to audio files.
            word_name (str): Name/label for the reference word.
        """
        for fp in filepaths:
            self.load_reference_from_file(fp, word_name)
        
    def save_reference(self, filepath: str, audio: np.ndarray) -> None:
        """
        Save reference audio to file.
        
        Args:
            filepath (str): Path to save the audio file.
            audio: Audio samples as numpy array.
        """
        sf.write(filepath, audio, self.sample_rate)
        print(f"Reference saved to {filepath}")
        
    def calculate_similarity(self, audio: np.ndarray, reference: dict) -> float:
        """
        Calculate similarity between audio and a reference sample.
        Args:
            audio: Audio samples as numpy array.
            reference: Reference dict with 'mean' and 'std'.
        Returns:
            float: Similarity score (0-100, higher is more similar).
        """
        candidate_mfcc_mean, candidate_mfcc_std = self.extract_mfcc(audio)
        sim_mean = 1 - cosine(reference["mean"], candidate_mfcc_mean)
        sim_std = 1 - cosine(reference["std"], candidate_mfcc_std)
        combined_similarity = (sim_mean * 0.7 + sim_std * 0.3)
        similarity_percent = combined_similarity * 100
        scaled_similarity = (similarity_percent ** 1.5) / (100 ** 0.5)
        return scaled_similarity
    
    def matches(self, audio: np.ndarray, threshold: float = 75) -> Tuple[bool, float, dict]:
        """
        Check if audio matches any reference sample.
        Args:
            audio: Audio samples to check.
            threshold (float): Similarity threshold (0-100). Default 75 for good separation.
        Returns:
            tuple: (matches: bool, best_similarity: float, best_reference: dict)
        """
        if not self.references:
            raise ValueError("No reference samples set. Use add_reference() or load_reference_from_file().")
        best_similarity = -float('inf')
        best_ref = None
        for ref in self.references:
            sim = self.calculate_similarity(audio, ref)
            if sim > best_similarity:
                best_similarity = sim
                best_ref = ref
        return best_similarity >= threshold, best_similarity, best_ref
